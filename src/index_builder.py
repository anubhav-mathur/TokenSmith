#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata

Entry point (called by main.py):
    build_index(markdown_files, cfg, keep_tables=True, do_visualize=False)
"""

import os
import pickle
import pathlib
import re
import json
from typing import List, Dict, Optional

import faiss
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer
from src.document_registry import DocumentRegistry

from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Default keywords to exclude sections
DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']

# ------------------------ Section Extraction with Fallbacks ----------------------

def extract_sections_with_fallback(markdown_file: str) -> List[Dict]:
    """
    Extract sections from markdown with fallback strategies.
    
    1. Try numbered heading extractor (e.g., "## 1.2.3 Title")
    2. If < 5 sections, fall back to any ## heading
    3. If still < 5, fall back to page-based splitting on "--- Page" markers
    
    Args:
        markdown_file: Path to markdown file
    
    Returns:
        List of section dictionaries with 'heading' and 'content' keys
    """
    # Strategy 1: Try numbered headings (existing extractor)
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )
    
    if len(sections) >= 5:
        return sections
    
    print(f"  ⚠ Numbered heading extraction returned only {len(sections)} sections. Trying fallback...")
    
    # Strategy 2: Fall back to any ## heading
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading file: {e}")
        return sections if sections else []
    
    # Split on any ## heading
    heading_pattern = r'(?=^## )'
    chunks = re.split(heading_pattern, content, flags=re.MULTILINE)
    
    fallback_sections = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        
        parts = chunk.split('\n', 1)
        heading = parts[0].strip().lstrip('#').strip()
        section_content = parts[1].strip() if len(parts) > 1 else ''
        
        # Skip excluded sections
        if any(kw.lower() in heading.lower() for kw in DEFAULT_EXCLUSION_KEYWORDS):
            continue
        
        if section_content:
            fallback_sections.append({
                'heading': heading if heading else 'Untitled',
                'content': section_content,
                'level': 1,
                'chapter': 0
            })
    
    if len(fallback_sections) >= 5:
        print(f"  ✓ Fallback strategy 2 (any ## heading) returned {len(fallback_sections)} sections")
        return fallback_sections
    
    print(f"  ⚠ Fallback strategy 2 returned only {len(fallback_sections)} sections. Trying page-based split...")
    
    # Strategy 3: Fall back to page-based splitting on --- Page markers
    page_pattern = r'--- Page (\d+) ---'
    page_sections = re.split(page_pattern, content)
    
    fallback_page_sections = []
    if page_sections[0].strip():
        fallback_page_sections.append({
            'heading': 'Page 0',
            'content': page_sections[0].strip(),
            'level': 1,
            'chapter': 0
        })
    
    # Process pairs of (page_num, content)
    for i in range(1, len(page_sections), 2):
        try:
            page_num = page_sections[i]
            page_content = page_sections[i+1].strip() if i+1 < len(page_sections) else ""
            
            if page_content:
                fallback_page_sections.append({
                    'heading': f'Page {page_num}',
                    'content': page_content,
                    'level': 1,
                    'chapter': 0
                })
        except (IndexError, ValueError):
            continue
    
    if fallback_page_sections:
        print(f"  ✓ Fallback strategy 3 (page-based) returned {len(fallback_page_sections)} sections")
        return fallback_page_sections
    
    # If all strategies fail, return what we got from strategy 1
    print(f"  ⚠ All fallback strategies exhausted, returning {len(sections)} sections from primary extraction")
    return sections

# ------------------------ Main index builder ----------------------------- 

def build_index(
    markdown_files: List[str],
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    registry: Optional[DocumentRegistry] = None,
    use_multiprocessing: bool = False,
    use_headings: bool = False
) -> None:
    """
    Extract sections from multiple markdown files, chunk, embed, and build 
    both FAISS and BM25 indexes with document tracking.

    Args:
        markdown_files: List of markdown file paths to index
        chunker: DocumentChunker instance
        chunk_config: ChunkConfig instance
        embedding_model_path: Path to embedding model
        artifacts_dir: Directory for index artifacts
        index_prefix: Prefix for generated files
        registry: Optional DocumentRegistry for tracking documents
        use_multiprocessing: Enable multiprocessing for embeddings
        use_headings: Include headings in chunk text

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
    """
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []
    
    # Convert artifacts_dir to Path
    artifacts_dir = pathlib.Path(artifacts_dir)

    page_to_chunk_ids = {}
    total_chunks = 0
    
    # Process each markdown file
    for file_idx, markdown_file in enumerate(markdown_files):
        print(f"\n[File {file_idx + 1}/{len(markdown_files)}] Processing: {markdown_file}")
        
        chunk_start = total_chunks  # Track start of this document's chunks
        
        # Extract sections with fallback strategies
        sections = extract_sections_with_fallback(markdown_file)
        print(f"  Extracted {len(sections)} sections")
        
        current_page = 1
        heading_stack = []
        page_count = 1  # Track pages for this document
        
        # Step 1: Chunk using DocumentChunker
        for i, c in enumerate(sections):
            # Determine current section level
            current_level = c.get('level', 1)
            chapter_num = c.get('chapter', 0)
            
            # Pop sections that are deeper or siblings
            while heading_stack and heading_stack[-1][0] >= current_level:
                heading_stack.pop()
            
            # Push pair of (level, heading)
            if c['heading'] != "Introduction":
                heading_stack.append((current_level, c['heading']))
            
            # Construct section path
            path_list = [h[1] for h in heading_stack]
            full_section_path = " ".join(path_list)
            full_section_path = f"Chapter {chapter_num} " + full_section_path
            
            # Use DocumentChunker to recursively split this section
            sub_chunks = chunker.chunk(c['content'])
            
            # Regex to find page markers like "--- Page 3 ---"
            page_pattern = re.compile(r'--- Page (\d+) ---')
            
            # Iterate through each chunk produced from this section
            for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
                # Track all pages this specific chunk touches
                chunk_pages = set()
                
                # Split the sub_chunk by page markers
                fragments = page_pattern.split(sub_chunk)
                
                # If there is content before the first page marker,
                # it belongs to the current_page.
                if fragments[0].strip():
                    page_to_chunk_ids.setdefault(current_page, set()).add(total_chunks + sub_chunk_id)
                    chunk_pages.add(current_page)
                
                # Process the new pages found within this sub_chunk
                for j in range(1, len(fragments), 2):
                    try:
                        new_page = int(fragments[j]) + 1
                        if fragments[j+1].strip():
                            page_to_chunk_ids.setdefault(new_page, set()).add(total_chunks + sub_chunk_id)
                            chunk_pages.add(new_page)
                        
                        current_page = new_page
                        page_count = max(page_count, new_page)
                    except (IndexError, ValueError):
                        continue
                
                # Clean sub_chunk by removing page markers
                clean_chunk = re.sub(page_pattern, '', sub_chunk).strip()
                
                # Skip introduction chunks for embedding
                if c["heading"] == "Introduction":
                    continue
                
                # Prepare metadata
                meta = {
                    "filename": markdown_file,
                    "mode": chunk_config.to_string(),
                    "char_len": len(clean_chunk),
                    "word_len": len(clean_chunk.split()),
                    "section": c['heading'],
                    "section_path": full_section_path,
                    "text_preview": clean_chunk[:100],
                    "page_numbers": sorted(list(chunk_pages)),
                    "chunk_id": total_chunks + sub_chunk_id
                }
                
                # Prepare chunk with prefix
                if use_headings:
                    chunk_prefix = (
                        f"Description: {full_section_path} "
                        f"Content: "
                    )
                else:
                    chunk_prefix = ""
                
                all_chunks.append(chunk_prefix + clean_chunk)
                sources.append(markdown_file)
                metadata.append(meta)
            
            total_chunks += len(sub_chunks)
        
        # Track document in registry
        chunk_end = total_chunks - 1
        # Only register if document produced at least one chunk
        if registry and chunk_end >= chunk_start:
            # Infer display name from filename
            stem = pathlib.Path(markdown_file).stem
            # Remove --extracted_markdown suffix if present
            if stem.endswith('--extracted_markdown'):
                stem = stem[:-len('--extracted_markdown')]
            # Replace underscores with spaces and title case
            display_name = stem.replace('_', ' ').title()
            
            # Infer doc_type from filename
            stem_lower = stem.lower()
            if 'slide' in stem_lower:
                doc_type = 'slides'
            elif 'paper' in stem_lower:
                doc_type = 'paper'
            else:
                doc_type = 'document'
            
            registry.add_document(
                filename=markdown_file,
                display_name=display_name,
                doc_type=doc_type,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                chunk_count=(chunk_end - chunk_start + 1),
                page_count=page_count
            )
            print(f"  ✓ Registered: chunks {chunk_start}-{chunk_end}, {page_count} pages")
        elif registry:
            print(f"  ⚠ Skipped {markdown_file}: no chunks after filtering")
    
    # Convert the sets to sorted lists for clean output
    final_map = {}
    for page, id_set in page_to_chunk_ids.items():
        final_map[page] = sorted(list(id_set))
    
    output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"\nSaved page to chunk ID map: {output_file}")
    
    # Step 2: Create embeddings for FAISS index
    print(f"Embedding {len(all_chunks):,} chunks with {pathlib.Path(embedding_model_path).stem} ...")
    embedder = SentenceTransformer(embedding_model_path)
    
    if use_multiprocessing:
        print("Starting multi-process pool for embeddings...")
        pool = embedder.start_multi_process_pool(workers=4)
        try:
            embeddings = embedder.encode_multi_process(
                all_chunks,
                pool,
                batch_size=32
            )
        finally:
            embedder.stop_multi_process_pool(pool)
    else:
        embeddings = embedder.encode(
            all_chunks,
            batch_size=8,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    # Step 3: Build FAISS index
    print(f"Building FAISS index for {len(all_chunks):,} chunks...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
    print(f"FAISS Index built successfully: {index_prefix}.faiss")
    
    # Step 4: Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 Index built successfully: {index_prefix}_bm25.pkl")
    
    # Step 5: Dump index artifacts
    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved all index artifacts with prefix: {index_prefix}")

# ------------------------ Helper functions ------------------------------

def preprocess_for_bm25(text: str) -> list[str]:
    """
    Simplifies text to keep only letters, numbers, underscores, hyphens,
    apostrophes, plus, and hash — suitable for BM25 tokenization.
    """
    # Convert to lowercase
    text = text.lower()

    # Keep only allowed characters
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)

    # Split by whitespace
    tokens = text.split()

    return tokens
