import sqlite3
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional
import tempfile
import shutil


@dataclass
class DocumentRecord:
    """Represents a indexed document with its chunk range and metadata."""
    doc_id: int
    filename: str
    display_name: str
    doc_type: str
    indexed_at: str  # ISO 8601 timestamp
    chunk_start: int
    chunk_end: int
    chunk_count: int
    page_count: int
    weight: float


class DocumentRegistry:
    """SQLite-backed registry for tracking indexed documents and their chunk ranges."""
    
    def __init__(self, artifacts_dir: pathlib.Path):
        """
        Initialize the document registry.
        
        Args:
            artifacts_dir: Path to directory where document_registry.db will be stored
        """
        self.artifacts_dir = pathlib.Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.artifacts_dir / "document_registry.db"
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column-by-name access
        self._create_table()
    
    def _create_table(self):
        """Create the documents table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                doc_type TEXT DEFAULT 'document',
                indexed_at TEXT NOT NULL,
                chunk_start INTEGER NOT NULL,
                chunk_end INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                page_count INTEGER DEFAULT 0,
                weight REAL DEFAULT 1.0
            )
        """)
        self.conn.commit()
    
    def clear(self):
        """Delete all rows from the documents table."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents")
        self.conn.commit()
    
    def add_document(
        self,
        filename: str,
        display_name: str,
        doc_type: str,
        chunk_start: int,
        chunk_end: int,
        chunk_count: int,
        page_count: int,
        weight: float = 1.0
    ) -> DocumentRecord:
        """
        Add a document to the registry.
        
        Args:
            filename: Original filename (e.g., "textbook--extracted_markdown.md")
            display_name: Human-readable name
            doc_type: Document type (e.g., "document", "slides")
            chunk_start: Index of first chunk for this document
            chunk_end: Index of last chunk for this document
            chunk_count: Total number of chunks in this document
            page_count: Total number of pages
            weight: Weighting factor for ranking (default 1.0)
        
        Returns:
            DocumentRecord: The inserted document record
        """
        indexed_at = datetime.now(timezone.utc).isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO documents
            (filename, display_name, doc_type, indexed_at, chunk_start, chunk_end, chunk_count, page_count, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (filename, display_name, doc_type, indexed_at, chunk_start, chunk_end, chunk_count, page_count, weight))
        self.conn.commit()
        
        doc_id = cursor.lastrowid
        return DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            display_name=display_name,
            doc_type=doc_type,
            indexed_at=indexed_at,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            chunk_count=chunk_count,
            page_count=page_count,
            weight=weight,
        )
    
    def get_all(self) -> List[DocumentRecord]:
        """
        Retrieve all documents ordered by chunk_start.
        
        Returns:
            List of DocumentRecord objects
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY chunk_start ASC")
        rows = cursor.fetchall()
        
        records = []
        for row in rows:
            records.append(DocumentRecord(
                doc_id=row['doc_id'],
                filename=row['filename'],
                display_name=row['display_name'],
                doc_type=row['doc_type'],
                indexed_at=row['indexed_at'],
                chunk_start=row['chunk_start'],
                chunk_end=row['chunk_end'],
                chunk_count=row['chunk_count'],
                page_count=row['page_count'],
                weight=row['weight'],
            ))
        
        return records
    
    def get_by_chunk_id(self, chunk_id: int) -> Optional[DocumentRecord]:
        """
        Find the document that contains the given chunk_id.
        
        A document contains chunk_id if: chunk_start <= chunk_id <= chunk_end
        
        Args:
            chunk_id: The chunk index to look up
        
        Returns:
            DocumentRecord if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM documents
            WHERE chunk_start <= ? AND chunk_end >= ?
            LIMIT 1
        """, (chunk_id, chunk_id))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return DocumentRecord(
            doc_id=row['doc_id'],
            filename=row['filename'],
            display_name=row['display_name'],
            doc_type=row['doc_type'],
            indexed_at=row['indexed_at'],
            chunk_start=row['chunk_start'],
            chunk_end=row['chunk_end'],
            chunk_count=row['chunk_count'],
            page_count=row['page_count'],
            weight=row['weight'],
        )
    
    def get_by_filename(self, filename: str) -> Optional[DocumentRecord]:
        """
        Find a document by filename.
        
        Args:
            filename: The filename to search for
        
        Returns:
            DocumentRecord if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE filename = ? LIMIT 1", (filename,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return DocumentRecord(
            doc_id=row['doc_id'],
            filename=row['filename'],
            display_name=row['display_name'],
            doc_type=row['doc_type'],
            indexed_at=row['indexed_at'],
            chunk_start=row['chunk_start'],
            chunk_end=row['chunk_end'],
            chunk_count=row['chunk_count'],
            page_count=row['page_count'],
            weight=row['weight'],
        )
    
    def set_weight(self, filename: str, weight: float):
        """
        Update the weight for a document.
        
        Args:
            filename: The document filename
            weight: The new weight value
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE documents SET weight = ? WHERE filename = ?",
            (weight, filename)
        )
        self.conn.commit()
    
    def get_weight_map(self) -> Dict[int, float]:
        """
        Generate a mapping of chunk_id to weight across all documents.
        
        For each document, every chunk in the range [chunk_start, chunk_end]
        gets the document's weight.
        
        Returns:
            Dict mapping chunk_id to weight
        """
        weight_map = {}
        documents = self.get_all()
        
        for doc in documents:
            for chunk_id in range(doc.chunk_start, doc.chunk_end + 1):
                weight_map[chunk_id] = doc.weight
        
        return weight_map
    
    def print_summary(self):
        """Print a formatted table of all documents with key metadata."""
        documents = self.get_all()
        
        if not documents:
            print("No documents registered.")
            return
        
        # Print header
        header = f"{'Doc ID':<8} {'Display Name':<30} {'Type':<12} {'Chunks':<10} {'Pages':<8} {'Weight':<8} {'Indexed At':<25}"
        print(header)
        print("-" * len(header))
        
        # Print rows
        for doc in documents:
            # Truncate display_name if too long
            display_name = doc.display_name[:27] + "..." if len(doc.display_name) > 30 else doc.display_name
            indexed_at = doc.indexed_at[:19]  # Show only date and time, not timezone
            
            row = f"{doc.doc_id:<8} {display_name:<30} {doc.doc_type:<12} {doc.chunk_count:<10} {doc.page_count:<8} {doc.weight:<8.2f} {indexed_at:<25}"
            print(row)
        
        print("-" * len(header))
        print(f"Total: {len(documents)} document(s)")


if __name__ == "__main__":
    print("Testing DocumentRegistry...\n")
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="tokensmith_test_")
    print(f"Using temp directory: {temp_dir}\n")
    
    try:
        # Create registry
        registry = DocumentRegistry(temp_dir)
        print("✓ Registry created")
        
        # Add first document
        doc1 = registry.add_document(
            filename="textbook--extracted_markdown.md",
            display_name="Database Textbook",
            doc_type="document",
            chunk_start=0,
            chunk_end=99,
            chunk_count=100,
            page_count=50,
            weight=1.0
        )
        print(f"✓ Added document 1: {doc1.display_name} (chunks 0-99)")
        
        # Add second document
        doc2 = registry.add_document(
            filename="slides--extracted_markdown.md",
            display_name="Database Slides",
            doc_type="slides",
            chunk_start=100,
            chunk_end=199,
            chunk_count=100,
            page_count=30,
            weight=0.8
        )
        print(f"✓ Added document 2: {doc2.display_name} (chunks 100-199)\n")
        
        # Test get_by_chunk_id with first document
        result1 = registry.get_by_chunk_id(50)
        assert result1 is not None, "Failed to find document for chunk 50"
        assert result1.doc_id == doc1.doc_id, "Wrong document returned for chunk 50"
        print(f"✓ get_by_chunk_id(50) returned: {result1.display_name}")
        
        # Test get_by_chunk_id with second document
        result2 = registry.get_by_chunk_id(150)
        assert result2 is not None, "Failed to find document for chunk 150"
        assert result2.doc_id == doc2.doc_id, "Wrong document returned for chunk 150"
        print(f"✓ get_by_chunk_id(150) returned: {result2.display_name}\n")
        
        # Test get_all
        all_docs = registry.get_all()
        print(f"✓ get_all() returned {len(all_docs)} document(s)\n")
        
        # Print summary
        print("Document Summary:")
        registry.print_summary()
        print()
        
        # Test get_weight_map
        weight_map = registry.get_weight_map()
        print(f"✓ Weight map has {len(weight_map)} entries")
        print(f"  Sample: chunk_id=50 → weight={weight_map.get(50)}")
        print(f"  Sample: chunk_id=150 → weight={weight_map.get(150)}\n")
        
        print("✓ All tests passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up temp directory")
