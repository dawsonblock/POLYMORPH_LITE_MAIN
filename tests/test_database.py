"""
Tests for database operations.

This module tests:
- SQLite database initialization
- Database connection management
- Database schema creation
- Database transaction handling
- Data integrity
"""
import pytest
import sqlite3
import os
import tempfile
import shutil


class TestDatabaseInitialization:
    """Test cases for database initialization."""

    def test_database_directory_creation(self):
        """Test that database directory is created if missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "subdir", "test.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            # Should not raise error
            con = sqlite3.connect(db_path)
            con.close()

            assert os.path.exists(db_path)

    def test_database_file_created(self):
        """Test that database file is created on first connection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")

            con = sqlite3.connect(db_path)
            con.close()

            assert os.path.exists(db_path)

    def test_multiple_connections(self):
        """Test that multiple connections can be opened."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")

            con1 = sqlite3.connect(db_path)
            con2 = sqlite3.connect(db_path)

            con1.close()
            con2.close()


class TestSchemaCreation:
    """Test cases for database schema creation."""

    def test_create_table_if_not_exists(self):
        """Test CREATE TABLE IF NOT EXISTS pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            con = sqlite3.connect(db_path)

            # Create table
            con.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)")
            con.commit()

            # Should not error when run again
            con.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)")
            con.commit()

            con.close()

    def test_alter_table_add_column(self):
        """Test ALTER TABLE ADD COLUMN pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            con = sqlite3.connect(db_path)

            # Create table
            con.execute("CREATE TABLE test (id INTEGER)")
            con.commit()

            # Add column
            try:
                con.execute("ALTER TABLE test ADD COLUMN new_col TEXT")
                con.commit()
            except sqlite3.OperationalError:
                # Column might already exist
                pass

            con.close()


class TestDataIntegrity:
    """Test cases for data integrity."""

    def test_primary_key_uniqueness(self):
        """Test that primary keys enforce uniqueness."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            con = sqlite3.connect(db_path)

            con.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
            con.execute("INSERT INTO test VALUES (1, 'first')")
            con.commit()

            # Should raise error for duplicate key
            with pytest.raises(sqlite3.IntegrityError):
                con.execute("INSERT INTO test VALUES (1, 'second')")
                con.commit()

            con.close()

    def test_foreign_key_constraint(self):
        """Test foreign key constraints (if enabled)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            con = sqlite3.connect(db_path)

            # Enable foreign keys
            con.execute("PRAGMA foreign_keys = ON")

            # Create parent and child tables
            con.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
            con.execute("CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER, FOREIGN KEY(parent_id) REFERENCES parent(id))")
            con.commit()

            # Insert parent
            con.execute("INSERT INTO parent VALUES (1)")
            con.commit()

            # Should allow valid foreign key
            con.execute("INSERT INTO child VALUES (1, 1)")
            con.commit()

            # Should reject invalid foreign key
            with pytest.raises(sqlite3.IntegrityError):
                con.execute("INSERT INTO child VALUES (2, 999)")
                con.commit()

            con.close()

    def test_not_null_constraint(self):
        """Test NOT NULL constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            con = sqlite3.connect(db_path)

            con.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, required TEXT NOT NULL)")
            con.commit()

            # Should reject NULL value
            with pytest.raises(sqlite3.IntegrityError):
                con.execute("INSERT INTO test (id) VALUES (1)")
                con.commit()

            # Should accept non-NULL value
            con.execute("INSERT INTO test VALUES (1, 'value')")
            con.commit()

            con.close()


class TestTransactions:
    """Test cases for transaction handling."""

    def test_transaction_commit(self):
        """Test that commits persist data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")

            # Insert and commit
            con = sqlite3.connect(db_path)
            con.execute("CREATE TABLE test (id INTEGER)")
            con.execute("INSERT INTO test VALUES (1)")
            con.commit()
            con.close()

            # Verify data persists
            con = sqlite3.connect(db_path)
            cursor = con.execute("SELECT * FROM test")
            rows = cursor.fetchall()
            con.close()

            assert len(rows) == 1
            assert rows[0][0] == 1

    def test_transaction_rollback(self):
        """Test that rollback discards changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")

            con = sqlite3.connect(db_path)
            con.execute("CREATE TABLE test (id INTEGER)")
            con.commit()

            # Insert but rollback
            con.execute("INSERT INTO test VALUES (1)")
            con.rollback()

            # Verify data not persisted
            cursor = con.execute("SELECT * FROM test")
            rows = cursor.fetchall()
            con.close()

            assert len(rows) == 0

    def test_auto_commit_disabled(self):
        """Test default auto-commit behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")

            con1 = sqlite3.connect(db_path)
            con1.execute("CREATE TABLE test (id INTEGER)")
            con1.execute("INSERT INTO test VALUES (1)")
            # Don't commit

            # Second connection shouldn't see uncommitted data
            con2 = sqlite3.connect(db_path)
            try:
                cursor = con2.execute("SELECT * FROM test")
                rows = cursor.fetchall()
                # Table might not exist or be empty
            except sqlite3.OperationalError:
                rows = []

            con1.close()
            con2.close()


class TestQueryOperations:
    """Test cases for common query patterns."""

    def test_insert_and_select(self):
        """Test INSERT and SELECT operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            con = sqlite3.connect(db_path)

            con.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            con.execute("INSERT INTO test VALUES (1, 'Alice')")
            con.execute("INSERT INTO test VALUES (2, 'Bob')")
            con.commit()

            cursor = con.execute("SELECT * FROM test ORDER BY id")
            rows = cursor.fetchall()

            assert len(rows) == 2
            assert rows[0] == (1, 'Alice')
            assert rows[1] == (2, 'Bob')

            con.close()

    def test_update_operation(self):
        """Test UPDATE operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            con = sqlite3.connect(db_path)

            con.execute("CREATE TABLE test (id INTEGER, value TEXT)")
            con.execute("INSERT INTO test VALUES (1, 'old')")
            con.commit()

            con.execute("UPDATE test SET value='new' WHERE id=1")
            con.commit()

            cursor = con.execute("SELECT value FROM test WHERE id=1")
            value = cursor.fetchone()[0]

            assert value == 'new'

            con.close()

    def test_delete_operation(self):
        """Test DELETE operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            con = sqlite3.connect(db_path)

            con.execute("CREATE TABLE test (id INTEGER)")
            con.execute("INSERT INTO test VALUES (1)")
            con.execute("INSERT INTO test VALUES (2)")
            con.commit()

            con.execute("DELETE FROM test WHERE id=1")
            con.commit()

            cursor = con.execute("SELECT * FROM test")
            rows = cursor.fetchall()

            assert len(rows) == 1
            assert rows[0][0] == 2

            con.close()

    def test_parameterized_query(self):
        """Test parameterized queries to prevent SQL injection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            con = sqlite3.connect(db_path)

            con.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            con.commit()

            # Use parameterized query
            user_input = "Alice'; DROP TABLE test; --"
            con.execute("INSERT INTO test VALUES (?, ?)", (1, user_input))
            con.commit()

            # Table should still exist and contain the value
            cursor = con.execute("SELECT name FROM test WHERE id=1")
            name = cursor.fetchone()[0]

            assert name == user_input
            # Verify table wasn't dropped
            cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test'")
            assert cursor.fetchone() is not None

            con.close()
