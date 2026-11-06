"""SQLite chunk storage with an FTS5 keyword index (no vectors)."""
import re
import sqlite3

import config


def connect():
    config.DATA_DIR.mkdir(exist_ok=True)
    db = sqlite3.connect(config.DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS chunks(
            id TEXT PRIMARY KEY, source TEXT, page INTEGER, type TEXT, image_path TEXT, text TEXT);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='rowid');
    """)
    return db


def reset(db):
    db.executescript("DELETE FROM chunks; INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all');")


def add(db, chunks):
    db.executemany(
        "INSERT OR REPLACE INTO chunks(id,source,page,type,image_path,text) VALUES(:id,:source,:page,:type,:image_path,:text)",
        chunks)
    db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    db.commit()


def get(db, ids):
    out = []
    for i in ids:
        r = db.execute("SELECT * FROM chunks WHERE id=?", (i,)).fetchone()
        if r:
            out.append(dict(r))
    return out


def all_chunks(db):
    return [dict(r) for r in db.execute("SELECT * FROM chunks ORDER BY rowid")]


def search(db, question, limit):
    """Keyword search: OR of the question's words, ranked by bm25."""
    words = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 2]
    if not words:
        return []
    q = " OR ".join(f'"{w}"' for w in words)
    rows = db.execute(
        "SELECT c.* FROM chunks_fts f JOIN chunks c ON c.rowid=f.rowid WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (q, limit)).fetchall()
    return [dict(r) for r in rows]
