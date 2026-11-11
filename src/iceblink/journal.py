"""Local SQLite event journal and transactional, monotonic run stops."""
from __future__ import annotations

from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from .core import Mode

NETWORK_FILESYSTEMS = {"nfs", "nfs4", "lustre", "cifs", "smbfs", "sshfs", "fuse.sshfs"}


def filesystem_type(path):
    resolved = Path(path).resolve()
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        mount = Path(fields[1].replace("\\040", " "))
        if resolved == mount or mount in resolved.parents:
            mounts.append((len(str(mount)), fields[2]))
    return max(mounts)[1] if mounts else "unknown"


class Journal:
    def __init__(self, path, *, create=True):
        self.path = Path(path).resolve()
        if filesystem_type(self.path) in NETWORK_FILESYSTEMS:
            raise ValueError("SQLite WAL requires node-local storage; use online backup for network storage")
        if not create and not self.path.exists():
            raise FileNotFoundError("Journal missing; guarded state cannot be reconstructed")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), timeout=2, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events(
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
            wall_ns INTEGER NOT NULL, monotonic_ns INTEGER NOT NULL,
            run_id TEXT, request_id TEXT, tool_call_id TEXT,
            component TEXT NOT NULL, version TEXT NOT NULL, kind TEXT NOT NULL,
            status TEXT NOT NULL, payload TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS events_run ON events(run_id, sequence);
          CREATE TABLE IF NOT EXISTS runs(
            id TEXT PRIMARY KEY, policy TEXT NOT NULL, maximum INTEGER NOT NULL,
            model_stop INTEGER NOT NULL DEFAULT 0, tool_stop INTEGER NOT NULL DEFAULT 0,
            ended INTEGER NOT NULL DEFAULT 0, degraded INTEGER NOT NULL DEFAULT 0,
            capability TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS requests(
            id TEXT PRIMARY KEY, execution TEXT UNIQUE NOT NULL, run_id TEXT NOT NULL REFERENCES runs(id),
            created_ns INTEGER NOT NULL, completed_ns INTEGER, status TEXT NOT NULL,
            required TEXT NOT NULL, capability TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS warnings(
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), request_id TEXT NOT NULL,
            evidence TEXT NOT NULL, status TEXT NOT NULL, delivered_request TEXT);
          CREATE TABLE IF NOT EXISTS actions(
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), request_id TEXT,
            mode INTEGER NOT NULL, status TEXT NOT NULL, evidence TEXT NOT NULL);
        """)
        os.chmod(self.path, 0o600)
        st = self.path.stat()
        self._file_identity = (st.st_dev, st.st_ino)
        schema = self.db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
        if schema and schema[0] != "1":
            raise ValueError("Unsupported journal schema")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('schema','1')")

    @contextmanager
    def transaction(self):
        with self.lock:
            if self.db.in_transaction:
                savepoint = "nested_" + uuid.uuid4().hex
                self.db.execute(f"SAVEPOINT {savepoint}")
                try:
                    yield self.db
                except BaseException:
                    self.db.execute(f"ROLLBACK TO {savepoint}")
                    raise
                finally:
                    self.db.execute(f"RELEASE {savepoint}")
                return
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def event(self, kind, *, run_id=None, request_id=None, tool_call_id=None,
              component="monitor", version="0.1.0", status="recorded", payload=None, event_id=None):
        eid = event_id or uuid.uuid4().hex
        # Callers pass whitelisted metadata. Never pass user strings/exceptions.
        encoded = json.dumps(payload or {}, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > 128 * 1024:
            raise ValueError("Event metadata exceeds budget")
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO events VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?)",
                            (eid, time.time_ns(), time.monotonic_ns(), run_id, request_id, tool_call_id,
                             component, version, kind, status, encoded))
        return eid

    def rows(self, query, parameters=()):
        with self.lock:
            return [dict(row) for row in self.db.execute(query, parameters)]

    def run(self, run_id):
        rows = self.rows("SELECT * FROM runs WHERE id=?", (run_id,))
        if not rows:
            raise KeyError("Unknown run")
        return rows[0]

    def stop(self, run_id, mode, *, source, request_id=None, evidence=()):
        mode = Mode.parse(mode)
        if mode < Mode.c:
            raise ValueError("A stop requires c or d")
        with self.transaction() as db:
            row = db.execute("SELECT maximum FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None or mode > row[0]:
                raise ValueError("Stop exceeds run authorization")
            db.execute("UPDATE runs SET model_stop=1,tool_stop=MAX(tool_stop,?) WHERE id=?",
                       (int(mode == Mode.d), run_id))
            action_id = uuid.uuid4().hex
            db.execute("INSERT INTO actions VALUES(?,?,?,?,?,?)",
                       (action_id, run_id, request_id, int(mode), "pending", json.dumps(evidence)))
            self.event("stop_requested", run_id=run_id, request_id=request_id,
                       component=source, payload={"mode": mode.name, "action_id": action_id,
                                                  "evidence": list(evidence)})
        return action_id

    def clear(self, run_id):
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM actions WHERE run_id=? AND status='pending'", (run_id,)).fetchone():
                raise ValueError("Stop action is still pending")
            self.run(run_id)
            db.execute("UPDATE runs SET model_stop=0,tool_stop=0,degraded=0 WHERE id=?", (run_id,))
            self.event("stop_cleared", run_id=run_id, component="operator")

    def heartbeat(self):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES('heartbeat',?)", (str(time.monotonic_ns()),))

    def health(self, run_id):
        st = self.path.stat()
        if (st.st_dev, st.st_ino) != self._file_identity:
            raise OSError("Journal file was replaced")
        heartbeat = self.rows("SELECT value FROM metadata WHERE key='heartbeat'")
        outstanding = self.rows("SELECT MIN(created_ns) AS oldest,COUNT(*) AS count FROM requests "
                                "WHERE run_id=? AND status='pending'", (run_id,))[0]
        oldest = self.rows("SELECT id FROM requests WHERE run_id=? AND status='pending' ORDER BY created_ns LIMIT 1", (run_id,))
        return {"heartbeat_ns": int(heartbeat[0]["value"]) if heartbeat else 0, **outstanding,
                "oldest_request_id": oldest[0]["id"] if oldest else None, "degraded": bool(self.run(run_id)["degraded"])}

    def events(self, after=0, run_id=None, limit=1000):
        limit = min(max(int(limit), 1), 10000)
        query = "SELECT * FROM events WHERE sequence>?"
        params = [after]
        if run_id is not None:
            query += " AND run_id=?"
            params.append(run_id)
        query += " ORDER BY sequence LIMIT ?"
        params.append(limit)
        rows = self.rows(query, params)
        for row in rows:
            row["payload"] = json.loads(row["payload"])
        return rows

    def backup(self, destination):
        """Online backup includes committed WAL contents. Safe for export to NFS."""
        destination = Path(destination)
        if destination.resolve() == self.path:
            raise ValueError("Backup destination equals live journal")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
        with self.lock, closing(sqlite3.connect(temp)) as target:
            self.db.backup(target)
            target.execute("PRAGMA journal_mode=DELETE")
        os.chmod(temp, 0o600)
        temp.replace(destination)

    def prune(self, before_wall_ns):
        # Stops, actions and warning delivery state survive event retention.
        with self.transaction() as db:
            removed = db.execute("DELETE FROM events WHERE wall_ns<?", (before_wall_ns,)).rowcount
            db.execute("DELETE FROM requests WHERE run_id IN (SELECT id FROM runs WHERE ended=1) "
                       "AND status!='pending' AND completed_ns<?", (before_wall_ns,))
        return removed

    def close(self):
        with self.lock:
            self.db.close()
