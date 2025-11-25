import os, sqlite3, time, json
from retrofitkit.metrics.exporter import Metrics

DB_DIR = os.environ.get("P4_DATA_DIR", "/mnt/data/Polymorph4_Retrofit_Kit_v1/data")
DB = os.path.join(DB_DIR, "system.db")
# Ensure dir exists (might be redundant if users.py runs first, but safe)
try:
    os.makedirs(os.path.dirname(DB), exist_ok=True)
except OSError:
    DB_DIR = "data"
    DB = os.path.join(DB_DIR, "system.db")
    os.makedirs(os.path.dirname(DB), exist_ok=True)
REQUIRED_ROLES = ["Operator", "QA"]
mx = Metrics.get()

def _ensure():
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS approvals (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, recipe_path TEXT, requested_by TEXT, status TEXT, approvals_json TEXT)")
    con.commit(); con.close()

def _update_metrics():
    # simple count of pending approvals
    con = sqlite3.connect(DB)
    n = con.execute("SELECT COUNT(*) FROM approvals WHERE status='PENDING'").fetchone()[0]
    con.close()
    mx.set("polymorph_approvals_pending", float(n))

def request(recipe_path: str, requested_by: str) -> int:
    _ensure()
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO approvals(ts,recipe_path,requested_by,status,approvals_json) VALUES(?,?,?,?,?)",
                (time.time(), recipe_path, requested_by, "PENDING", "[]"))
    rid = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.commit(); con.close()
    _update_metrics()
    return rid

def list_pending(limit=200):
    _ensure()
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT id,ts,recipe_path,requested_by,status,approvals_json FROM approvals ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    out = []
    for r in rows:
        out.append({"id": r[0], "ts": r[1], "recipe_path": r[2], "requested_by": r[3], "status": r[4], "approvals": json.loads(r[5] or "[]")})
    _update_metrics()
    return out

def approve(req_id: int, email: str, role: str):
    _ensure()
    con = sqlite3.connect(DB)
    row = con.execute("SELECT approvals_json,status FROM approvals WHERE id=?", (req_id,)).fetchone()
    if not row:
        con.close(); raise ValueError("Request not found")
    if row[1] != "PENDING":
        con.close(); _update_metrics(); return
    approvals = json.loads(row[0] or "[]")
    if any(a.get("email")==email for a in approvals):
        con.close(); _update_metrics(); return
    approvals.append({"email": email, "role": role, "ts": time.time()})
    have_roles = set(a.get("role") for a in approvals)
    status = "APPROVED" if all(r in have_roles for r in REQUIRED_ROLES) else "PENDING"
    con.execute("UPDATE approvals SET approvals_json=?, status=? WHERE id=?", (json.dumps(approvals), status, req_id))
    con.commit(); con.close()
    _update_metrics()
