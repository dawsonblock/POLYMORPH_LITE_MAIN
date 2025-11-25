import os, sqlite3, bcrypt, json, time
from typing import Optional, Dict, Any
import pyotp
from retrofitkit.compliance.audit import Audit

DB_DIR = os.environ.get("P4_DATA_DIR", "/mnt/data/Polymorph4_Retrofit_Kit_v1/data")
DB = os.path.join(DB_DIR, "system.db")
try:
    os.makedirs(os.path.dirname(DB), exist_ok=True)
except OSError:
    DB_DIR = "data"
    DB = os.path.join(DB_DIR, "system.db")
    os.makedirs(os.path.dirname(DB), exist_ok=True)

class Users:
    def __init__(self):
        self._init()
        self.audit = Audit()

    def _init(self):
        con = sqlite3.connect(DB)
        con.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT, role TEXT, pw BLOB, created REAL)")
        try:
            con.execute("ALTER TABLE users ADD COLUMN mfa_secret TEXT")
        except sqlite3.OperationalError:
            pass
        con.commit()
        con.close()

    def create(self, email: str, name: str, role: str, password: str):
        pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        con = sqlite3.connect(DB)
        con.execute("INSERT OR REPLACE INTO users(email,name,role,pw,created) VALUES(?,?,?,?,?)", (email, name, role, pw, time.time()))
        con.commit()
        con.close()
        self.audit.record("USER_CREATED", "system", email, {"role": role})

    def enable_mfa(self, email: str) -> str:
        """Enable MFA for user and return the secret."""
        secret = pyotp.random_base32()
        con = sqlite3.connect(DB)
        con.execute("UPDATE users SET mfa_secret=? WHERE email=?", (secret, email))
        con.commit()
        con.close()
        self.audit.record("MFA_ENABLED", email, email, {})
        return secret

    def authenticate(self, email: str, password: str, mfa_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        con = sqlite3.connect(DB)
        cur = con.execute("SELECT email,name,role,pw,mfa_secret FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        con.close()
        
        if not row:
            self.audit.record("LOGIN_FAILED", "system", email, {"reason": "user_not_found"})
            return None
            
        if bcrypt.checkpw(password.encode(), row[3]):
            # Check MFA if enabled
            mfa_secret = row[4]
            if mfa_secret:
                if not mfa_token:
                    self.audit.record("LOGIN_FAILED", email, email, {"reason": "mfa_required"})
                    return {"mfa_required": True}
                totp = pyotp.TOTP(mfa_secret)
                if not totp.verify(mfa_token):
                    self.audit.record("LOGIN_FAILED", email, email, {"reason": "invalid_mfa"})
                    return None

            self.audit.record("LOGIN_SUCCESS", email, email, {"role": row[2]})
            return {"email": row[0], "name": row[1], "role": row[2]}
            
        self.audit.record("LOGIN_FAILED", "system", email, {"reason": "invalid_password"})
        return None
