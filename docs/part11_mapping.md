# 21 CFR Part 11 Mapping (Implementation Aids)

| Requirement | Mechanism | Notes |
|---|---|---|
| Audit trail (11.10(e), 11.10(k)) | `compliance.audit` hash‑chained records | Append‑only SQLite, SHA‑256 chain |
| Electronic signatures (11.100, 11.200, 11.300) | `compliance.signatures` RSA sign; `users` roles | Two‑person option; PKI keypair |
| Authority checks (11.10(g)) | RBAC via `users` | Operator/Engineer/QA/Admin |
| Record protection (11.10(c),(d)) | Storage layout + backups | Add your backup job/retention policy |
| Device checks (11.10(h)) | Driver version logs | To be completed per device vendor |

> This table helps validation teams author the IQ/OQ/PQ docs. Certification requires a formal validation package.
