# Risk Assessment (RA)

**System:** POLYMORPH LITE
**Date:** 2025-11-29

## 1. Methodology
Risks are assessed based on Severity (S), Probability (P), and Detectability (D).
Risk Priority Number (RPN) = S * P * D.

## 2. Risk Register

| ID | Hazard | Consequence | Severity (1-5) | Probability (1-5) | Detectability (1-5) | RPN | Mitigation | Residual Risk |
|----|--------|-------------|----------------|-------------------|---------------------|-----|------------|---------------|
| RA-01 | Data Loss | Loss of experimental results | 5 | 2 | 2 | 20 | Database backups (RDS automated), RAID storage | Low |
| RA-02 | Unauthorized Access | Data manipulation | 5 | 2 | 3 | 30 | RBAC, Strong Passwords, MFA, Audit Logs | Low |
| RA-03 | Laser Safety Failure | Eye injury | 5 | 1 | 1 | 5 | Hardware Interlocks (Door, E-Stop), Warning Lights | Low |
| RA-04 | Audit Trail Failure | Regulatory non-compliance | 4 | 2 | 2 | 16 | Immutable logs, Database constraints | Low |
| RA-05 | AI Hallucination | Incorrect experiment parameters | 3 | 3 | 2 | 18 | Human-in-the-loop review step, Confidence scores | Medium |

## 3. Conclusion
All identified high risks have been mitigated to an acceptable level.
