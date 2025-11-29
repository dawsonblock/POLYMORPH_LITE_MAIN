# Standard Operating Procedure: Electronic Signatures
## POLYMORPH_LITE v4.0 - LabOS Polymorph Edition

**SOP ID**: SOP-ES-002  
**Version**: 4.0  
**Effective Date**: 2024-11-29  
**Review Frequency**: Annual

---

## 1. Purpose

This SOP establishes procedures for the creation, use, and verification of electronic signatures in POLYMORPH_LITE to ensure compliance with 21 CFR Part 11.211 - Electronic Signatures.

## 2. Scope

Applies to all electronic signature activities including:
- Workflow approvals
- Data review and approval
- Deviation investigations
- Change control approvals
- Batch record releases

## 3. Responsibilities

| Role | Responsibility |
|------|----------------|
| **System Owner** | Ensure signature system integrity |
| **QA Manager** | Verify signature compliance |
| **IT Administrator** | Maintain signature infrastructure |
| **Authorized Signers** | Proper use of electronic signatures |

## 4. Electronic Signature Components

Per 21 CFR 11.50, electronic signatures must include:
1. **Printed name** of the signer
2. **Date and time** of signature
3. **Meaning** of signature (e.g., reviewed, approved)

Additionally, POLYMORPH_LITE captures:
- User ID
- IP address
- Session ID
- Cryptographic hash of signed data

## 5. Procedures

### 5.1 Electronic Signature Setup

**Initial Configuration:**
1. User completes signature training
2. User creates unique signature credentials:
   - Username (cannot be shared)
   - Password (meets complexity requirements)
3. User signs Signature Agreement Form (ES-001)
4. System administrator activates signature capability
5. Signature linked to user identity in system

**Signature Meaning Assignment:**
- System defines signature meanings (reviewed, approved, witnessed)
- User can only sign with assigned meanings
- Signature meaning displayed at time of signing

### 5.2 Applying Electronic Signatures

**Standard Signature Process:**
1. User navigates to record requiring signature
2. System displays data to be signed
3. User clicks "Sign" button
4. Signature dialog appears with:
   - Username field (pre-populated, read-only)
   - Password field (entry required)
   - Signature meaning (dropdown)
   - Reason for signing (text field)
5. User enters password and reason
6. User confirms signature
7. System verifies credentials
8. Signature applied and locked

**Signature Verification:**
- Password verified against user account
- User authorized for signature meaning
- No concurrent signatures on same record
- Signature hash generated and stored

### 5.3 Multi-Level Signatures

**Sequential Signature Requirements:**

Some records require multiple signatures in specific order:
1. **Operator**: Execution completion
2. **Reviewer**: Data review
3. **Approver**: Final approval

**System Enforcement:**
- Next signature unavailable until prior complete
- System prevents out-of-order signatures
- All signatures recorded with timestamps

### 5.4 Signature Manifestations

Per 21 CFR 11.50(b), signatures appear as:
```
Electronically signed by: Jane Doe (jdoe)
Date: 2024-11-29 14:35:22 UTC
Meaning: Approved  
Reason: Data review complete, no anomalies
```

**Linked Signature Components (also displayed):**
```
Method: Password-based authentication
IP Address: 10.0.1.45
Session: sess_abc123xyz
Hash: a1b2c3d4e5f6...
```

### 5.5 Signature Verification

**User Verification:**
1. Click signature to view details
2. System displays full signature metadata
3. Verify signer identity, time, meaning
4. Check cryptographic hash integrity

**Automated Verification:**
- System validates hash on every record access
- Tampering detected if hash mismatch
- Alert generated for hash failures
- Record locked pending investigation

### 5.6 Password Management for Signatures

**Password Requirements:**
- Minimum 12 characters
- Complexity enforced (upper, lower, number, special)
- Changed every 90 days
- Cannot reuse last 5 passwords
- 3 failed attempts locks signature capability

**Password Reset:**
1. User requests reset via system
2. IT verifies user identity (in person or 2FA)
3. Temporary password issued
4. User must change on next login
5. Signature capability restored

### 5.7 Signature Delegation

**When Delegation Permitted:**
- Extended absence (vacation, medical leave)
- Emergency situations
- Workload distribution (pre-approved)

**Delegation Procedure:**
1. Original signer completes Delegation Form (ES-002)
2. QA approves delegation
3. Delegate acknowledges responsibilities
4. System records delegation period
5. All delegated signatures clearly marked
6. Original signer notified of each signature

**Delegation Restrictions:**
- Maximum 30 days
- Cannot sub-delegate
- Original signer remains accountable
- QA can revoke at any time

### 5.8 Signature Revocation/Removal

**Conditions for Signature Removal:**
- Data entry error discovered
- Wrong signature meaning applied
- Signature applied prematurely
- System error during signing

**Removal Procedure:**
1. User submits Signature Removal Request (ES-003)
2. Provide detailed justification
3. QA reviews and approves
4. IT performs removal with full audit trail
5. Original signature retained in history
6. Removal event logged
7. Record re-opened for correct signature

**Prohibited:**
- Cannot remove signatures to hide errors
- Cannot remove approved signatures without investigation
- All removals subject to audit review

### 5.9 Emergency Signature Procedures

**Emergency Override:**
- Used only when signer unavailable and time-critical
- Requires dual authorization (QA + IT)
- Emergency signature clearly marked
- Original signer notified within 24 hours
- Full justification documented

## 6. System Controls

### 6.1 Technical Controls

**Authentication:**
- Username/password combination required
- Passwords encrypted in database
- Failed login attempts logged
- Account lockout after 3 failures

**Data Integrity:**
- SHA-256 hash of signed data
- Hash verified on every access
- Tamper detection automated
- Hash mismatch triggers alert

**Audit Trail:**
- All signature events logged
- Logs tamper-proof (hash-chained)
- Logs include user, time, action, result
- Logs retained permanently

### 6.2 Administrative Controls

**Training:**
- Initial signature training required
- Annual refresher mandated
- Training includes:
  - 21 CFR Part 11 requirements
  - Password security
  - Proper signature use
  - Consequences of misuse

**Monitoring:**
- Weekly review of signature activity
- Monthly trend analysis
- Quarterly audit by QA
- Annual system validation

## 7. Signature Security

**User Commitments:**
- Keep password confidential
- Never share credentials
- Report lost/compromised passwords immediately
- Sign out when leaving workstation
- Do not sign for others

**Consequences of Misuse:**
- First violation: Retraining
- Second violation: Signature suspension
- Third violation: Access revoked
- Intentional misuse: Termination + regulatory report

## 8. Records and Forms

| Document | Retention |
|----------|-----------|
| Signature Agreement (ES-001) | Permanent |
| Delegation Form (ES-002) | 7 years |
| Removal Request (ES-003) | 7 years |
| Electronic Signature Logs | Permanent |
| Training Records | Permanent |

## 9. Definitions

- **Electronic Signature**: Computer-generated representation of signer's identity
- **Signature Meaning**: Purpose of signature (reviewed, approved, witnessed)
- **Hash**: Cryptographic checksum verifying data integrity
- **Manifestation**: Visual representation of signature

## 10. References

- 21 CFR Part 11 - Electronic Records; Electronic Signatures
- 21 CFR 11.50 - Signature Manifestations
- 21 CFR 11.70 - Signature/Record Linking
- POLYMORPH_LITE System Validation Package

## 11. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 4.0 | 2024-11-29 | QA Team | Initial for v4.0 |

---

**END OF SOP**
