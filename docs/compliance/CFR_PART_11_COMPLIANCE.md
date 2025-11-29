# 21 CFR Part 11 Compliance Guide
## POLYMORPH-LITE Electronic Records and Electronic Signatures

**Document ID**: COMP-CFR11-001  
**Version**: 1.0  
**Date**: 2024-11-28  
**Applies To**: POLYMORPH-LITE v3.1+

---

## 1. Introduction

This document describes how POLYMORPH-LITE complies with 21 CFR Part 11 requirements for electronic records and electronic signatures in FDA-regulated environments.

### 1.1 Scope

21 CFR Part 11 applies to records in electronic form that are:
- Created, modified, maintained, archived, retrieved, or transmitted
- Required by FDA regulations
- Submitted to FDA

### 1.2 POLYMORPH-LITE Applicability

POLYMORPH-LITE generates and maintains electronic records for:
- Laboratory workflows and experiments
- Sample tracking and lineage
- Instrument calibration
- Audit trails
- Quality control data

---

## 2. Subpart B - Electronic Records

### 2.1 Validation (§11.10(a))

**Requirement**: Systems must be validated to ensure accuracy, reliability, consistent intended performance, and the ability to discern invalid or altered records.

**POLYMORPH-LITE Implementation**:
- **Installation Qualification (IQ)**: Documented in `docs/validation/completed/IQ-DEMO-001.md`
- **Operational Qualification (OQ)**: Documented in `docs/validation/completed/OQ-DEMO-001.md`
- **Performance Qualification (PQ)**: Documented in `docs/validation/completed/PQ-DEMO-001.md`
- **Validation Master Plan**: This document serves as the master plan
- **Change Control**: All changes tracked in Git with commit messages and approval workflow

**Evidence**:
- Validation protocols with test results
- Traceability matrix linking requirements to tests
- Signed approval records

---

### 2.2 Ability to Generate Accurate and Complete Copies (§11.10(b))

**Requirement**: The ability to generate accurate and complete copies of records in both human-readable and electronic form suitable for inspection, review, and copying by FDA.

**POLYMORPH-LITE Implementation**:
- **Human-Readable**: Web UI displays all records with full metadata
- **Electronic Export**: API endpoints provide JSON/CSV export
- **PDF Reports**: Compliance reports generated with all required information
- **Audit Trail Export**: Complete audit trail exportable for review

**API Endpoints**:
```bash
# Export sample records
GET /api/samples?format=json
GET /api/samples?format=csv

# Export audit trail
GET /api/compliance/audit?format=json
GET /api/compliance/audit?format=pdf

# Export workflow execution
GET /api/workflow-builder/executions/{run_id}?format=json
```

---

### 2.3 Protection of Records (§11.10(c))

**Requirement**: Protection of records to enable their accurate and ready retrieval throughout the records retention period.

**POLYMORPH-LITE Implementation**:
- **Database Backups**: Automated daily backups (see `scripts/backup_database.py`)
- **Backup Retention**: 30-day retention policy (configurable)
- **Off-site Storage**: Backups stored separately from production
- **Backup Verification**: Monthly restoration tests
- **Data Integrity**: PostgreSQL ACID compliance ensures data consistency

**Backup Procedure**:
```bash
# Automated daily backup
0 2 * * * /usr/bin/python3 /app/scripts/backup_database.py

# Manual backup
python3 scripts/backup_database.py

# Restore from backup
psql polymorph_db < backup_YYYYMMDD.sql
```

---

### 2.4 Limiting System Access (§11.10(d))

**Requirement**: Limiting system access to authorized individuals.

**POLYMORPH-LITE Implementation**:
- **Authentication**: JWT-based authentication required for all API access
- **Role-Based Access Control (RBAC)**: 6 standard roles with granular permissions
- **Password Policy**: 12+ characters, complexity requirements, 90-day expiry
- **Multi-Factor Authentication (MFA)**: Supported for admin accounts
- **Session Management**: 15-minute token expiry, 30-minute inactivity timeout
- **Account Lockout**: 5 failed attempts = 30-minute lockout

**Role Permissions**:
| Role | Create | Read | Update | Delete | Approve | Audit |
|------|--------|------|--------|--------|---------|-------|
| Viewer | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Technician | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Scientist | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| QA | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Admin | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Compliance | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |

---

### 2.5 Use of Secure, Computer-Generated, Time-Stamped Audit Trails (§11.10(e))

**Requirement**: Use of secure, computer-generated, time-stamped audit trails to independently record the date and time of operator entries and actions that create, modify, or delete electronic records.

**POLYMORPH-LITE Implementation**:
- **Comprehensive Audit Trail**: All actions logged with timestamp, user, and details
- **Immutable Records**: Audit trail uses cryptographic hash chain to prevent tampering
- **Time Synchronization**: All timestamps in UTC from server clock (NTP-synchronized)
- **Audit Events Captured**:
  - User login/logout
  - Record creation/modification/deletion
  - Workflow execution start/stop
  - AI decisions
  - Approval actions
  - Configuration changes

**Audit Trail Structure**:
```python
{
  "id": "uuid",
  "timestamp": "2024-11-28T22:00:00Z",  # UTC
  "event": "SAMPLE_CREATED",
  "actor": "scientist@lab.com",
  "subject": "SAMPLE-001",
  "details": {...},
  "prev_hash": "sha256_of_previous_event",
  "hash": "sha256_of_this_event",
  "signature": "digital_signature"
}
```

**Hash Chain Verification**:
```bash
# Verify audit trail integrity
GET /api/compliance/audit/verify-chain
```

---

### 2.6 Use of Operational System Checks (§11.10(f))

**Requirement**: Use of operational system checks to enforce permitted sequencing of steps and events, as appropriate.

**POLYMORPH-LITE Implementation**:
- **Workflow Validation**: Workflow definitions validated before execution
- **State Machine**: Workflow execution follows defined state transitions
- **Prerequisite Checks**: Required approvals verified before execution
- **Safety Interlocks**: Hardware safety checks before hazardous operations
- **Data Validation**: Pydantic models enforce data types and constraints

**Example Workflow Sequencing**:
```
1. Workflow Definition Created
   ↓ (requires approval)
2. Workflow Approved by QA
   ↓ (requires activation)
3. Workflow Activated
   ↓ (can now execute)
4. Workflow Execution Started
   ↓ (follows defined steps)
5. Steps Execute in Order
   ↓ (safety checks at each step)
6. Workflow Completed
   ↓ (audit trail closed)
```

---

### 2.7 Use of Authority Checks (§11.10(g))

**Requirement**: Use of authority checks to ensure that only authorized individuals can use the system, electronically sign a record, access the operation or computer system input or output device, alter a record, or perform the operation at hand.

**POLYMORPH-LITE Implementation**:
- **Permission Checks**: Every API endpoint checks user permissions
- **Electronic Signatures**: Require password re-entry and specific role
- **Audit Access**: Only QA and Compliance roles can access full audit trail
- **Approval Authority**: Only QA/Admin can approve workflows
- **Configuration Changes**: Only Admin can modify system configuration

**Code Example**:
```python
@router.post("/workflows/{id}/approve")
async def approve_workflow(
    id: str,
    current_user: dict = Depends(get_current_user)
):
    # Check user has approval authority
    if not any(role in ["qa", "admin", "compliance"] 
               for role in current_user["roles"]):
        raise HTTPException(403, "Insufficient permissions")
    
    # Require password re-entry for approval
    # (electronic signature)
    ...
```

---

### 2.8 Use of Device Checks (§11.10(h))

**Requirement**: Use of device (e.g., terminal) checks to determine, as appropriate, the validity of the source of data input or operational instruction.

**POLYMORPH-LITE Implementation**:
- **IP Address Logging**: All requests logged with source IP
- **Device Fingerprinting**: Browser/device information captured
- **Geolocation**: Optional geofencing for sensitive operations
- **Trusted Devices**: Optional device registration for MFA
- **Session Binding**: Tokens bound to specific device/browser

---

### 2.9 Determination of Record Falsification (§11.10(i))

**Requirement**: Determination that persons who develop, maintain, or use electronic record/electronic signature systems have the education, training, and experience to perform their assigned tasks.

**POLYMORPH-LITE Implementation**:
- **Training Records**: User training tracked in database
- **Competency Assessment**: Required before system access granted
- **Role Assignment**: Roles assigned based on qualifications
- **Training Documentation**: SOPs and user manuals provided
- **Audit of Training**: Training records included in compliance reports

**Training Requirements**:
| Role | Required Training |
|------|------------------|
| Technician | Basic LIMS, Workflow Execution |
| Scientist | Full LIMS, Workflow Creation, Data Analysis |
| QA | Compliance, Approval Procedures, Audit Review |
| Admin | System Administration, Security, Backup/Recovery |

---

### 2.10 Establishment of Controls (§11.10(j))

**Requirement**: The establishment of, and adherence to, written policies that hold individuals accountable and responsible for actions initiated under their electronic signatures.

**POLYMORPH-LITE Implementation**:
- **Electronic Signature Policy**: Documented in this guide
- **User Agreement**: Users must accept policy before system access
- **Non-Repudiation**: Electronic signatures cannot be repudiated
- **Accountability**: All actions tied to specific user accounts
- **Disciplinary Procedures**: Policy violations documented

**Electronic Signature Policy**:
1. Electronic signatures are legally binding
2. Users are responsible for all actions under their account
3. Passwords must not be shared
4. Suspicious activity must be reported immediately
5. Violations may result in access revocation

---

### 2.11 System Documentation Controls (§11.10(k))

**Requirement**: Use of appropriate controls over systems documentation including:
- Adequate controls over the distribution of, access to, and use of documentation for system operation and maintenance
- Revision and change control procedures to maintain an audit trail that documents time-sequenced development and modification of systems documentation

**POLYMORPH-LITE Implementation**:
- **Version Control**: All code and documentation in Git
- **Change Control**: All changes require commit message and review
- **Documentation Access**: Documentation publicly available (open source)
- **Revision History**: Complete history in Git log
- **Release Notes**: Changes documented in `RELEASE_NOTES.md`

---

## 3. Subpart C - Electronic Signatures

### 3.1 Electronic Signatures - General Requirements (§11.50)

**Requirement**: Signed electronic records shall contain information associated with the signing that clearly indicates all of the following:
- The printed name of the signer
- The date and time when the signature was executed
- The meaning (such as review, approval, responsibility, or authorship) associated with the signature

**POLYMORPH-LITE Implementation**:
```python
{
  "signature": {
    "signer_name": "Dr. Jane Smith",
    "signer_email": "jane.smith@lab.com",
    "signed_at": "2024-11-28T22:00:00Z",
    "meaning": "APPROVED",
    "reason": "Workflow validation complete",
    "signature_hash": "sha256_hash",
    "public_key": "rsa_public_key"
  }
}
```

---

### 3.2 Signature Manifestations (§11.70)

**Requirement**: Electronic signatures and handwritten signatures executed to electronic records shall be linked to their respective electronic records to ensure that the signatures cannot be excised, copied, or otherwise transferred.

**POLYMORPH-LITE Implementation**:
- **Cryptographic Binding**: Signatures use RSA private key
- **Hash Verification**: Signature includes hash of signed data
- **Immutable Storage**: Signatures stored in audit trail (hash chain)
- **Tamper Detection**: Any modification invalidates signature

---

### 3.3 Signature/Record Linking (§11.100)

**Requirement**: Electronic signatures shall be unique to one individual and shall not be reused by, or reassigned to, anyone else.

**POLYMORPH-LITE Implementation**:
- **Unique Keys**: Each user has unique RSA key pair
- **Key Storage**: Private keys encrypted and stored securely
- **No Sharing**: Keys cannot be exported or shared
- **Revocation**: Keys revoked when user leaves organization

---

### 3.4 Signature Verification (§11.200)

**Requirement**: Electronic signatures that are not based upon biometrics shall employ at least two distinct identification components such as an identification code and password.

**POLYMORPH-LITE Implementation**:
- **Two-Factor**: Email/username + password
- **Optional MFA**: TOTP for additional security
- **Password Re-entry**: Required for critical actions (signatures)
- **Session Timeout**: Automatic logout after inactivity

---

## 4. Compliance Verification

### 4.1 Self-Assessment Checklist

| Requirement | Status | Evidence |
|------------|--------|----------|
| §11.10(a) Validation | ✅ | IQ/OQ/PQ documents |
| §11.10(b) Accurate copies | ✅ | Export APIs, PDF reports |
| §11.10(c) Record protection | ✅ | Backup procedures |
| §11.10(d) Access limits | ✅ | RBAC, authentication |
| §11.10(e) Audit trails | ✅ | Hash-chained audit log |
| §11.10(f) System checks | ✅ | Workflow validation |
| §11.10(g) Authority checks | ✅ | Permission system |
| §11.10(h) Device checks | ✅ | IP logging, fingerprinting |
| §11.10(i) Training | ✅ | Training records |
| §11.10(j) Policies | ✅ | This document |
| §11.10(k) Documentation | ✅ | Git version control |
| §11.50 Signature info | ✅ | Signature metadata |
| §11.70 Signature linking | ✅ | Cryptographic binding |
| §11.100 Unique signatures | ✅ | Unique RSA keys |
| §11.200 Two-factor | ✅ | Email + password + MFA |

---

## 5. Limitations and Disclaimers

### 5.1 Current Limitations

- **Biometric Signatures**: Not currently supported
- **Hardware Tokens**: Not currently supported
- **Full GMP Validation**: Requires site-specific validation
- **Regulatory Submission**: Not pre-approved by FDA

### 5.2 Recommendations for Full Compliance

1. **Site-Specific Validation**: Perform IQ/OQ/PQ at your facility
2. **Training Program**: Implement formal training and competency assessment
3. **SOPs**: Develop site-specific standard operating procedures
4. **Quality Manual**: Integrate into your quality management system
5. **Regulatory Consultation**: Consult with regulatory affairs specialist

---

## 6. References

- 21 CFR Part 11: Electronic Records; Electronic Signatures
- FDA Guidance: Part 11, Electronic Records; Electronic Signatures — Scope and Application
- GAMP 5: A Risk-Based Approach to Compliant GxP Computerized Systems
- ISPE GAMP Good Practice Guide: Data Integrity by Design

---

## 7. Approval

**Prepared By**: ________________  
**Title**: ________________  
**Date**: ________________

**Reviewed By** (QA): ________________  
**Title**: ________________  
**Date**: ________________

**Approved By** (Compliance): ________________  
**Title**: ________________  
**Date**: ________________

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-11-28 | POLYMORPH Team | Initial 21 CFR Part 11 compliance guide |

**Next Review Date**: 2025-11-28
