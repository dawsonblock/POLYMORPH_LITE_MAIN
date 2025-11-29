# Standard Operating Procedure: User Access Management
## POLYMORPH_LITE v4.0 - LabOS Polymorph Edition

**SOP ID**: SOP-UA-001  
**Version**: 4.0  
**Effective Date**: 2024-11-29  
**Review Frequency**: Annual

---

## 1. Purpose

This SOP defines the procedures for managing user access to the POLYMORPH_LITE system, ensuring compliance with 21 CFR Part 11 requirements for access controls and audit trails.

## 2. Scope

Applies to all personnel requiring access to POLYMORPH_LITE v4.0, including:
- Laboratory operators
- Quality assurance personnel
- System administrators
- Maintenance personnel
- Auditors

## 3. Responsibilities

| Role | Responsibility |
|------|----------------|
| **System Owner** | Overall accountability for access control |
| **IT Administrator** | User account creation, modification, deletion |
| **QA Manager** | Review and approval of access requests |
| **Department Manager** | Authorization of access for team members |
| **User** | Compliance with password and access policies |

## 4. Procedures

### 4.1 Access Request

**Step 1: Request Submission**
1. Employee completes Access Request Form (Form-UA-001)
2. Specify required access level:
   - **Operator**: Execute workflows, view data
   - **Analyst**: Execute workflows, analyze data, generate reports
   - **Administrator**: Full system access
   - **Read-Only**: View-only access for auditors

**Step 2: Manager Approval**
1. Department manager reviews business justification
2. Manager signs approval on form
3. Forward to QA for review

**Step 3: QA Review**
1. QA verifies training completion
2. QA confirms role-appropriate access level
3. QA approves and forwards to IT

**Step 4: Account Creation**
1. IT Administrator creates user account in system
2. Initial password generated (must be changed on first login)
3. Access level assigned per approved request
4. Account details emailed to user

### 4.2 Password Requirements

All user passwords must meet the following criteria:
- Minimum 12 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character (!@#$%^&*)
- Not matching previous 5 passwords
- Changed every 90 days

### 4.3 Account Activation

**First Login Procedure:**
1. User accesses system login page
2. Enter username and temporary password
3. System prompts for password change
4. User creates new password meeting requirements
5. User reviews and accepts:
   - Terms of Use
   - 21 CFR Part 11 acknowledgment
   - Data privacy notice

**Training Verification:**
- Account remains inactive until training completion verified
- Training records linked to user account
- Annual refresher training required

### 4.4 Access Modification

**When Access Changes Required:**
- Role change
- Department transfer
- Additional access needed
- Access reduction

**Procedure:**
1. Submit Access Modification Form (Form-UA-002)
2. Manager approval required
3. QA review for compliance
4. IT implements changes within 24 hours
5. User notified of changes

### 4.5 Account Deactivation

**Immediate Deactivation Triggers:**
- Employment termination
- Extended leave (>30 days)
- Security violation
- Repeated login failures (3 attempts)

**Procedure:**
1. Manager/HR notifies IT of trigger event
2. IT deactivates account immediately
3. All active sessions terminated
4. Access revocation logged in audit trail
5. QA notified for review

**Account Deletion:**
- Accounts deactivated for >180 days may be deleted
- Audit trail retained per retention policy
- QA approval required before deletion

### 4.6 Periodic Access Review

**Quarterly Review:**
1. IT generates current user access report
2. Department managers review team access
3. Identify unnecessary access
4. Submit deactivation requests
5. QA reviews compliance with role-based access

**Annual Recertification:**
1. All users complete access recertification
2. Confirm current role and responsibilities
3. Acknowledge continued need for access
4. Re-complete 21 CFR Part 11 training
5. Failure to recertify results in deactivation

### 4.7 Emergency Access

**Emergency Account Usage:**
- Pre-configured emergency account available
- Requires two-person authorization
- Time-limited (24 hours maximum)
- All actions logged and reviewed

**Procedure:**
1. Contact on-call IT administrator
2. Provide business justification
3. IT and QA both approve emergency access
4. Emergency account credentials provided
5. Account auto-expires after time limit
6. Full audit review within 24 hours

## 5. Audit Trail Requirements

All access-related events must be logged:
- Login attempts (successful and failed)
- Password changes
- Access level modifications
- Account activation/deactivation
- Emergency access usage
- Session timeouts

**Audit Log Review:**
- Weekly review by IT
- Monthly review by QA
- Quarterly review by System Owner
- Anomalies investigated within 48 hours

## 6. Training Requirements

**Initial Training:**
- System navigation
- Role-specific functions
- Password security
- 21 CFR Part 11 compliance
- Data integrity principles

**Annual Refresher:**
- Policy updates
- New features
- Security awareness
- Compliance reminders

## 7. Forms and Records

| Form/Record | Retention |
|-------------|-----------|
| Access Request Form (UA-001) | 7 years |
| Access Modification Form (UA-002) | 7 years |
| Access Review Reports | 7 years |
| Audit Logs | Permanent |
| Training Records | Permanent |

## 8. Definitions

- **User**: Any person with valid credentials to access system
- **Access Level**: Defined set of permissions assigned to role
- **Audit Trail**: Electronic record of all system activities
- **Emergency Access**: Temporary elevated access for critical situations

## 9. References

- 21 CFR Part 11 - Electronic Records; Electronic Signatures
- ISO/IEC 27001 - Information Security Management
- Company IT Security Policy
- POLYMORPH_LITE User Manual v4.0

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 4.0 | 2024-11-29 | QA Team | Initial v4.0 release |

---

## Appendix A: Access Request Form Template

```
POLYMORPH_LITE Access Request Form (UA-001)

Requestor Information:
Name: ________________________
Employee ID: ________________________
Department: ________________________
Email: ________________________

Access Details:
☐ New Account
☐ Modify Existing
☐ Deactivate Account

Requested Access Level:
☐ Operator
☐ Analyst  
☐ Administrator
☐ Read-Only

Business Justification:
_____________________________________________
_____________________________________________

Training Completion Date: _______________

Approvals:
Department Manager: _____________ Date: _______
QA Manager: _____________ Date: _______

IT Use Only:
Account Created: _____________ Date: _______
Username: _________________________
Initial Password Sent: ☐
```

---

**END OF SOP**
