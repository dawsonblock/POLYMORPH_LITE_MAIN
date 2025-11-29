# SOP: User Management

**SOP Number:** SOP-IT-001
**Revision:** 1.0
**Effective Date:** [YYYY-MM-DD]

## 1. Purpose
To define the procedures for creating, modifying, and deactivating user accounts in POLYMORPH LITE to ensure secure access control.

## 2. Scope
Applies to all users of the POLYMORPH LITE system.

## 3. Responsibilities
- **System Administrator**: Manages user accounts.
- **QA**: Reviews access logs periodically.

## 4. Procedure

### 4.1 New User Account Creation
1.  Request for access must be approved by Department Manager.
2.  Administrator creates account with:
    - Unique Username
    - Temporary Password (must be changed on first login)
    - Assigned Role (Operator, Scientist, Admin) based on "Least Privilege".
3.  User receives credentials securely.

### 4.2 Password Policy
- Minimum length: 12 characters.
- Complexity: Mix of upper, lower, numbers, symbols.
- Expiration: Every 90 days.
- History: Cannot reuse last 5 passwords.

### 4.3 Account Deactivation
1.  Upon employee termination or role change, access must be revoked within 24 hours.
2.  Administrator disables the account (do not delete, to preserve audit trail).

### 4.4 Periodic Review
1.  QA reviews active user list quarterly.
2.  Discrepancies are reported and resolved immediately.

## 5. References
- 21 CFR Part 11.10(d) - Limiting System Access
