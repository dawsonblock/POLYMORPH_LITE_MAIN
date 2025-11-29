# Operational Qualification (OQ) Protocol

**Protocol Number:** OQ-XXXX
**Revision:** 1.0
**System:** POLYMORPH LITE
**Date:** [YYYY-MM-DD]

## 1. Approval
| Role | Name | Signature | Date |
|------|------|-----------|------|
| Author | | | |
| Reviewer | | | |
| QA | | | |

## 2. Objective
To verify that the POLYMORPH LITE system functions according to its functional specifications.

## 3. Scope
This protocol covers the testing of:
- User Access Control
- Hardware Control (DAQ, Raman)
- Workflow Execution
- Data Integrity & Audit Trails

## 4. Test Cases

### 4.1 Security & Access Control
| ID | Test Case | Expected Result | Actual Result | Pass/Fail |
|----|-----------|-----------------|---------------|-----------|
| OQ-1.1 | Login with valid credentials | Access granted | | |
| OQ-1.2 | Login with invalid credentials | Access denied | | |
| OQ-1.3 | Session timeout | Auto-logout after X min | | |

### 4.2 Hardware Control
| ID | Test Case | Expected Result | Actual Result | Pass/Fail |
|----|-----------|-----------------|---------------|-----------|
| OQ-2.1 | Connect to DAQ | Status: Connected | | |
| OQ-2.2 | Set Voltage (AO) | Voltage output matches setpoint | | |
| OQ-2.3 | Acquire Spectrum | Spectrum data returned | | |
| OQ-2.4 | Safety Interlock Trigger | System halts operation | | |

### 4.3 Workflow Engine
| ID | Test Case | Expected Result | Actual Result | Pass/Fail |
|----|-----------|-----------------|---------------|-----------|
| OQ-3.1 | Load valid workflow | Workflow validates successfully | | |
| OQ-3.2 | Run workflow | Steps execute in order | | |
| OQ-3.3 | Abort workflow | Execution stops immediately | | |

### 4.4 Electronic Records (Part 11)
| ID | Test Case | Expected Result | Actual Result | Pass/Fail |
|----|-----------|-----------------|---------------|-----------|
| OQ-4.1 | Audit Trail Generation | Action logged with user/timestamp | | |
| OQ-4.2 | Data Immutability | Cannot delete raw data | | |

## 5. Deviations
| ID | Description | Impact | Resolution |
|----|-------------|--------|------------|
| | | | |

## 6. Conclusion
The system operation [ ] Meets / [ ] Does Not Meet requirements.

**QA Signature:** ____________________ **Date:** ___________
