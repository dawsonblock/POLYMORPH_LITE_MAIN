# Operational Qualification (OQ)
## POLYMORPH-LITE Golden Path Demo

**Document ID**: OQ-DEMO-001  
**Version**: 1.0  
**Date**: 2024-11-28  
**System**: POLYMORPH-LITE v3.1  
**Use Case**: Crystallization Screening with Raman + AI  
**Prerequisites**: IQ-DEMO-001 completed and approved

---

## 1. Purpose

This Operational Qualification (OQ) verifies that the POLYMORPH-LITE system operates correctly according to specifications for the Golden Path Demo workflow.

## 2. Scope

- User authentication and authorization
- Sample and project management
- Workflow execution
- AI service integration
- Audit trail generation
- Report generation

## 3. Test Environment

**Hardware**: Simulation mode (no physical instruments)  
**Software Version**: POLYMORPH-LITE v3.1  
**Database**: PostgreSQL 15  
**Test Data**: Demo samples and workflows

---

## 4. Operational Tests

### 4.1 User Authentication

**Test ID**: OQ-001  
**Objective**: Verify user login and JWT token generation

**Procedure**:
1. Navigate to http://localhost:3001
2. Enter credentials: `demo@polymorph.local` / `demo123`
3. Click "Login"

**Expected Result**: 
- Successful login
- JWT token generated
- Redirected to dashboard

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________  
**Evidence**: Screenshot attached: ☐ Yes ☐ No

---

### 4.2 Project Creation

**Test ID**: OQ-002  
**Objective**: Verify project creation via API

**Procedure**:
```bash
curl -X POST http://localhost:8001/api/samples/projects \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "TEST-OQ-001",
    "name": "OQ Test Project",
    "description": "Operational qualification test",
    "status": "active"
  }'
```

**Expected Result**: 
- HTTP 201 Created
- Project ID returned
- Project visible in database

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

### 4.3 Sample Creation with Metadata

**Test ID**: OQ-003  
**Objective**: Verify sample creation with full metadata

**Procedure**:
```bash
curl -X POST http://localhost:8001/api/samples \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "sample_id": "SAMPLE-OQ-001",
    "lot_number": "LOT-OQ-2024",
    "project_id": "TEST-OQ-001",
    "metadata": {
      "compound": "Test Compound",
      "temperature_c": 25,
      "solvent": "ethanol"
    }
  }'
```

**Expected Result**:
- HTTP 201 Created
- Sample ID returned
- Metadata stored correctly

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

### 4.4 Workflow Execution - Happy Path

**Test ID**: OQ-004  
**Objective**: Execute complete Golden Path workflow

**Procedure**:
```bash
./scripts/run_hero_demo.sh
```

**Expected Result**:
- All 7 steps complete successfully
- Workflow status: "completed"
- AI decision generated
- Audit trail created

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________  
**Duration**: ________ seconds  
**Evidence**: Log file attached: ☐ Yes ☐ No

---

### 4.5 Device Simulation

**Test ID**: OQ-005  
**Objective**: Verify Raman spectrometer simulation

**Procedure**:
1. Execute workflow with Raman acquisition step
2. Verify spectrum data is generated
3. Check spectrum format and quality

**Expected Result**:
- Spectrum data array generated
- Wavelength range: 200-2000 nm
- Intensity values realistic
- No errors in simulation

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

### 4.6 AI Service Integration

**Test ID**: OQ-006  
**Objective**: Verify AI classification service

**Procedure**:
1. Send Raman spectrum to AI service
2. Receive classification result
3. Verify confidence score

**Expected Result**:
- AI service responds within 5 seconds
- Polymorph classification returned
- Confidence score between 0-1
- Result stored in workflow execution

**Actual Result**: ☐ Pass ☐ Fail  
**AI Confidence**: ________ %  
**Classification**: ________________  
**Tested By**: ________________  
**Date**: ________________

---

### 4.7 Audit Trail Generation

**Test ID**: OQ-007  
**Objective**: Verify audit events are logged

**Procedure**:
1. Execute workflow
2. Query audit trail API
3. Verify all events logged

**Expected Events**:
- [ ] User login
- [ ] Project creation
- [ ] Sample creation
- [ ] Workflow start
- [ ] Spectrum acquisition
- [ ] AI decision
- [ ] Workflow completion

**Actual Result**: ☐ Pass ☐ Fail  
**Events Logged**: ________ / 7  
**Tested By**: ________________  
**Date**: ________________

---

### 4.8 Error Handling - Invalid Input

**Test ID**: OQ-008  
**Objective**: Verify system handles invalid input gracefully

**Procedure**:
```bash
curl -X POST http://localhost:8001/api/samples \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "sample_id": "",
    "lot_number": null
  }'
```

**Expected Result**:
- HTTP 422 Unprocessable Entity
- Validation error message returned
- No database corruption

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

### 4.9 Concurrent Workflow Execution

**Test ID**: OQ-009  
**Objective**: Verify system handles multiple concurrent workflows

**Procedure**:
1. Start 3 workflows simultaneously
2. Monitor execution status
3. Verify all complete successfully

**Expected Result**:
- All workflows execute without interference
- Correct results for each workflow
- No race conditions or deadlocks

**Actual Result**: ☐ Pass ☐ Fail  
**Workflows Completed**: ________ / 3  
**Tested By**: ________________  
**Date**: ________________

---

### 4.10 Data Integrity

**Test ID**: OQ-010  
**Objective**: Verify data integrity throughout workflow

**Procedure**:
1. Execute workflow with known sample
2. Verify data at each step
3. Check final results match expected

**Expected Result**:
- Sample ID preserved throughout
- Metadata intact
- Spectrum data not corrupted
- AI result linked to correct sample

**Actual Result**: ☐ Pass ☐ Fail  
**Tested By**: ________________  
**Date**: ________________

---

## 5. Performance Tests

### 5.1 Workflow Execution Time

| Test Run | Duration (seconds) | Status | Pass/Fail |
|----------|-------------------|--------|-----------|
| Run 1 | | | ☐ Pass ☐ Fail |
| Run 2 | | | ☐ Pass ☐ Fail |
| Run 3 | | | ☐ Pass ☐ Fail |
| Average | | | |

**Acceptance Criteria**: < 120 seconds

---

### 5.2 API Response Time

| Endpoint | Response Time (ms) | Pass/Fail |
|----------|-------------------|-----------|
| /health | | ☐ Pass ☐ Fail |
| /api/samples | | ☐ Pass ☐ Fail |
| /api/workflow-builder/executions | | ☐ Pass ☐ Fail |

**Acceptance Criteria**: < 2000ms

---

## 6. Summary

**Total Tests**: 10  
**Tests Passed**: ________  
**Tests Failed**: ________  
**Pass Rate**: ________ %

**Minimum Pass Rate**: 90%

---

## 7. Deviations

List any deviations from expected operation:

1. ________________________________________________________________
2. ________________________________________________________________
3. ________________________________________________________________

---

## 8. Approval

**System Operationally Qualified**: ☐ Yes ☐ No ☐ Conditional

**Conditions** (if conditional):
________________________________________________________________
________________________________________________________________

**Performed By**: ________________  
**Signature**: ________________  
**Date**: ________________

**Reviewed By**: ________________  
**Signature**: ________________  
**Date**: ________________

**Approved By** (QA/Compliance): ________________  
**Signature**: ________________  
**Date**: ________________

---

## 9. Attachments

- [ ] Test execution logs
- [ ] API response samples
- [ ] Audit trail export
- [ ] Workflow execution screenshots
- [ ] Performance test results

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-11-28 | POLYMORPH Team | Initial OQ for Golden Path Demo |

**Next Document**: PQ-DEMO-001 (Performance Qualification)
