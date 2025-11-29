# Security Hardening Checklist
## POLYMORPH-LITE Production Deployment

**Version**: 1.0  
**Date**: 2024-11-28  
**Applies To**: POLYMORPH-LITE v3.1+

---

## Overview

This checklist ensures POLYMORPH-LITE is deployed with production-grade security. Complete all items before deploying to a GMP/GLP environment.

---

## 1. Secret Management

### 1.1 Environment Variables

- [ ] All secrets moved to environment variables (not hardcoded)
- [ ] `.env` file excluded from version control (in `.gitignore`)
- [ ] Separate `.env` files for dev/staging/prod
- [ ] Production secrets stored in secure vault (AWS Secrets Manager, HashiCorp Vault, etc.)

**Critical Secrets**:
```bash
# Required in .env
SECRET_KEY=<256-bit random key>
JWT_SECRET_KEY=<256-bit random key>
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_PASSWORD=<strong password>
RSA_PRIVATE_KEY_PATH=/secure/path/to/private.pem
```

**Generation Commands**:
```bash
# Generate SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate RSA key pair
openssl genrsa -out private.pem 4096
openssl rsa -in private.pem -pubout -out public.pem
```

---

### 1.2 Key Rotation

- [ ] Secret rotation policy documented
- [ ] Key rotation procedure tested
- [ ] Old keys archived securely
- [ ] Rotation schedule defined (recommend: quarterly)

**Rotation Procedure**:
1. Generate new keys
2. Update environment variables
3. Restart services
4. Verify functionality
5. Archive old keys (do not delete immediately)

---

### 1.3 Access Control

- [ ] Secrets accessible only to authorized personnel
- [ ] Principle of least privilege enforced
- [ ] Secret access logged and audited
- [ ] Emergency access procedure documented

---

## 2. TLS/SSL Configuration

### 2.1 Certificate Management

- [ ] Valid TLS certificate obtained (Let's Encrypt, commercial CA)
- [ ] Certificate expiry monitoring enabled
- [ ] Auto-renewal configured
- [ ] Certificate chain complete

**Nginx TLS Configuration**:
```nginx
server {
    listen 443 ssl http2;
    server_name polymorph.yourlab.com;

    ssl_certificate /etc/ssl/certs/polymorph.crt;
    ssl_certificate_key /etc/ssl/private/polymorph.key;
    
    # Modern TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_prefer_server_ciphers off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Session cache
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
}
```

---

### 2.2 HTTP to HTTPS Redirect

- [ ] All HTTP traffic redirected to HTTPS
- [ ] HSTS header enabled
- [ ] Mixed content warnings resolved

```nginx
server {
    listen 80;
    server_name polymorph.yourlab.com;
    return 301 https://$server_name$request_uri;
}
```

---

### 2.3 Internal Service Communication

- [ ] Backend ↔ Database: TLS enabled
- [ ] Backend ↔ Redis: TLS enabled
- [ ] Backend ↔ AI Service: TLS enabled
- [ ] Certificate validation enforced

---

## 3. Authentication & Authorization

### 3.1 Password Policy

- [ ] Minimum length: 12 characters
- [ ] Complexity requirements enforced
- [ ] Password history: prevent reuse of last 5
- [ ] Account lockout after 5 failed attempts
- [ ] Password expiry: 90 days (configurable)

**Configuration** (`retrofitkit/core/config.py`):
```python
PASSWORD_MIN_LENGTH = 12
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_LOWERCASE = True
PASSWORD_REQUIRE_DIGITS = True
PASSWORD_REQUIRE_SPECIAL = True
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30
```

---

### 3.2 Multi-Factor Authentication (MFA)

- [ ] MFA enabled for admin accounts
- [ ] MFA encouraged for all users
- [ ] TOTP (Time-based One-Time Password) supported
- [ ] Backup codes generated and stored securely

---

### 3.3 Session Management

- [ ] JWT tokens with short expiry (15 minutes)
- [ ] Refresh tokens with longer expiry (7 days)
- [ ] Token revocation mechanism implemented
- [ ] Session timeout after inactivity (30 minutes)

---

### 3.4 Role-Based Access Control (RBAC)

- [ ] Roles defined and documented
- [ ] Permissions granular and specific
- [ ] Default role: minimal permissions
- [ ] Admin role: restricted to authorized personnel

**Standard Roles**:

| Role | Permissions | Use Case |
|------|------------|----------|
| **Viewer** | Read-only access to samples, workflows, results | Auditors, observers |
| **Technician** | Execute workflows, create samples | Lab technicians |
| **Scientist** | Full LIMS access, workflow creation | Research scientists |
| **QA** | Approve workflows, review audit trails | Quality assurance |
| **Admin** | Full system access, user management | System administrators |
| **Compliance** | Audit access, signature verification | Compliance officers |

---

## 4. Network Security

### 4.1 Firewall Configuration

- [ ] Firewall enabled on all servers
- [ ] Only required ports open
- [ ] Default deny policy
- [ ] Port scanning detection enabled

**Required Ports**:
```bash
# Inbound
443/tcp   # HTTPS (public)
22/tcp    # SSH (admin only, IP-restricted)

# Internal only
5432/tcp  # PostgreSQL (backend → database)
6379/tcp  # Redis (backend → cache)
3000/tcp  # AI Service (backend → AI)
9090/tcp  # Prometheus (monitoring)
3000/tcp  # Grafana (monitoring)
```

---

### 4.2 Network Segmentation

- [ ] Database on separate network segment
- [ ] AI service isolated
- [ ] DMZ for public-facing services
- [ ] Internal services not directly accessible

---

### 4.3 Rate Limiting

- [ ] API rate limiting enabled
- [ ] Login attempt rate limiting
- [ ] DDoS protection configured

**Nginx Rate Limiting**:
```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;

location /api/ {
    limit_req zone=api burst=20 nodelay;
}

location /auth/login {
    limit_req zone=login burst=3 nodelay;
}
```

---

## 5. Database Security

### 5.1 Access Control

- [ ] Database user has minimal required permissions
- [ ] No root/superuser access from application
- [ ] Connection pooling configured
- [ ] Prepared statements used (SQL injection prevention)

**Database User Permissions**:
```sql
-- Create dedicated user
CREATE USER polymorph_app WITH PASSWORD 'strong_password';

-- Grant only required permissions
GRANT CONNECT ON DATABASE polymorph_db TO polymorph_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO polymorph_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO polymorph_app;

-- Revoke dangerous permissions
REVOKE CREATE ON SCHEMA public FROM polymorph_app;
```

---

### 5.2 Encryption

- [ ] Data at rest: database encryption enabled
- [ ] Data in transit: TLS for all connections
- [ ] Backup encryption enabled
- [ ] Encryption keys rotated regularly

---

### 5.3 Backup & Recovery

- [ ] Automated daily backups
- [ ] Backups stored off-site
- [ ] Backup restoration tested monthly
- [ ] Backup retention policy: 30 days

**Backup Script** (`scripts/backup_database.py`):
```bash
# Run daily via cron
0 2 * * * /usr/bin/python3 /app/scripts/backup_database.py
```

---

## 6. Application Security

### 6.1 Input Validation

- [ ] All user input validated
- [ ] Pydantic models enforce types
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (output encoding)
- [ ] CSRF protection enabled

---

### 6.2 Security Headers

- [ ] Content-Security-Policy configured
- [ ] X-Frame-Options: DENY
- [ ] X-Content-Type-Options: nosniff
- [ ] Referrer-Policy: no-referrer
- [ ] Permissions-Policy configured

**Nginx Security Headers**:
```nginx
add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';" always;
```

---

### 6.3 Dependency Management

- [ ] All dependencies up to date
- [ ] Security advisories monitored
- [ ] Automated dependency scanning (Dependabot, Snyk)
- [ ] Vulnerable dependencies patched within 7 days

**Check for vulnerabilities**:
```bash
# Python
pip-audit

# Node.js
npm audit
```

---

## 7. Logging & Monitoring

### 7.1 Security Logging

- [ ] All authentication attempts logged
- [ ] Failed login attempts logged
- [ ] Privilege escalation logged
- [ ] Data access logged (audit trail)
- [ ] Configuration changes logged

---

### 7.2 Log Protection

- [ ] Logs stored securely
- [ ] Log tampering detection
- [ ] Logs retained for 1 year (compliance requirement)
- [ ] Sensitive data not logged (passwords, tokens)

---

### 7.3 Monitoring & Alerting

- [ ] Failed login alerts
- [ ] Unusual activity alerts
- [ ] System health monitoring
- [ ] Security event correlation

**Critical Alerts**:
- 5+ failed login attempts in 5 minutes
- Privilege escalation attempts
- Audit trail tampering detected
- Unauthorized API access
- Database connection failures

---

## 8. Compliance & Audit

### 8.1 21 CFR Part 11 Requirements

- [ ] Electronic signatures implemented
- [ ] Audit trail immutable (hash chain)
- [ ] User authentication enforced
- [ ] System validation documented (IQ/OQ/PQ)
- [ ] Change control process defined

---

### 8.2 Data Integrity (ALCOA+)

- [ ] **Attributable**: All actions linked to users
- [ ] **Legible**: Data readable and permanent
- [ ] **Contemporaneous**: Recorded at time of action
- [ ] **Original**: Primary records preserved
- [ ] **Accurate**: Data verified and validated
- [ ] **Complete**: All data captured
- [ ] **Consistent**: Data relationships maintained
- [ ] **Enduring**: Long-term data retention
- [ ] **Available**: Data accessible when needed

---

### 8.3 Audit Readiness

- [ ] Audit trail export functionality
- [ ] Compliance reports automated
- [ ] Validation documentation complete
- [ ] SOPs documented
- [ ] Training records maintained

---

## 9. Incident Response

### 9.1 Incident Response Plan

- [ ] Security incident procedure documented
- [ ] Incident response team identified
- [ ] Communication plan defined
- [ ] Escalation path clear

**Incident Severity Levels**:
- **Critical**: Data breach, system compromise
- **High**: Failed authentication, privilege escalation
- **Medium**: Suspicious activity, policy violations
- **Low**: Minor security events

---

### 9.2 Breach Notification

- [ ] Breach notification procedure defined
- [ ] Regulatory requirements understood
- [ ] Stakeholder communication plan
- [ ] Legal counsel identified

---

## 10. Deployment Checklist

### Pre-Deployment

- [ ] Security review completed
- [ ] Penetration testing performed
- [ ] Vulnerability scan passed
- [ ] Code review completed
- [ ] Dependencies audited

### Deployment

- [ ] Production environment isolated
- [ ] Secrets properly configured
- [ ] TLS certificates valid
- [ ] Firewall rules applied
- [ ] Monitoring enabled
- [ ] Backups configured

### Post-Deployment

- [ ] Security scan performed
- [ ] Logs reviewed
- [ ] Access controls verified
- [ ] Incident response tested
- [ ] Documentation updated

---

## 11. Ongoing Security

### Daily

- [ ] Monitor security alerts
- [ ] Review failed login attempts
- [ ] Check system health

### Weekly

- [ ] Review audit logs
- [ ] Check for security updates
- [ ] Verify backup success

### Monthly

- [ ] Test backup restoration
- [ ] Review access permissions
- [ ] Update security documentation
- [ ] Security training for team

### Quarterly

- [ ] Rotate secrets
- [ ] Penetration testing
- [ ] Security policy review
- [ ] Compliance audit

---

## 12. Security Contacts

**Internal**:
- Security Officer: ________________
- Compliance Officer: ________________
- IT Administrator: ________________

**External**:
- Security Consultant: ________________
- Legal Counsel: ________________
- Regulatory Contact: ________________

---

## Approval

**Security Hardening Completed**: ☐ Yes ☐ No

**Reviewed By**: ________________  
**Signature**: ________________  
**Date**: ________________

**Approved By** (CISO/Security Officer): ________________  
**Signature**: ________________  
**Date**: ________________

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-11-28 | POLYMORPH Team | Initial security hardening checklist |

**Next Review Date**: ________________
