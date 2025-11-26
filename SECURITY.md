# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security issues to:
- **Email**: security@polymorph4.com
- **Subject**: `[SECURITY] Brief description`

### What to Include

Please include:
1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if available)
5. Your contact information

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-30 days
  - Medium: 30-90 days
  - Low: Best effort

### Disclosure Policy

- Report the vulnerability first (coordinated disclosure)
- Allow reasonable time for a fix (90 days)
- We'll acknowledge your contribution in release notes

### Security Measures

POLYMORPH-4 Lite implements:

#### Authentication & Authorization
- JWT-based authentication (HS256)
- Multi-factor authentication (MFA via TOTP)
- Role-based access control (RBAC) 
- Session management
- Password strength requirements

#### Data Protection
- Encrypted communications (TLS/SSL in production)
- Secure password storage (bcrypt, Argon2 dependency ready for upgrade)
- Audit trail integrity (SHA-256 hash chains + RSA signatures)
- Database integrity checks (not encrypted at rest)

#### API Security
- Rate limiting (per IP, per user)
- CORS protection
- Input validation
- SQL injection prevention
- XSS protection

#### Infrastructure
- Container security (Docker)
- Secrets management
- Regular dependency updates
- Security scanning (CI/CD)

#### Compliance
- 21 CFR Part 11 controls
- Audit trails (immutable via hash chains, cryptographically signed with RSA)
- Electronic signatures (RSA, 2048-bit minimum)
- Data integrity validation

### Security Updates

Subscribe to security advisories:
- GitHub Security Advisories
- Release notes
- Email list (enterprise customers)

### Bug Bounty

We currently do not have a bug bounty program. However, we deeply appreciate responsible disclosure and will acknowledge security researchers in our release notes.

### Known Security Considerations

1. **Validation**: While system includes 21 CFR Part 11 mechanisms, formal validation (IQ/OQ/PQ) is the operator's responsibility
2. **Deployment**: Default configurations are for development; production deployments must follow security hardening guides
3. **Dependencies**: Regular updates required; monitor security advisories
4. **Database Storage**: SQLite databases store data unencrypted on disk; integrity is protected via hash chains and signatures
5. **Password Hashing**: Current implementation uses bcrypt; Argon2 dependency is included for future upgrade
6. **Single-Node Optimized**: Rate limiting and session management are in-process; multi-node deployments require Redis backend

### Security Best Practices

When deploying POLYMORPH-4 Lite:

1. **Change Default Secrets**
   - Generate strong `SECRET_KEY`
   - Set unique `REDIS_PASSWORD`
   - Update database credentials

2. **Enable HTTPS**
   - Use valid SSL/TLS certificates
   - Enforce HTTPS-only
   - Configure HSTS headers

3. **Network Security**
   - Use firewall rules
   - Restrict access to Redis/database
   - Configure VPN for remote access

4. **Regular Updates**
   - Monitor for security patches
   - Test updates in staging
   - Maintain backup before updates

5. **Monitoring**
   - Enable security logging
   - Monitor failed login attempts
   - Set up alerting
   - Regular log review

### Contact

- **Security Email**: security@polymorph4.com
- **General Support**: support@polymorph4.com
- **Website**: https://polymorph4.com

---

**Last Updated**: 2025-11-25

Thank you for helping keep POLYMORPH-4 Lite secure!
