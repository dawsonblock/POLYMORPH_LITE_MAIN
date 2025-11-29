# Deployment Hardening Guide
## POLYMORPH_LITE v4.0 Production Security

**Version**: 4.0  
**Last Updated**: 2024-11-29

---

## Overview

This guide provides comprehensive security hardening procedures for deploying POLYMORPH_LITE v4.0 to production environments. Follow these procedures to ensure enterprise-grade security posture.

---

## Pre-Deployment Security Checklist

### Infrastructure

- [ ] VPC configured with private subnets
- [ ] Network ACLs configured
- [ ] Security groups follow least-privilege
- [ ] VPC Flow Logs enabled
- [ ] NAT Gateway for outbound traffic
- [ ] No public IPs on application servers
- [ ] Database in private subnet only

### Secrets Management

- [ ] All secrets in AWS Secrets Manager
- [ ] KMS encryption enabled
- [ ] Secrets rotated (not using defaults)
- [ ] No hardcoded secrets in code
- [ ] Environment variables validated
- [ ] Secret access audited

### Kubernetes Security

- [ ] Pod security contexts applied
- [ ] Network policies implemented
- [ ] RBAC configured (least privilege)
- [ ] Image scanning enabled
- [ ] Resource limits set
- [ ] Readiness/liveness probes configured
- [ ] Non-root containers enforced

### Database Security

- [ ] Encryption at rest enabled
- [ ] Encryption in transit (TLS)
- [ ] Strong passwords (16+ chars)
- [ ] Regular automated backups
- [ ] Point-in-time recovery enabled
- [ ] Connection limits configured
- [ ] Monitoring enabled

---

## Step-by-Step Hardening

### 1. Infrastructure Layer

**VPC Configuration:**
```hcl
# Ensure private subnets for sensitive resources
resource "aws_subnet" "private" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = false  # Critical!
}

# Database subnet group (private only)
resource "aws_db_subnet_group" "main" {
  subnet_ids = aws_subnet.private[*].id
}
```

**Security Group Hardening:**
```hcl
# Application security group - restrictive
resource "aws_security_group" "app" {
  name        = "polymorph-app-sg"
  description = "Application tier security group"
  vpc_id      = aws_vpc.main.id

  # Only allow from load balancer
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "HTTP from ALB only"
  }

  # No direct outbound to internet
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "HTTPS within VPC"
  }
}
```

### 2. Kubernetes Hardening

**Apply Pod Security Standards:**
```bash
# Label namespace with security standard
kubectl label namespace polymorph-lite \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

**Update All Deployments:**
```yaml
# Apply to backend-deployment.yaml
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
        seccompProfile:
          type: RuntimeDefault
      
      containers:
      - name: backend
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          capabilities:
            drop: ["ALL"]
        
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

**Network Policies:**
```bash
# Apply network segmentation
kubectl apply -f infra/k8s/base/network-policies.yaml

# Verify policies
kubectl get networkpolicies -n polymorph-lite
kubectl describe networkpolicy database-network-policy -n polymorph-lite
```

### 3. Secrets Management

**Create Secrets in AWS:**
```bash
# Create KMS key
aws kms create-key \
  --description "POLYMORPH_LITE production encryption" \
  --key-usage ENCRYPT_DECRYPT

# Store application secrets
aws secretsmanager create-secret \
  --name polymorph-lite/production/app-secrets \
  --kms-key-id <key-id> \
  --secret-string '{
    "jwt_secret_key": "<strong-random-key>",
    "postgres_password": "<strong-random-password>",
    "redis_password": "<strong-random-password>"
  }'
```

**Configure K8s External Secrets:**
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: polymorph-secrets
  namespace: polymorph-lite
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: polymorph-secrets
    creationPolicy: Owner
  data:
  - secretKey: JWT_SECRET_KEY
    remoteRef:
      key: polymorph-lite/production/app-secrets
      property: jwt_secret_key
```

### 4. Database Hardening

**Enable Encryption:**
```hcl
resource "aws_db_instance" "main" {
  # ... other config ...
  
  storage_encrypted = true
  kms_key_id        = aws_kms_key.polymorph.arn
  
  # Enforce TLS
  parameter_group_name = aws_db_parameter_group.postgres_tls.name  
}

resource "aws_db_parameter_group" "postgres_tls" {
  family = "postgres15"
  
  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }
}
```

**Connection Security:**
```python
# In application config
DATABASE_URL = (
    "postgresql://user:pass@host:5432/db"
    "?sslmode=require"
    "&sslrootcert=/path/to/rds-ca-cert.pem"
)
```

### 5. TLS/SSL Configuration

**Certificate Management:**
```bash
# Using cert-manager for K8s
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create Let's Encrypt issuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@your-domain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

**Ingress with TLS:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: polymorph-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - polymorph.your-domain.com
    secretName: polymorph-tls
  rules:
  - host: polymorph.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: polymorph-frontend
            port:
              number: 80
```

### 6. Monitoring & Logging

**Enable CloudWatch Logs:**
```hcl
resource "aws_cloudwatch_log_group" "app" {
  name              = "/aws/eks/polymorph-lite"
  retention_in_days = 90
  kms_key_id        = aws_kms_key.polymorph.arn
}
```

**Configure Prometheus Alerts:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-alerts
data:
  alerts.yml: |
    groups:
    - name: security
      rules:
      - alert: HighFailedLoginRate
        expr: rate(failed_logins_total[5m]) > 10
        annotations:
          summary: "High rate of failed logins"
      
      - alert: UnauthorizedAccess
        expr: unauthorized_requests_total > 0
        annotations:
          summary: "Unauthorized access attempt detected"
```

### 7. Backup & DR

**Automated Backups:**
```hcl
resource "aws_db_instance" "main" {
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
}

# S3 backup bucket
resource "aws_s3_bucket" "backups" {
  bucket = "polymorph-lite-backups"
  
  lifecycle_rule {
    enabled = true
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}
```

---

## Post-De ployment Verification

### Security Scan Checklist

```bash
# 1. Verify no pods running as root
kubectl get pods -n polymorph-lite -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.securityContext.runAsUser}{"\n"}{end}'

# 2. Check network policies
kubectl get networkpolicies -n polymorph-lite

# 3. Verify TLS certificates
kubectl get certificates -n polymorph-lite

# 4. Check for exposed services
kubectl get svc -n polymorph-lite -o wide

# 5. Audit RBAC
kubectl auth can-i --list --as=system:serviceaccount:polymorph-lite:default

# 6. Scan images for vulnerabilities
trivy image polymorph-backend:v4.0
trivy image polymorph-ai:v4.0
```

### Penetration Testing

Run automated security scans:
```bash
# OWASP ZAP scan
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t https://polymorph.your-domain.com

# SSL Labs scan
curl -s "https://api.ssllabs.com/api/v3/analyze?host=polymorph.your-domain.com"
```

---

## Incident Response

### Security Event Response

1. **Detection**: Alert triggered
2. **Containment**: Isolate affected resources
3. **Investigation**: Review audit logs
4. **Remediation**: Apply fixes
5. **Recovery**: Restore services
6. **Post-Mortem**: Document lessons learned

### Emergency Contacts

- **Security Team**: security@your-company.com
- **On-Call**: +1-XXX-XXX-XXXX
- **AWS Support**: Enterprise support ticket

---

## Compliance Verification

### 21 CFR Part 11 Requirements

- [ ] Electronic signatures implemented
- [ ] Audit trails tamper-proof
- [ ] Access controls enforced
- [ ] Data integrity validated
- [ ] System validation documented

### GDPR Requirements (if applicable)

- [ ] Data encryption at rest
- [  ] Data encryption in transit
- [ ] Data retention policies
- [ ] Right to erasure capability
- [ ] Data processing agreements

---

## Maintenance

### Weekly
- Review CloudWatch alerts
- Check failed login attempts
- Verify backup completion

### Monthly
- Rotate secrets
- Review access logs
- Update security patches
- Scan for vulnerabilities

### Quarterly
- Penetration testing
- Security audit
- Disaster recovery drill
- Access review

### Annually
- Full security assessment
- Compliance audit
- Infrastructure review
- Update security policies

---

## References

- AWS Security Best Practices
- CIS Kubernetes Benchmark
- NIST Cybersecurity Framework
- 21 CFR Part 11
- ISO/IEC 27001

---

**Document Version**: 4.0  
**Last Review**: 2024-11-29  
**Next Review**: 2025-11-29
