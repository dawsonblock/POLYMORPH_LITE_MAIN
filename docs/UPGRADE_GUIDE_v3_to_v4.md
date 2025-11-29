# Upgrade Guide: v3.0 → v4.0
## POLYMORPH_LITE LabOS Polymorph Edition

**Target Audience**: DevOps Engineers, System Administrators  
**Estimated Upgrade Time**: 2-4 hours  
**Downtime Required**: ~30 minutes

---

## Pre-Upgrade Checklist

### Backup Everything
```bash
# 1. Database backup
kubectl exec -n polymorph-lite postgres-0 -- pg_dump \
  -U polymorph polymorph_lite > backup_v3_$(date +%Y%m%d).sql

# 2. Application configuration
kubectl get configmaps -n polymorph-lite -o yaml > configmaps_backup.yaml
kubectl get secrets -n polymorph-lite -o yaml > secrets_backup.yaml

# 3. Persistent volumes
kubectl get pvc -n polymorph-lite -o yaml > pvc_backup.yaml
```

### Verify Current State
```bash
# Check currentversion
kubectl describe deployment polymorph-backend -n polymorph-lite | grep Image

# Verify health
curl https://your-domain.com/health

# Document current workflows
psql -U polymorph -h localhost -c "SELECT id, name FROM workflows;"
```

### Review Breaking Changes
- ✅ No API breaking changes in v4.0
- ⚠️ New environment variables required
- ⚠️ Database schema changes (3 new tables)
- ⚠️ K8s resource requirements increased

---

## Upgrade Procedure

### Step 1: Update Infrastructure (Terraform)

```bash
cd infra/terraform

# Review changes
terraform plan

# Apply infrastructure updates
terraform apply

# Verify VPC, EKS, RDS updates
terraform output
```

**New Resources Created:**
- KMS key for encryption
- Secrets Manager entries
- S3 backup bucket
- VPC flow logs
- Enhanced security groups

### Step 2: Database Migration

```bash
# 1. Stop application pods (minimize writes)
kubectl scale deployment polymorph-backend -n polymorph-lite --replicas=0

# 2. Run Alembic migrations
kubectl exec -it -n polymorph-lite postgres-0 -- \
  alembic upgrade head

# 3. Verify new tables
kubectl exec -it -n polymorph-lite postgres-0 -- psql -U polymorph -c "\dt"

# Expected new tables:
# - polymorph_events
# - polymorph_signatures
# - polymorph_reports
```

### Step 3: Update Kubernetes Configurations

```bash
# 1. Update ConfigMaps with new environment variables
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: polymorph-config
  namespace: polymorph-lite
data:
  POLYMORPH_DISCOVERY_ENABLED: "true"
  AI_SERVICE_URL: "http://ai-service:3000"
  # ... existing config ...
EOF

# 2. Apply security contexts
kubectl apply -f infra/k8s/base/pod-security.yaml

# 3. Apply network policies
kubectl apply -f infra/k8s/base/network-policies.yaml

# 4. Update deployments
kubectl apply -f infra/k8s/overlays/production/
```

### Step 4: Deploy v4.0 Images

```bash
# 1. Pull new images
docker pull your-registry/polymorph-backend:v4.0
docker pull your-registry/polymorph-ai:v4.0
docker pull your-registry/polymorph-frontend:v4.0

# 2. Update deployments
kubectl set image deployment/polymorph-backend \
  backend=your-registry/polymorph-backend:v4.0 \
  -n polymorph-lite

kubectl set image deployment/ai-service \
  ai=your-registry/polymorph-ai:v4.0 \
  -n polymorph-lite

kubectl set image deployment/polymorph-frontend \
  frontend=your-registry/polymorph-frontend:v4.0 \
  -n polymorph-lite

# 3. Wait for rollout
kubectl rollout status deployment/polymorph-backend -n polymorph-lite
kubectl rollout status deployment/ai-service -n polymorph-lite
kubectl rollout status deployment/polymorph-frontend -n polymorph-lite
```

### Step 5: Verify Deployment

```bash
# 1. Check pod health
kubectl get pods -n polymorph-lite

# 2. Verify services
kubectl get svc -n polymorph-lite

# 3. Test health endpoints
curl https://your-domain.com/health
curl https://your-domain.com/api/polymorph/statistics

# 4. Check logs
kubectl logs -f deployment/polymorph-backend -n polymorph-lite

# 5. Verify new features
curl https://your-domain.com/api/version
# Should return: {"service_version": "v4.0.0", ...}
```

### Step 6: Post-Deployment Configuration

```bash
# 1. Initialize Polymorph Discovery
# Load initial model if available
kubectl cp ai/models/polymorph_detector_v1.0.0.pt \
  ai-service-pod:/app/ai/models/

# 2. Restart AI service to load model
kubectl rollout restart deployment/ai-service -n polymorph-lite

# 3. Verify model loaded
curl http://ai-service:3000/version
```

---

## Feature Activation

### Enable Polymorph Discovery

1. Navigate to Admin Panel
2. Go to Settings → Features
3. Enable "Polymorph Discovery v1.0"
4. Configure AI service URL if not using default
5. Test detection with sample spectrum

### Configure Tier-1 Hardware

1. Install drivers on worker nodes:
   ```bash
   # NI DAQ
   ssh worker-node
   sudo apt install ni-daqmx

   # Ocean Optics
   pip install seabreeze
   seabreeze_os_setup
   ```

2. Verify device discovery:
   ```python
   from retrofitkit.drivers.discovery import get_discovery_service
   service = get_discovery_service()
   devices = service.discover_all()
   print(devices)
   ```

### Enable Operator Wizard

1. Update frontend routing (automatic in v4.0)
2. Access at: `https://your-domain.com/operator-wizard`
3. Test workflow execution
4. Verify signature capture

---

## Troubleshooting

### Database Migration Fails

**Symptom**: Alembic migration errors

**Solution**:
```bash
# Check current revision
alembic current

# If stuck, manually apply
alembic stamp head
alembic upgrade head

# Verify tables created
psql -U polymorph -c "\d polymorph_events"
```

### Pods Not Starting

**Symptom**: CrashLoopBackOff

**Solution**:
```bash
# Check logs
kubectl logs pod-name -n polymorph-lite

# Common issues:
# 1. Missing secrets
kubectl get secrets -n polymorph-lite

#2. Resource limits too low
kubectl describe pod pod-name -n polymorph-lite

# 3. Image pull errors
kubectl describe pod pod-name -n polymorph-lite | grep -A5 Events
```

### AI Service Not Loading Model

**Symptom**: "Polymorph model not available" error

**Solution**:
```bash
# Verify model file exists
kubectl exec -it ai-service-pod -- ls -l /app/ai/models/

# Check model_version.json
kubectl exec -it ai-service-pod -- cat /app/ai/model_version.json

# If missing, copy model
kubectl cp ai/models/polymorph_detector_v1.0.0.pt \
  ai-service-pod:/app/ai/models/

# Restart pod
kubectl delete pod ai-service-pod
```

---

## Rollback Procedure

If upgrade fails, rollback to v3.0:

```bash
# 1. Stop v4.0 pods
kubectl scale deployment polymorph-backend --replicas=0 -n polymorph-lite

# 2. Restore database from backup
kubectl exec -i postgres-0 -n polymorph-lite -- \
  psql -U polymorph polymorph_lite < backup_v3_YYYYMMDD.sql

# 3. Downgrade Alembic
alembic downgrade -1

# 4. Deploy v3.0 images
kubectl set image deployment/polymorph-backend \
  backend=your-registry/polymorph-backend:v3.0 \
  -n polymorph-lite

# 5. Restore configurations
kubectl apply -f configmaps_backup.yaml
kubectl apply -f secrets_backup.yaml

# 6. Scale up
kubectl scale deployment polymorph-backend --replicas=3 -n polymorph-lite
```

---

## Post-Upgrade Tasks

### Validation

- [ ] Run test suite: `pytest tests/ -v`
- [ ] Execute sample workflow
- [ ] Test Polymorph Explorer UI
- [ ] Verify Operator Wizard
- [ ] Check audit logs
- [ ] Test electronic signatures

### Documentation

- [ ] Update runbooks
- [ ] Train operators on new features
- [ ] Update SOPs
- [ ] Document any customizations

### Monitoring

- [ ] Verify Prometheus scraping
- [ ] Check Grafana dashboards
- [ ] Confirm alerts working
- [ ] Review CloudWatch logs

---

## Performance Tuning

After upgrade, optimize for your workload:

```yaml
# Increase replicas for high load
kubectl scale deployment polymorph-backend --replicas=5

# Adjust resource limits
resources:
  requests:
    memory: "512Mi"  # Increase if needed
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"

# Enable autoscaling
kubectl autoscale deployment polymorph-backend \
  --cpu-percent=70 --min=3 --max=10
```

---

## Support

If you encounter issues:

1. Check logs: `kubectl logs -f deployment/polymorph-backend`
2. Review GitHub issues: https://github.com/your-org/issues
3. Contact support: support@polymorph-lite.io
4. Slack: #polymorph-upgrades

---

## Appendix: Environment Variables

New in v4.0:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POLYMORPH_DISCOVERY_ENABLED` | No | `false` | Enable Polymorph Discovery feature |
| `AI_MODEL_VERSION` | No | `1.0.0` | AI model version to load |
| `OPERATOR_WIZARD_ENABLED` | No | `true` | Enable Operator Wizard UI |
| `TIER1_AUTO_DISCOVERY` | No | `true` | Auto-discover Tier-1 hardware |

---

**Upgrade Complete!** Your POLYMORPH_LITE system is now running v4.0 with all new features enabled.
