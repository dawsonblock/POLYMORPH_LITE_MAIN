# Production Deployment Quick Reference

## Pre-Deployment Checklist

- [ ] Copy `.env.production.example` to `.env`
- [ ] Update all `CHANGE_THIS_*` values in `.env`
- [ ] Generate secret key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- [ ] Configure SSL certificates
- [ ] Set up DNS records
- [ ] Configure firewall rules
- [ ] Test backup/restore procedure

## Deployment Commands

### Quick Deploy
```bash
./deploy_production.sh
```

### Manual Steps
```bash
# 1. Run tests
PYTHONPATH=. pytest tests/ -v

# 2. Build frontend
cd gui-v2/frontend && npm run build && cd ../..

# 3. Start services
docker-compose up -d

# 4. Check health
curl http://localhost:8001/health
curl http://localhost:3000/healthz
```

## Post-Deployment

### Verify Services
```bash
docker-compose ps
docker-compose logs -f backend
```

### Access Points
- Frontend: http://localhost:3000
- API: http://localhost:8001/docs
- Grafana: http://localhost:3030

### Monitoring
```bash
# View logs
docker-compose logs -f

# Check metrics
open http://localhost:9090  # Prometheus
open http://localhost:3030  # Grafana
```

## Troubleshooting

### Services Won't Start
```bash
docker-compose logs backend
docker-compose logs ai-service
```

### Health Check Fails
```bash
docker-compose restart backend
curl -v http://localhost:8001/health
```

### Reset Everything
```bash
docker-compose down -v
rm -rf data/*.db
docker-compose up -d
```

## Security Reminders

1. **Change default passwords** in `.env`
2. **Enable MFA** for all users
3. **Configure SSL/TLS** certificates
4. **Set up firewall** rules
5. **Enable audit logging**
6. **Configure backups**

## Support

- Documentation: `/docs`
- Issues: GitHub Issues
- Emergency: Check logs first, then contact support
