# 🧬 Running POLYMORPH-LITE on Google Colab

This guide explains how to run the complete POLYMORPH-LITE Lab OS on Google Colab for free.

## 📋 Prerequisites

- A Google account
- (Optional) An ngrok account for stable public URLs: https://ngrok.com

---

## 🚀 Quick Start (One-Click)

**Click this button to open the notebook directly in Google Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dawsonblock/POLYMORPH_LITE_MAIN/blob/main/colab/POLYMORPH_LITE_Colab.ipynb)

---

## 📖 Step-by-Step Guide

### Step 1: Open the Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Open Notebook**
3. Select the **GitHub** tab
4. Enter: `dawsonblock/POLYMORPH_LITE_MAIN`
5. Select `colab/POLYMORPH_LITE_Colab.ipynb`

### Step 2: Connect to Runtime

1. Click **Connect** in the top-right corner
2. Wait for Colab to allocate a VM (usually 10-30 seconds)
3. You'll see a green checkmark when connected

### Step 3: Run Setup Cells

Run each cell in order by clicking the **Play** button or pressing `Shift+Enter`:

| Cell | Purpose | Time |
|------|---------|------|
| 1️⃣ Clone Repo | Downloads POLYMORPH-LITE code | ~10s |
| 2️⃣ Install Core Deps | Installs FastAPI, PyTorch, etc. | ~30s |
| 3️⃣ Install Requirements | Installs remaining packages | ~60s |
| 4️⃣ Configure Environment | Sets up SQLite, paths, secrets | ~1s |
| 5️⃣ Initialize Database | Creates database tables | ~2s |

### Step 4: Start the Backend Server

1. **(Optional)** Enter your ngrok auth token for stable URLs
2. Run the server cell - you'll see output like:

```
🚀 POLYMORPH-LITE Backend Running!
============================================================

📡 Public URL: https://abc123.ngrok.io
📖 API Docs:   https://abc123.ngrok.io/docs
❤️  Health:     https://abc123.ngrok.io/health
📊 Metrics:    https://abc123.ngrok.io/metrics
```

### Step 5: Test the API

Run the test cells to verify everything works:

- **Health Check**: Should return `{"status": "healthy"}`
- **Metrics**: Should show Prometheus metrics

---

## 🧠 Using the AI (PMM) Brain

The notebook includes demos for:

### PMM (Polymorph Mode Memory)
```python
from pmm_brain import StaticPseudoModeMemory

pmm = StaticPseudoModeMemory(latent_dim=128, max_modes=32)

# Process synthetic spectra
for i in range(50):
    latent = torch.randn(1, 128)
    recon, comp = pmm(latent)
```

### Gating Engine
```python
from retrofitkit.core.gating import GatingEngine

rules = [{"name": "peak", "direction": "above", "threshold": 100, "consecutive": 3}]
gating = GatingEngine(rules, cooldown_sec=5.0)

triggered = gating.update({"t": 0, "peak_intensity": 150})
```

### Safety Guardrails
```python
from retrofitkit.core.safety.guardrails import SafetyGuardrails

safety = SafetyGuardrails(max_intensity=65535)
safe, msg = safety.check_over_intensity(spectrum)
```

---

## 💾 Saving Your Work

### Download Checkpoints

Before your Colab session ends, download your AI checkpoints:

```python
# List checkpoints
!ls -la /content/POLYMORPH_LITE_MAIN/checkpoints/

# Download (will appear in your browser downloads)
from google.colab import files
files.download('/content/POLYMORPH_LITE_MAIN/checkpoints/your_checkpoint.npz')
```

### Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints to Drive
!cp -r /content/POLYMORPH_LITE_MAIN/checkpoints /content/drive/MyDrive/polymorph_checkpoints
```

---

## ⚠️ Colab Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| **12hr session limit** | Session disconnects after 12 hours | Save checkpoints regularly |
| **90min idle timeout** | Disconnects if idle | Keep browser tab active |
| **SQLite only** | No PostgreSQL support | Use for demos, not production |
| **No Redis** | No caching | Acceptable for demos |
| **ngrok rate limits** | Free tier has limits | Sign up for ngrok account |

---

## 🔧 Troubleshooting

### "ModuleNotFoundError"
Run the dependency installation cells again:
```python
!pip install -q fastapi uvicorn sqlalchemy
```

### "Address already in use"
Restart the runtime: **Runtime → Restart runtime**

### "ngrok tunnel not found"
Get a free auth token at https://ngrok.com and enter it in the notebook.

### "Database locked"
This can happen with SQLite. Restart the runtime.

---

## 🔗 Connecting Your Frontend

Use the ngrok URL as your API endpoint in your frontend:

```typescript
// In your Next.js app
const API_URL = "https://your-ngrok-url.ngrok.io";

// Fetch data
const response = await fetch(`${API_URL}/api/v1/workflows`);
```

---

## 📚 Next Steps

1. **Explore the API**: Visit `/docs` for interactive Swagger UI
2. **Train PMM**: Use the calibration demo to train on your data
3. **Test Workflows**: Load and execute the sample workflows
4. **Deploy to Cloud**: When ready, deploy to GCP/AWS with Docker

---

## 🆘 Getting Help

- **GitHub Issues**: https://github.com/dawsonblock/POLYMORPH_LITE_MAIN/issues
- **API Docs**: `{your-url}/docs`

---

*Happy experimenting! 🔬*
