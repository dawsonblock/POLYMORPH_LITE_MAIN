# 🧬 POLYMORPH-LITE on Google Colab

Quick start notebooks to run POLYMORPH-LITE Lab OS on Google Colab.

## 🚀 Quick Start

1. Open `POLYMORPH_LITE_Colab.ipynb` in Google Colab
2. Run each cell in order
3. Access the API via the ngrok public URL

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dawsonblock/POLYMORPH_LITE_MAIN/blob/main/colab/POLYMORPH_LITE_Colab.ipynb)

## 📦 What's Included

| Section | Description |
|---------|-------------|
| Setup | Clone repo, install dependencies |
| Database | Initialize SQLite database |
| API Server | Start FastAPI with ngrok tunnel |
| PMM Demo | Test AI brain with synthetic spectra |
| Gating Demo | Test threshold-based gating |
| Safety Demo | Test safety guardrails |

## ⚠️ Limitations

- Uses SQLite instead of PostgreSQL
- Redis disabled (no caching)
- No persistent storage between sessions
- ngrok free tier has rate limits

## 💡 Tips

1. **Save checkpoints**: Download `/content/POLYMORPH_LITE_MAIN/checkpoints/` before session ends
2. **ngrok auth**: Sign up at ngrok.com for higher rate limits
3. **GPU**: Enable GPU runtime for faster PyTorch operations

## 🔗 Links

- [Main Repository](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN)
- [API Documentation](https://github.com/dawsonblock/POLYMORPH_LITE_MAIN#api)
