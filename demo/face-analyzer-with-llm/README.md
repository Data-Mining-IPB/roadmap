# Face Pose Analyzer

Analisis ekspresi wajah, postur, dan bahasa tubuh secara lokal menggunakan SmolVLM2 via llama.cpp. Tidak ada data yang dikirim ke server eksternal.

---

## Setup

Install dependensi sistem:
```bash
brew install llama.cpp uv
```

Buka project dan install dependensi Python:
```bash
cd face-pose-analyzer
uv init
uv python pin 3.11
uv add gradio pillow openai
```

---

## Menjalankan

1. Jalankan llama-server (download model otomatis ~500 MB):
```bash
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF \
  --port 8080 --n-gpu-layers 99 --ctx-size 4096 --jinja
```
2. Pada terminal baru, jalankan aplikasi:
```bash
uv run python app.py
```

Buka browser di http://localhost:7860

---

## Pilihan Model

Jika RAM lebih besar (8 GB), gunakan model yang lebih kecil:
```bash
llama-server \
    -hf ggml-org/SmolVLM2-2.2B-Instruct-GGUF \
    --port 8080 \
    --n-gpu-layers 99 \
    --jinja
```

---

## Verifikasi Metal aktif

Setelah server berjalan, cek output terminal. Harus ada baris:

    ggml_metal: loaded library libggml-metal.dylib
    ggml_metal: GPU name: .....

Atau cek health endpoint:

```bash
curl http://localhost:8080/health
```