# Voice Cloner

A simple local web app that clones a voice from a short audio clip and speaks any text you type — powered by [OmniVoice](https://github.com/k2-fsa/OmniVoice).

Runs entirely on your computer. No cloud API key required.

---

## Minimum system requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10/11 (64-bit), Linux, or macOS | Windows 11 or Linux |
| **GPU** | NVIDIA GPU with **8 GB VRAM** (CUDA) | NVIDIA RTX 3060 / 4060 / 5070 or better, **12 GB+ VRAM** |
| **CUDA** | CUDA **12.8** compatible driver | Latest Game Ready / Studio driver |
| **RAM** | 16 GB | 32 GB |
| **Disk space** | **~15 GB** free (models + PyTorch + env) | 25 GB+ SSD |
| **Python** | **3.11 or 3.12** (64-bit) | 3.12 |
| **CPU-only** | Possible but **very slow** — not recommended | Use a CUDA GPU |

### Notes

- Apple Silicon (MPS) and Intel Arc (XPU) can work via OmniVoice, but this guide focuses on **NVIDIA CUDA** (best supported path).
- First model download is several gigabytes and needs a stable internet connection.
- Closing other GPU-heavy apps (games, other AI tools) helps avoid out-of-memory errors.

---

## What you get

1. Open a local page in your browser  
2. Upload a **3–10 second** clear voice clip  
3. Type the text you want spoken  
4. Click **Generate speech** and download a WAV file  

---

## Install (Windows — step by step)

### 1. Install Git

Download and install Git from: https://git-scm.com/download/win  

Restart the terminal after installing.

### 2. Install Python 3.12

Download from: https://www.python.org/downloads/  

During setup, check **"Add python.exe to PATH"**.

Verify:

```bat
python --version
```

You should see something like `Python 3.12.x`.

### 3. Install an NVIDIA GPU driver

Update your driver from: https://www.nvidia.com/Download/index.aspx  

Verify GPU:

```bat
nvidia-smi
```

You should see your GPU name and driver version.

### 4. Clone this repository

```bat
git clone https://github.com/OsamaAtiq12/voice-cloner.git
cd voice-cloner
```

> If the GitHub URL differs after you fork/rename, use your repo URL instead.

### 5. Create a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

Your prompt should show `(.venv)`.

### 6. Upgrade pip

```bat
python -m pip install --upgrade pip
```

### 7. Install PyTorch with CUDA 12.8

```bat
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA works:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

You want `True` and your GPU name printed.

### 8. Install Voice Cloner dependencies

```bat
pip install -r requirements.txt
```

### 9. Download the OmniVoice model

Models are **not** included in the git repo (they are large). Download them once:

```bat
python scripts\download_models.py
```

This saves weights into `models\OmniVoice` (and the helper ASR model OmniVoice uses to read your reference clip).

If Hugging Face is slow or blocked in your region:

```bat
set HF_ENDPOINT=https://hf-mirror.com
python scripts\download_models.py
```

### 10. Start the app

Double-click `start.bat`, **or** run:

```bat
.venv\Scripts\activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 7870
```

### 11. Open in your browser

Go to: [http://127.0.0.1:7870](http://127.0.0.1:7870)

The first launch may take 1–2 minutes while the model loads into GPU memory.

---

## Install (Linux / macOS)

```bash
git clone https://github.com/OsamaAtiq12/voice-cloner.git
cd voice-cloner

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# NVIDIA Linux (CUDA 12.8):
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

# Apple Silicon (no CUDA wheel):
# pip install torch torchaudio

pip install -r requirements.txt
python scripts/download_models.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 7870
```

Open [http://127.0.0.1:7870](http://127.0.0.1:7870).

---

## How to use (after install)

1. In **Text to speak**, type what you want said.  
2. In **Reference audio**, drop a clear **3–10 second** clip of the voice.  
3. Adjust **Speaking controls** if you want (defaults are fine).  
4. Click **Generate speech**.  
5. Play the result or download the WAV.  

More detail: open **How it works** in the app navbar, or visit `/how-it-works`.

---

## Project layout

```text
voice_cloner/
  app/                 # FastAPI backend + OmniVoice engine
  static/              # Web UI
  scripts/             # download_models.py
  models/              # Created after download (not in git)
  start.bat            # Windows launcher
  requirements.txt
  README.md
```

---

## Troubleshooting

**`torch.cuda.is_available()` is False**  
- Update NVIDIA driver  
- Reinstall the CUDA 12.8 PyTorch wheels from step 7  
- Confirm `nvidia-smi` works in the same terminal  

**Out of memory / CUDA OOM**  
- Close other GPU apps  
- Lower **Quality steps** to `16`  
- Use a shorter reference clip and shorter text  

**Model download fails**  
- Check internet / firewall  
- Try `HF_ENDPOINT=https://hf-mirror.com`  
- Re-run `python scripts/download_models.py`  

**Port 7870 already in use**  
- Change the port:  
  `python -m uvicorn app.main:app --host 127.0.0.1 --port 7871`

**App says model not found**  
- Run `python scripts/download_models.py` again  
- Confirm `models/OmniVoice/model.safetensors` exists  

---

## Responsible use

Only clone voices you own or have clear permission to use. Do not use this tool for scams, impersonation, or anything illegal.

---

## Credits

- TTS model: [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)  
- App: local Voice Cloner studio UI on FastAPI  
