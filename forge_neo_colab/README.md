# Forge-Neo Colab

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoshikawaryuukou/my-colab/blob/main/forge_neo_colab/Forge_Neo_COLAB.ipynb)

Minimal flow for running Stable Diffusion WebUI Forge - Neo on Colab.

## Files

```text
forge_neo_colab/
  Forge_Neo_COLAB.ipynb   Colab notebook entry point
  forge_neo_colab.py      install, download, launch helper
  models.example.json     model manifest example
  README.md               usage notes
```

## 1. Load Helper

```python
!curl -sLo /content/forge_neo_colab.py https://raw.githubusercontent.com/hoshikawaryuukou/my-colab/main/forge_neo_colab/forge_neo_colab.py
%run /content/forge_neo_colab.py paths
```

When testing from an uploaded local copy, upload `forge_neo_colab/forge_neo_colab.py` to `/content`
and run the same `%run` command.

## 2. Install Forge-Neo

```python
%run /content/forge_neo_colab.py install
```

This installs system tools, installs `uv`, clones:

```text
https://github.com/Haoming02/sd-webui-forge-classic
```

By default it resolves the latest Git tag at install time. You can also pin a
known-good tag, branch, or commit SHA:

```python
%run /content/forge_neo_colab.py install --ref 2.24
%run /content/forge_neo_colab.py install --ref neo
```

Then it creates:

```text
/content/sd-webui-forge-neo/venv
```

using Python 3.13.

## 3. Optional Tokens

```python
import os
os.environ["CIVITAI_TOKEN"] = "..."
os.environ["HF_TOKEN"] = "..."
```

Use these only when downloading gated/private models.

## 4. Download Models Directly

```python
%fn_download ckpt https://huggingface.co/user/repo/resolve/main/model.safetensors
%fn_download lora https://civitai.com/models/122359/detail-tweaker-xl
%fn_download vae https://drive.google.com/file/d/.../view?usp=sharing
```

Available destinations:

```text
ckpt        -> /content/sd-webui-forge-neo/models/Stable-diffusion
lora        -> /content/sd-webui-forge-neo/models/Lora
vae         -> /content/sd-webui-forge-neo/models/VAE
embeddings  -> /content/sd-webui-forge-neo/models/embeddings
upscalers   -> /content/sd-webui-forge-neo/models/ESRGAN
```

## 5. Download From Manifest

Create `/content/models.json`:

```json
{
  "checkpoints": [
    "https://huggingface.co/user/repo/resolve/main/model.safetensors"
  ],
  "loras": [
    {
      "url": "https://civitai.com/models/122359/detail-tweaker-xl",
      "name": "detail-tweaker-xl.safetensors"
    }
  ],
  "vae": [
    "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
  ]
}
```

Then run:

```python
%run /content/forge_neo_colab.py download --manifest /content/models.json
```

## 6. Mount Google Drive

For direct import from Drive folders:

```python
from google.colab import drive
drive.mount("/content/drive")

!ln -s /content/drive/MyDrive/ForgeNeo/checkpoints/drive_ckpt "$CKPT/drive_ckpt"
!ln -s /content/drive/MyDrive/ForgeNeo/loras/drive_lora "$LORA/drive_lora"
!ln -s /content/drive/MyDrive/ForgeNeo/vae/drive_vae "$VAE/drive_vae"
```

Or download individual Drive share links with `%fn_download`.

## 7. Launch With Gradio Tunnel

```python
%run /content/forge_neo_colab.py launch
```

Wait for:

```text
PUBLIC URL: https://xxxx.gradio.live
```

Open that URL to use Forge-Neo.

The default tunnel is Gradio. For NSFW or private use, add WebUI auth:

```python
%run /content/forge_neo_colab.py launch -- --gradio-auth me:some-long-random-password
```

To use ngrok instead:

```python
import os
os.environ["NGROK_TOKEN"] = "..."

%run /content/forge_neo_colab.py launch --tunnel ngrok -- --gradio-auth me:some-long-random-password
```

Cloudflare is still available as a fallback:

```python
%run /content/forge_neo_colab.py launch --tunnel cloudflared -- --gradio-auth me:some-long-random-password
```

Extra launch args can be appended:

```python
%run /content/forge_neo_colab.py launch -- --xformers
```
