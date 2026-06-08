"""
Minimal Colab helper for Stable Diffusion WebUI Forge - Neo.

Goals:
- install Haoming02/sd-webui-forge-classic branch "neo" with uv/Python 3.13
- download models from Civitai, Hugging Face, GitHub, direct URLs, or Google Drive
- expose the local WebUI with Gradio, ngrok, or Cloudflare tunnel
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

try:
    from IPython import get_ipython
    from IPython.core.magic import register_line_magic
except Exception:  # pragma: no cover - used outside IPython too
    get_ipython = None
    register_line_magic = None


ROOT = Path(os.environ.get("COLAB_ROOT", "/content"))
WEBUI = Path(os.environ.get("FORGE_NEO_HOME", ROOT / "sd-webui-forge-neo"))
VENV = WEBUI / "venv"
BIN = ROOT / "bin"
MODEL_TMP = ROOT / "models-tmp"

REPO_URL = "https://github.com/Haoming02/sd-webui-forge-classic"
REPO_REF = os.environ.get("FORGE_NEO_REF", "latest-tag")
PYTHON_VERSION = "3.13.12"

MODEL_DIRS = {
    "ckpt": WEBUI / "models" / "Stable-diffusion",
    "checkpoint": WEBUI / "models" / "Stable-diffusion",
    "checkpoints": WEBUI / "models" / "Stable-diffusion",
    "lora": WEBUI / "models" / "Lora",
    "loras": WEBUI / "models" / "Lora",
    "vae": WEBUI / "models" / "VAE",
    "embeddings": WEBUI / "models" / "embeddings",
    "upscaler": WEBUI / "models" / "ESRGAN",
    "upscalers": WEBUI / "models" / "ESRGAN",
    "text_encoder": WEBUI / "models" / "text_encoder",
    "unet": WEBUI / "models" / "unet",
}


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(shlex.quote(str(part)) for part in cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def capture(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, text=True).strip()


def ensure_dirs() -> None:
    BIN.mkdir(parents=True, exist_ok=True)
    MODEL_TMP.mkdir(parents=True, exist_ok=True)
    for path in set(MODEL_DIRS.values()):
        path.mkdir(parents=True, exist_ok=True)


def export_ipython_vars() -> None:
    ensure_dirs()
    ip = get_ipython() if get_ipython else None
    if not ip:
        return

    values = {
        "WebUI": WEBUI,
        "Models": WEBUI / "models",
        "CKPT": MODEL_DIRS["ckpt"],
        "LORA": MODEL_DIRS["lora"],
        "VAE": MODEL_DIRS["vae"],
        "Embeddings": MODEL_DIRS["embeddings"],
        "Upscalers": MODEL_DIRS["upscalers"],
        "TEXT_ENCODER": MODEL_DIRS["text_encoder"],
        "UNET": MODEL_DIRS["unet"],
        "MODEL_TMP": MODEL_TMP,
    }
    ip.user_ns.update(values)

    for key, value in values.items():
        os.environ[key] = str(value)

    print("Forge-Neo paths exported:")
    for key in ["WebUI", "CKPT", "LORA", "VAE", "Embeddings", "Upscalers"]:
        print(f"  {key} = {values[key]}")


def install_system_tools() -> None:
    run(["apt-get", "-qq", "update"])
    run(["apt-get", "-qq", "-y", "install", "aria2", "git", "git-lfs", "curl", "tar"])
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "uv", "gdown"])


def resolve_repo_ref(ref: str) -> str:
    if ref != "latest-tag":
        return ref

    output = capture(["git", "ls-remote", "--tags", "--sort=-v:refname", REPO_URL])
    for line in output.splitlines():
        tag_ref = line.split()[-1]
        if tag_ref.endswith("^{}"):
            continue
        tag = tag_ref.removeprefix("refs/tags/")
        if tag:
            return tag

    raise RuntimeError("Could not resolve latest Forge-Neo tag.")


def clone_or_update(ref: str = REPO_REF) -> str:
    ref = resolve_repo_ref(ref)
    if (WEBUI / ".git").exists():
        run(["git", "fetch", "--tags", "origin"], cwd=WEBUI)
        run(["git", "checkout", ref], cwd=WEBUI)
        current_branch = capture(["git", "branch", "--show-current"], cwd=WEBUI)
        if current_branch:
            run(["git", "pull", "--ff-only", "origin", current_branch], cwd=WEBUI)
    else:
        WEBUI.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", ref, REPO_URL, str(WEBUI)], check=False)
        if not (WEBUI / ".git").exists():
            run(["git", "clone", "--depth", "1", REPO_URL, str(WEBUI)])
            run(["git", "fetch", "--tags", "origin"], cwd=WEBUI)
            run(["git", "checkout", ref], cwd=WEBUI)

    print(f"Forge-Neo source ref: {ref}")
    return ref


def setup_python() -> None:
    run([sys.executable, "-m", "uv", "python", "install", PYTHON_VERSION])
    run([sys.executable, "-m", "uv", "venv", str(VENV), "--python", "3.13", "--seed"])


def install(ref: str = REPO_REF) -> None:
    install_system_tools()
    resolved_ref = clone_or_update(ref)
    setup_python()
    ensure_dirs()
    export_ipython_vars()
    checked_out = capture(["git", "rev-parse", "--short", "HEAD"], cwd=WEBUI)
    print(f"\nInstall step complete. Forge-Neo ref: {resolved_ref} ({checked_out})")
    print("launch.py will install Forge-Neo requirements on first launch.")


def civitai_headers(token: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": "CivitaiLink:Automatic1111"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_json(url: str, headers: dict[str, str]) -> dict[str, Any] | None:
    try:
        res = requests.get(url, headers=headers, timeout=30)
        if res.status_code != 200:
            print(f"  Civitai API returned HTTP {res.status_code}: {url}")
            return None
        return res.json()
    except Exception as exc:
        print(f"  Civitai API failed: {exc}")
        return None


def first_civitai_file(data: dict[str, Any], version_id: str | None = None) -> tuple[str | None, str | None]:
    version = None
    versions = data.get("modelVersions")

    if versions:
        if version_id:
            version = next((item for item in versions if str(item.get("id")) == str(version_id)), None)
        version = version or versions[0]
    else:
        version = data

    if not version:
        return None, None

    files = version.get("files", [])
    file_info = next((item for item in files if item.get("downloadUrl")), None)
    if not file_info:
        return None, None

    return file_info.get("downloadUrl"), file_info.get("name") or version.get("name")


def resolve_civitai(url: str, filename: str | None, token: str | None) -> tuple[str | None, str | None]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    headers = civitai_headers(token)

    if "/api/download/models/" in parsed.path:
        version_id = parsed.path.rstrip("/").split("/")[-1]
        api = f"https://{host}/api/v1/model-versions/{version_id}"
        data = get_json(api, headers)
        if not data:
            return url, filename
        download_url, detected_name = first_civitai_file(data, version_id)
        return download_url or url, filename or detected_name

    match = re.search(r"/models/(\d+)", parsed.path)
    if not match:
        return url, filename

    model_id = match.group(1)
    version_id = parse_qs(parsed.query).get("modelVersionId", [None])[0]
    api = f"https://{host}/api/v1/models/{model_id}"
    data = get_json(api, headers)
    if not data:
        return None, None

    download_url, detected_name = first_civitai_file(data, version_id)
    return download_url, filename or detected_name


def normalize_url(url: str, filename: str | None, civitai_token: str | None) -> tuple[str | None, str | None]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if "github.com" in host:
        url = url.replace("/blob/", "/raw/")
        return url, filename or Path(urlparse(url).path).name

    if "huggingface.co" in host:
        url = url.split("?")[0].replace("/blob/", "/resolve/")
        return url, filename or Path(urlparse(url).path).name

    if "civitai.com" in host or "civitai.red" in host:
        return resolve_civitai(url, filename, civitai_token)

    return url, filename or Path(parsed.path).name or None


def download_with_aria2(url: str, dest: Path, filename: str | None, headers: list[str]) -> None:
    cmd = [
        "aria2c",
        "--allow-overwrite=true",
        "--console-log-level=warn",
        "--summary-interval=5",
        "-c",
        "-x16",
        "-s16",
        "-k1M",
    ]
    for header in headers:
        cmd.append(f"--header={header}")
    if filename:
        cmd += ["-o", filename]
    cmd.append(url)
    run(cmd, cwd=dest)


def download_google_drive(url: str, dest: Path, filename: str | None) -> None:
    cmd = ["gdown", "--fuzzy", url]
    if "drive.google.com/drive/folders" in url:
        cmd.insert(1, "--folder")
    if filename:
        cmd += ["-O", str(dest / filename)]
    run(cmd, cwd=dest)


def download_one(
    url: str,
    kind: str,
    filename: str | None = None,
    civitai_token: str | None = None,
    hf_token: str | None = None,
) -> None:
    ensure_dirs()
    dest = MODEL_DIRS.get(kind.lower())
    if not dest:
        raise ValueError(f"Unknown model kind: {kind}. Use one of: {', '.join(sorted(MODEL_DIRS))}")

    parsed = urlparse(url)
    if "drive.google.com" in parsed.netloc.lower():
        download_google_drive(url, dest, filename)
        return

    resolved_url, resolved_name = normalize_url(url, filename, civitai_token)
    if not resolved_url:
        print(f"  Skipped: could not resolve {url}")
        return

    headers = ["User-Agent: Mozilla/5.0"]
    if hf_token and "huggingface.co" in urlparse(resolved_url).netloc.lower():
        headers.append(f"Authorization: Bearer {hf_token}")
    if civitai_token and ("civitai.com" in resolved_url or "civitai.red" in resolved_url):
        headers.append(f"Authorization: Bearer {civitai_token}")

    print(f"\nDownloading {kind}: {resolved_name or resolved_url}")
    print(f"  -> {dest}")
    download_with_aria2(resolved_url, dest, resolved_name, headers)


def iter_manifest_items(data: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    aliases = {
        "checkpoint": "ckpt",
        "checkpoints": "ckpt",
        "ckpt": "ckpt",
        "lora": "lora",
        "loras": "lora",
        "vae": "vae",
        "embeddings": "embeddings",
        "upscaler": "upscalers",
        "upscalers": "upscalers",
        "text_encoder": "text_encoder",
        "unet": "unet",
    }
    items: list[tuple[str, str, str | None]] = []

    for key, value in data.items():
        kind = aliases.get(key.lower())
        if not kind:
            continue
        entries = value if isinstance(value, list) else [value]
        for entry in entries:
            if isinstance(entry, str):
                items.append((kind, entry, None))
            elif isinstance(entry, dict) and entry.get("url"):
                items.append((kind, entry["url"], entry.get("name")))

    return items


def download_manifest(path: Path, civitai_token: str | None = None, hf_token: str | None = None) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for kind, url, name in iter_manifest_items(data):
        download_one(url, kind, name, civitai_token, hf_token)


def install_cloudflared() -> Path:
    binary = BIN / "cloudflared"
    if binary.exists():
        return binary

    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    run(["curl", "-L", "-o", str(binary), url])
    mode = binary.stat().st_mode
    binary.chmod(mode | stat.S_IEXEC)
    return binary


def install_ngrok() -> Path:
    binary = BIN / "ngrok"
    if binary.exists():
        return binary

    archive = BIN / "ngrok.tgz"
    url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
    run(["curl", "-L", "-o", str(archive), url])
    run(["tar", "-xzf", str(archive), "-C", str(BIN), "ngrok"])
    archive.unlink(missing_ok=True)
    mode = binary.stat().st_mode
    binary.chmod(mode | stat.S_IEXEC)
    return binary


def install_gradio_tunnel() -> Path:
    script = BIN / "gradio-tunnel.py"
    if script.exists():
        return script

    url = "https://raw.githubusercontent.com/gutris1/segsmaker/main/script/gradio-tunnel.py"
    run(["curl", "-L", "-o", str(script), url])
    return script


def stream_tunnel_output(proc: subprocess.Popen[str], pattern: str) -> None:
    regex = re.compile(pattern)
    seen_url = False

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        match = regex.search(line)
        if match:
            seen_url = True
            print(f"\nPUBLIC URL: {match.group(0)}\n")
        elif not seen_url:
            print(line)


def start_cloudflared(port: int) -> subprocess.Popen[str]:
    binary = install_cloudflared()
    proc = subprocess.Popen(
        [str(binary), "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=stream_tunnel_output,
        args=(proc, r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com"),
        daemon=True,
    ).start()
    return proc


def start_gradio_tunnel(port: int) -> subprocess.Popen[str]:
    script = install_gradio_tunnel()
    proc = subprocess.Popen(
        [str(VENV / "bin" / "python"), str(script), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=stream_tunnel_output,
        args=(proc, r"https://[\w-]+\.gradio\.live"),
        daemon=True,
    ).start()
    return proc


def start_ngrok(port: int, token: str | None) -> subprocess.Popen[str]:
    binary = install_ngrok()
    if not token:
        raise ValueError("ngrok tunnel requires --ngrok-token or NGROK_TOKEN env var.")

    run([str(binary), "config", "add-authtoken", token])
    proc = subprocess.Popen(
        [str(binary), "http", f"http://127.0.0.1:{port}", "--log", "stdout"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=stream_tunnel_output,
        args=(proc, r"https://[\w-]+\.ngrok-free\.[\w.-]+"),
        daemon=True,
    ).start()
    return proc


def warn_if_no_auth(args: list[str]) -> None:
    has_auth = any(arg == "--gradio-auth" or arg.startswith("--gradio-auth=") for arg in args)
    if has_auth:
        return
    print(
        "\nWARNING: launching a public tunnel without --gradio-auth.\n"
        "For private or NSFW use, add: -- --gradio-auth user:strong_password\n"
    )


def launch(
    args: list[str],
    port: int = 7860,
    tunnel: str = "gradio",
    ngrok_token: str | None = None,
) -> None:
    ensure_dirs()
    export_ipython_vars()

    if not (WEBUI / "launch.py").exists():
        raise RuntimeError("Forge-Neo is not installed yet. Run: %run script/forge_neo_colab.py install")

    if tunnel != "none":
        warn_if_no_auth(args)

    tunnel_proc = None
    if tunnel == "gradio":
        tunnel_proc = start_gradio_tunnel(port)
        time.sleep(2)
    elif tunnel == "ngrok":
        tunnel_proc = start_ngrok(port, ngrok_token or os.environ.get("NGROK_TOKEN"))
        time.sleep(2)
    elif tunnel == "cloudflared":
        tunnel_proc = start_cloudflared(port)
        time.sleep(2)
    elif tunnel != "none":
        raise ValueError("Supported tunnels: gradio, ngrok, cloudflared, none.")

    default_args = [
        "--uv",
        "--listen",
        "--port",
        str(port),
        "--cuda-malloc",
        "--cuda-stream",
        "--enable-insecure-extension-access",
        "--disable-console-progressbars",
        "--theme",
        "dark",
    ]
    cmd = [str(VENV / "bin" / "python"), "launch.py", *default_args, *args]

    try:
        run(cmd, cwd=WEBUI)
    finally:
        if tunnel_proc and tunnel_proc.poll() is None:
            tunnel_proc.terminate()


if register_line_magic:

    @register_line_magic
    def fn_paths(_: str = "") -> None:
        export_ipython_vars()

    @register_line_magic
    def fn_download(line: str) -> None:
        """
        Usage:
          %fn_download ckpt https://...
          %fn_download lora https://... optional-name.safetensors
        """
        parts = shlex.split(line)
        if len(parts) < 2:
            print("Usage: %fn_download <ckpt|lora|vae|embeddings|upscalers> <url> [filename]")
            return
        kind, url = parts[0], parts[1]
        name = parts[2] if len(parts) >= 3 else None
        download_one(
            url,
            kind,
            name,
            os.environ.get("CIVITAI_TOKEN"),
            os.environ.get("HF_TOKEN"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge-Neo Colab helper")
    sub = parser.add_subparsers(dest="command", required=True)

    i = sub.add_parser("install")
    i.add_argument("--ref", default=os.environ.get("FORGE_NEO_REF", REPO_REF))
    sub.add_parser("paths")

    d = sub.add_parser("download")
    d.add_argument("--manifest", type=Path)
    d.add_argument("--kind")
    d.add_argument("--url")
    d.add_argument("--name")
    d.add_argument("--civitai-token", default=os.environ.get("CIVITAI_TOKEN"))
    d.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))

    l = sub.add_parser("launch")
    l.add_argument("--port", type=int, default=7860)
    l.add_argument("--tunnel", choices=["gradio", "ngrok", "cloudflared", "none"], default="gradio")
    l.add_argument("--ngrok-token", default=os.environ.get("NGROK_TOKEN"))
    l.add_argument("launch_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.command == "install":
        install(args.ref)
    elif args.command == "paths":
        export_ipython_vars()
    elif args.command == "download":
        if args.manifest:
            download_manifest(args.manifest, args.civitai_token, args.hf_token)
        elif args.kind and args.url:
            download_one(args.url, args.kind, args.name, args.civitai_token, args.hf_token)
        else:
            raise SystemExit("download requires --manifest or --kind plus --url")
    elif args.command == "launch":
        launch_args = args.launch_args
        if launch_args and launch_args[0] == "--":
            launch_args = launch_args[1:]
        launch(launch_args, args.port, args.tunnel, args.ngrok_token)


if __name__ == "__main__":
    main()
