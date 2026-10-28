"""Check whether the Vision Brain can use your GPU, and say how to fix it if not.

    python check_gpu.py            # report
    python check_gpu.py --install  # also install the NVIDIA (CUDA) build of PyTorch if it would help

Exit code (used by launch.bat): 0 = GPU ready, 2 = NVIDIA GPU found but PyTorch is CPU-only, 1 = no usable NVIDIA GPU.
"""
import shutil
import subprocess
import sys

CUDA_INDEX = "https://download.pytorch.org/whl/cu128"   # CUDA 12.8 build: RTX 20/30/40/50 series, GTX 16 series


def nvidia_gpu():
    """Name of the NVIDIA GPU from the driver's nvidia-smi tool, or None."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        line = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
        return line or None
    except Exception:  # noqa: BLE001
        return None


_PROBE = r"""
import json
try:
    import torch
    ok = torch.cuda.is_available()
    kernel = None
    if ok:
        try:      # RTX 50-series (Blackwell) cards need a CUDA 12.8+ build, or every GPU operation fails
            kernel = float((torch.ones(8, device="cuda") * 2).sum().item()) == 16.0
        except Exception as e:
            kernel = str(e)[:200]
    print(json.dumps({"ver": torch.__version__, "cuda": torch.version.cuda, "ok": ok, "kernel": kernel,
                      "name": torch.cuda.get_device_name(0) if ok else None}))
except Exception as e:
    print(json.dumps({"err": str(e)}))
"""


def torch_status():
    """Checked in a separate Python process: if this process imported torch, Windows would lock its DLLs
    and pip could not replace them."""
    import json
    try:
        out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=120)
        info = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return f"couldn't check PyTorch ({e})", False
    if "err" in info:
        return f"PyTorch is not installed ({info['err']})", False
    build = f"CUDA {info['cuda']}" if info["cuda"] else "CPU-only"
    msg = f"PyTorch {info['ver']} ({build} build)"
    if info["ok"] and info.get("kernel") is not True:
        return msg + f" - sees {info['name']} but can't run on it (build too old for this GPU)", "old"
    if info["ok"]:
        msg += f" -> using {info['name']}"
    return msg, info["ok"]


def install_cuda_torch():
    """Downloads the CUDA build first and only then swaps it in, so a failed download (e.g. Wi-Fi drops
    halfway through the ~3 GB) leaves the existing PyTorch untouched."""
    py = [sys.executable, "-m", "pip"]
    print(f"\nInstalling the NVIDIA build of PyTorch from {CUDA_INDEX} (about 3 GB, takes a while).")
    print("Your current PyTorch keeps working until the download has finished.")
    rc = subprocess.call(py + ["install", "--force-reinstall", "--no-deps", "torch", "torchvision", "--index-url", CUDA_INDEX])
    if rc != 0:
        print("The GPU build didn't install (internet connection?). Your existing PyTorch is unchanged.")
        return False
    # add anything the new build needs that isn't installed yet (usually nothing)
    subprocess.call(py + ["install", "torch", "torchvision", "--index-url", CUDA_INDEX])
    return True


def main():
    gpu = nvidia_gpu()
    msg, cuda_ok = torch_status()
    print("=" * 60)
    print(f"  NVIDIA GPU : {gpu or 'none found (nvidia-smi missing)'}")
    print(f"  PyTorch    : {msg}")
    print("=" * 60)
    if cuda_ok is True:
        print("  GPU READY. The Vision Brain will use it automatically (YOLO26l, HD camera, FP16).")
        return 0
    if gpu is None and cuda_ok != "old":
        print("  No NVIDIA GPU detected. Intel/AMD graphics can't run PyTorch here, so the Vision Brain")
        print("  uses the CPU with the fast YOLO26n + OpenVINO model instead (launch.bat option 6).")
        print("  If you DO have an NVIDIA card, install/update its driver from nvidia.com/drivers first.")
        print("  Intel graphics (after option 6): you can try  python yolo_detection.py --device intel:gpu")
        return 1
    if cuda_ok == "old":
        print("  Your PyTorch is too old for this graphics card. RTX 50-series cards (like the RTX 5050)")
        print("  need the CUDA 12.8 build. Reinstalling it fixes this.")
    elif "CUDA" in msg:
        print("  You have an NVIDIA GPU and a GPU build of PyTorch, but PyTorch can't use the card.")
        print("  Update the NVIDIA driver (nvidia.com/drivers or the GeForce/NVIDIA App), restart, and re-run this.")
        return 1
    else:
        print("  You have an NVIDIA GPU, but the installed PyTorch is the CPU-only build")
        print("  (that's what plain 'pip install' gives you on Windows).")
    if "--install" in sys.argv:
        if install_cuda_torch():
            msg, cuda_ok = torch_status()
            print(f"\n  Now: {msg}")
            if cuda_ok is True:
                print("  GPU READY.")
                return 0
            print("  Installed, but PyTorch still can't see the GPU: update the NVIDIA driver and restart.")
        return 2
    print("  Fix: run launch.bat option 6 and answer Y, or run:")
    print(f"     pip install --force-reinstall --no-deps torch torchvision --index-url {CUDA_INDEX}")
    print(f"     pip install torch torchvision --index-url {CUDA_INDEX}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
