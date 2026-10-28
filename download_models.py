"""Download the detection models once (needs internet), so the demo runs offline afterwards.
Called by launch.bat option 6.  Safe to re-run: models that are already here are skipped.

  NVIDIA GPU : YOLO26l (default on a GPU) + YOLO26x ("maximum accuracy") + YOLO26n (CPU fallback)
  CPU only   : YOLO26n (default on a CPU) + YOLO26s, plus a 2-3x faster OpenVINO copy of YOLO26n
  Both       : YOLOE-26s (open-vocabulary mode) and its text encoder
               + the offline speech model for voice goals (Vosk small English, ~40 MB)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)   # Ultralytics saves some downloads (e.g. the YOLOE text encoder) in the current folder


def ultralytics_supports_yolo26():
    """Checked in a separate process so an upgrade below can't conflict with an already-imported copy."""
    probe = ("from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES as N; "
             "print('yolo26n.pt' in N and 'yoloe-26s-seg.pt' in N)")
    try:
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=180)
        return out.stdout.strip().endswith("True")
    except Exception:  # noqa: BLE001
        return False


def has_cuda():
    """True only if PyTorch can really compute on the NVIDIA GPU (an RTX 50 card with an older
    PyTorch build is 'available' but fails on the first operation)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        (torch.ones(8, device="cuda") * 2).sum().item()
        return True
    except Exception:  # noqa: BLE001
        return False


if ultralytics_supports_yolo26():
    print("Ultralytics already supports YOLO26 + YOLOE - keeping the installed version.")
else:
    print("Updating Ultralytics (YOLO26 / YOLOE need a recent version)...")
    subprocess.call([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])

from ultralytics import YOLO  # noqa: E402  (import after a possible upgrade)

gpu = has_cuda()
print(f"\nGraphics: {'NVIDIA GPU found - getting the large models' if gpu else 'no usable NVIDIA GPU - getting the fast CPU models'}")
wanted = ["yolo26n.pt"] + (["yolo26l.pt", "yolo26x.pt"] if gpu else ["yolo26s.pt"])
status = {}
for name in wanted:
    path = os.path.join(HERE, name)
    try:
        if not os.path.exists(path):
            print(f"  downloading {name} ...")
        YOLO(path)
        status[name] = "OK"
    except Exception as e:  # noqa: BLE001
        status[name] = f"FAILED ({e})"
if status.get("yolo26n.pt") != "OK":
    print("YOLO26 isn't available - getting YOLOv8s as a fallback.")
    try:
        YOLO(os.path.join(HERE, "yolov8s.pt"))
        status["yolov8s.pt"] = "OK"
    except Exception as e:  # noqa: BLE001
        status["yolov8s.pt"] = f"FAILED ({e})"

if not gpu and status.get("yolo26n.pt") == "OK":
    print("Making a fast CPU version of YOLO26n with OpenVINO (2-3x faster on Intel laptops, takes a minute)...")
    try:
        YOLO(os.path.join(HERE, "yolo26n.pt")).export(format="openvino", imgsz=480)
        status["yolo26n_openvino_model"] = "OK"
    except Exception as e:  # noqa: BLE001
        status["yolo26n_openvino_model"] = f"skipped ({e}) - the normal model still works"

print("Open-vocabulary model (YOLOE) + its text encoder...")
try:
    from ultralytics import YOLOE
    m = YOLOE(os.path.join(HERE, "yoloe-26s-seg.pt"))
    try:
        m.set_classes(["red barrel", "person"])
    except TypeError:
        m.set_classes(["red barrel", "person"], m.get_text_pe(["red barrel", "person"]))
    status["yoloe-26s-seg.pt"] = "OK"
except Exception as e:  # noqa: BLE001
    status["yoloe-26s-seg.pt"] = f"FAILED ({e}) - the normal mode still works"

VOSK_NAME = "vosk-model-small-en-us-0.15"
print("Voice goals: offline speech recognition (Vosk)...")
try:
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "--quiet", "vosk", "sounddevice"])
    if rc != 0:
        raise RuntimeError("pip could not install vosk / sounddevice")
    if not os.path.isdir(os.path.join(HERE, VOSK_NAME)):
        import urllib.request
        import zipfile
        url = f"https://alphacephei.com/vosk/models/{VOSK_NAME}.zip"
        zpath = os.path.join(HERE, VOSK_NAME + ".zip")
        print(f"  downloading {url} (~40 MB) ...")
        urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(HERE)
        os.remove(zpath)
    status[VOSK_NAME] = "OK"
except Exception as e:  # noqa: BLE001
    status[VOSK_NAME] = f"skipped ({e}) - typing goals still works"

print("\n" + "=" * 60)
for name, st in status.items():
    print(f"  {'OK ' if st == 'OK' else '-- '} {name:<26} {'' if st == 'OK' else st}".rstrip())
print("=" * 60)
if all(st == "OK" or st.startswith("skipped") for st in status.values()):
    print("All models are saved in this folder. The demo now works without internet.")
else:
    print("Some downloads failed (see above). Check the internet connection and run option 6 again.")
