"""
SightSeeker - fine-tune YOLO on your own objects
==================================================

When does training help?
  * The pretrained model already learned the 80 COCO classes (bottle, cup, phone, person...) from
    ~118k labelled images. Re-training those classes on a small free dataset usually makes it WORSE.
    For better accuracy on those, use a bigger pretrained model instead:  yolo_detection.py --model l
  * Fine-tuning DOES help for objects COCO doesn't know (your robot's target, a red barrel, a
    specific box) or for your exact camera/lighting. ~50-150 labelled photos per object is a good start.

Dataset format: standard YOLO ("Ultralytics") format, described by a data.yaml such as:

    path: datasets/my_objects
    train: images/train
    val: images/val
    names:
      0: red barrel
      1: blue crate

Free sources: label your own photos with a free tool (makesense.ai, Label Studio, CVAT) and export
"YOLO" format; or download a YOLO-format dataset from Roboflow Universe / GitHub. Check the licence.

Train (a GPU is strongly recommended - Google Colab's free GPU works, see train_on_colab.ipynb):
    python train_custom.py --data datasets/my_objects/data.yaml
    python train_custom.py --data coco8.yaml --epochs 3        # 8-image smoke test (auto-downloads from GitHub)

Then use it:
    python yolo_detection.py --model runs/detect/sightseeker/weights/best.pt --goal "find the red barrel"
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Fine-tune YOLO for SightSeeker")
    ap.add_argument("--data", required=True, help="dataset yaml (YOLO format), e.g. datasets/my_objects/data.yaml or coco8.yaml")
    ap.add_argument("--model", default="yolo26s.pt", help="starting checkpoint (default yolo26s.pt, downloaded automatically)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="lower it (8 or 4) if you run out of memory")
    ap.add_argument("--freeze", type=int, default=10, help="freeze the first N layers (good for small datasets)")
    ap.add_argument("--device", default=None, help="'0' for the first GPU, 'cpu' to force CPU (very slow)")
    ap.add_argument("--name", default="sightseeker")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("Install first:  pip install ultralytics")
        sys.exit(1)

    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        print("WARNING: no GPU found - training on CPU takes hours. Use Google Colab (train_on_colab.ipynb).")

    start = args.model if os.path.exists(args.model) else os.path.join(SCRIPT_DIR, args.model)
    model = YOLO(start if os.path.exists(start) else args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        device=args.device,
        patience=15,          # stop early when validation stops improving
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    metrics = model.val(data=args.data, device=args.device)
    best = os.path.join(str(model.trainer.save_dir), "weights", "best.pt")
    print("\n================ RESULT ================")
    print(f"mAP50-95: {metrics.box.map:.3f}   mAP50: {metrics.box.map50:.3f}")
    print(f"Best weights: {best}")
    print(f'Use them:  python yolo_detection.py --model "{best}"')


if __name__ == "__main__":
    main()
