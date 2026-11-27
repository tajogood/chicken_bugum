import argparse
import os
import shutil
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from ultralytics import YOLO, settings
import torch
import yaml

ROOT = Path(__file__).resolve().parent
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    data = Path(args.data).expanduser().resolve()
    if not data.is_file():
        raise SystemExit("dataset/data.yaml missing. Download YOLOv8 export first; see README.")
    config = yaml.safe_load(data.read_text())
    config["path"] = str(data.parent)
    for key in ("train", "val", "test"):
        if key not in config:
            continue
        raw = config[key]
        if not isinstance(raw, str):
            raise SystemExit("Expected standard Roboflow YOLO directory export")
        candidates = [data.parent / raw, data.parent / raw.removeprefix("../")]
        found = next((x.resolve() for x in candidates if x.is_dir()), None)
        if found is None:
            raise SystemExit(f"Missing dataset split {key}: {raw}")
        config[key] = str(found)
        if not list(found.glob("*")):
            raise SystemExit(f"Empty split: {found}")
    normalized = data.parent / "local_data.yaml"
    normalized.write_text(yaml.safe_dump(config, allow_unicode=True))
    print("Classes:", config.get("names"))
    settings.update({"sync": False})
    device = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO("yolo11n.pt")
    model.train(data=str(normalized), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                device=device, workers=0, amp=False, patience=15, project=str(ROOT / "runs"),
                name="organ", plots=True, seed=42)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise SystemExit("Training did not produce best.pt")
    target = ROOT / "models" / "best.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)
    print("Saved:", target)
    print("Training results:", model.trainer.save_dir)

if __name__ == "__main__":
    main()
