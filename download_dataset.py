from pathlib import Path
from getpass import getpass
import os
from roboflow import Roboflow

def main():
    root = Path(__file__).resolve().parent / "dataset"
    if (root / "data.yaml").is_file():
        print("Dataset already exists:", root)
        return
    key = os.getenv("ROBOFLOW_API_KEY") or getpass("Roboflow dataset download API key: ")
    project = Roboflow(api_key=key).workspace("na-u0cmg").project("biovision_chicken")
    project.version(1).download("yolov8", location=str(root))
    if not (root / "data.yaml").is_file():
        raise SystemExit("Dataset download did not produce data.yaml")

if __name__ == "__main__":
    main()
