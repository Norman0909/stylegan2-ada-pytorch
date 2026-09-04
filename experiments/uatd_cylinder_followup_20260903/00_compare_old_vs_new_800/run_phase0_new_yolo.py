import csv
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer


OUT = Path(r"C:\C2\Document\sonar_research\generate_model\stylegan2-ada-pytorch\experiments\uatd_cylinder_followup_20260903\00_compare_old_vs_new_800\downstream_yolo")
DETECT_ROOT = Path(r"C:\C2\Document\sonar_research\detect_model\yolo\yolov26\yolo_26_11_8_sonar")
DATA = OUT / "datasets" / "new_augmented_effective800" / "data_aug.yaml"
PROJECT = OUT / "yolo_runs"


def read_results_csv_without_polars(self):
    """Avoid the environment's incompatible Polars CPU feature probe."""
    try:
        with self.csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        return {key: [row[key] for row in rows] for key in rows[0]}
    except Exception:
        return {}


def main():
    BaseTrainer.read_results_csv = read_results_csv_without_polars
    weights = DETECT_ROOT / "weights" / "yolo26n.pt"
    print(f"Pretrained: {weights}")
    print(f"Dataset:    {DATA}")
    print(f"Project:    {PROJECT}")
    model = YOLO("yolo26n.yaml", task="detect", verbose=True)
    results = model.train(
        data=str(DATA),
        epochs=300,
        patience=50,
        batch=24,
        imgsz=640,
        device=0,
        workers=4,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        pretrained=str(weights),
        seed=42,
        deterministic=True,
        project=str(PROJECT),
        name="new_augmented_effective800_retry",
        exist_ok=True,
    )
    print(f"Best model: {results.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
