import json
from pathlib import Path

from ultralytics import YOLO


OUT = Path(r"C:\C2\Document\sonar_research\generate_model\stylegan2-ada-pytorch\experiments\uatd_cylinder_followup_20260903\00_compare_old_vs_new_800\downstream_yolo")
DATA = r"C:\C2\Document\sonar_research\dataset\Sonar_Dataset\UATD\processed\dataset.yaml"
DETECT_ROOT = Path(r"C:\C2\Document\sonar_research\detect_model\yolo\yolov26\yolo_26_11_8_sonar")
MODELS = {
    "baseline": DETECT_ROOT / "runs" / "detect" / "sonar_yolo26n" / "weights" / "best.pt",
    "old_augmented_800k": DETECT_ROOT / "runs" / "detect" / "sonar_yolo26n_aug_cylinder_20260822" / "weights" / "best.pt",
    "new_context_effective800": OUT / "yolo_runs" / "new_augmented_effective800_retry" / "weights" / "best.pt",
}
NAMES = [
    "ball", "cube", "human body", "tyre", "square cage",
    "plane", "rov", "circle cage", "cylinder", "metal bucket",
]


def validate(label: str, weights: Path):
    model = YOLO(str(weights), task="detect")
    metrics = model.val(
        data=DATA,
        split="test",
        batch=16,
        imgsz=640,
        device=0,
        workers=4,
        verbose=False,
        plots=False,
        project=str(OUT / "validation_runs"),
        name=label,
        exist_ok=True,
    )
    return {
        "weights": str(weights),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_map50": [float(x) for x in metrics.box.ap50],
        "per_class_map50_95": [float(x) for x in metrics.box.ap],
    }


def main():
    results = {label: validate(label, path) for label, path in MODELS.items()}
    payload = {"data": DATA, "split": "test", "imgsz": 640, "batch": 16, "class_names": NAMES, "results": results}
    output = OUT / "validation_results.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, result in results.items():
        print(label, json.dumps({k: result[k] for k in ("map50", "map50_95", "precision", "recall")}))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
