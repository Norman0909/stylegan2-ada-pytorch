import os
from ultralytics import YOLO

RUNS = r"C:\C2\Document\sonar_research\detect_model\yolo\yolov26\yolo_26_11_8_sonar\runs\detect"
DATA = r"C:\C2\Document\sonar_research\dataset\Sonar_Dataset\SeabedObjectsDetection-KLSG-labelled\processed\dataset.yaml"
OUT = r"c:\C2\Document\sonar_research\generate_model\stylegan2-ada-pytorch\experiments\mine_20260830\yolo_eval"


def main():
    runs = [
        ("baseline",        "yolo26n_baseline_SSS_mine_20260901"),
        ("aug_v2",          "yolo26n_aug_SSS_mine_20260901"),
        ("real_v2",         "yolo26n_real_SSS_mine_20260901"),
        ("aug_v3_smooth",   "yolo26n_aug_smooth_SSS_mine_20260901"),
        ("real_v3_smooth",  "yolo26n_real_smooth_SSS_mine_20260901"),
    ]
    for tag, run in runs:
        w = os.path.join(RUNS, run, "weights", "best.pt")
        if not os.path.exists(w):
            print("==== %s: MISSING %s ====" % (tag, w))
            continue
        m = YOLO(w)
        r = m.val(data=DATA, split="val", imgsz=640, batch=24, device=0, workers=0,
                  project=OUT, name=tag, exist_ok=True, verbose=False)
        print("====", tag, "====")
        names = r.names
        for i, (ap50, ap) in enumerate(zip(r.box.ap50, r.box.ap)):
            print("cls %d (%s): AP50=%.4f AP50-95=%.4f" % (i, names[i], ap50, ap))
        print("overall mAP50=%.4f mAP50-95=%.4f" % (r.box.map50, r.box.map))


if __name__ == "__main__":
    main()
