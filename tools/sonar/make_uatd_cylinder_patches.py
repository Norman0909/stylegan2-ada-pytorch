#!/usr/bin/env python
"""Build single-channel target-context patches from UATD cylinder labels.

The UATD annotations provide the target bounding box and frame-level sonar
metadata, but no pixel-level acoustic-shadow mask.  This script therefore
builds a deterministic *target-context* patch: a square window centred on the
cylinder, with configurable context around the box.  The surrounding pixels
are retained so a visible shadow can remain in the training sample; the
manifest explicitly records that shadow inclusion is not annotation-proven.

Example:

    python tools/sonar/make_uatd_cylinder_patches.py \
        --uatd-root C:/data/UATD \
        --out experiments/uatd_cylinder_context_256/dataset/train \
        --contact-sheet experiments/uatd_cylinder_context_256/train_preview.png

The output PNGs are grayscale (one channel), square, and ready to be packed
with ``dataset_tool.py`` for StyleGAN2-ADA.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
DEFAULT_CYLINDER_CLASS_ID = 8
SONAR_FIELDS = ("range", "azimuth", "elevation", "soundspeed", "frequency")


def _find_image(images_dir: Path, stem: str) -> Path | None:
    """Find an image for a label stem without assuming a particular extension."""

    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _read_boxes(label_path: Path, width: int, height: int, class_id: int) -> list[dict[str, object]]:
    """Read YOLO boxes for one class and convert them to pixel coordinates."""

    boxes: list[dict[str, object]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            current_class = int(float(fields[0]))
            cx, cy, box_width, box_height = (float(value) for value in fields[1:5])
        except ValueError:
            raise ValueError(f"invalid YOLO row in {label_path}:{line_number}: {line!r}") from None
        if current_class != class_id:
            continue
        if not all(math.isfinite(value) for value in (cx, cy, box_width, box_height)):
            raise ValueError(f"non-finite YOLO row in {label_path}:{line_number}")
        if box_width <= 0 or box_height <= 0:
            continue

        x0 = max(0.0, (cx - box_width / 2.0) * width)
        y0 = max(0.0, (cy - box_height / 2.0) * height)
        x1 = min(float(width), (cx + box_width / 2.0) * width)
        y1 = min(float(height), (cy + box_height / 2.0) * height)
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append({
            "bbox_xyxy": [x0, y0, x1, y1],
            "bbox_width": x1 - x0,
            "bbox_height": y1 - y0,
        })
    return boxes


def _read_sonar_metadata(xml_path: Path | None) -> dict[str, object]:
    """Read the frame metadata stored in UATD's XML, if available."""

    if xml_path is None or not xml_path.is_file():
        return {}
    root = ET.parse(xml_path).getroot()
    metadata: dict[str, object] = {}
    for field in SONAR_FIELDS:
        element = root.find(f"./sonar/{field}")
        if element is None or element.text is None:
            continue
        value = element.text.strip()
        if not value:
            continue
        try:
            metadata[field] = float(value)
        except ValueError:
            # Frequency is stored as values such as ``1200k``.
            metadata[field] = value
    return metadata


def _edge_padded_crop(image: Image.Image, center_x: float, center_y: float, side: int) -> Image.Image:
    """Crop a square window and extend the nearest border when it leaves the frame."""

    if side < 1:
        raise ValueError(f"crop side must be positive, got {side}")
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = array.shape
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    right = left + side
    bottom = top + side

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        array = np.pad(
            array,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="edge",
        )
        left += pad_left
        top += pad_top
    crop = array[top:top + side, left:left + side]
    if crop.shape != (side, side):
        raise RuntimeError(f"unexpected crop shape {crop.shape}, expected {(side, side)}")
    return Image.fromarray(crop, mode="L")


def _write_contact_sheet(
    items: Iterable[tuple[Path, list[float]]],
    output_path: Path,
    patch_size: int = 256,
    columns: int = 6,
    thumbnail: int = 160,
) -> None:
    """Write a compact grayscale contact sheet for visual QA."""

    selected = list(items)
    if not selected:
        return
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("L", (columns * thumbnail, rows * thumbnail), color=0)
    draw = ImageDraw.Draw(sheet)
    for index, (path, bbox) in enumerate(selected):
        with Image.open(path) as image:
            patch = image.convert("L").resize((thumbnail - 4, thumbnail - 4), Image.Resampling.LANCZOS)
        x = (index % columns) * thumbnail + 2
        y = (index // columns) * thumbnail + 2
        sheet.paste(patch, (x, y))
        draw.rectangle((x - 1, y - 1, x + thumbnail - 4, y + thumbnail - 4), outline=255)
        scale = (thumbnail - 4) / patch_size
        draw.rectangle(
            (
                x + bbox[0] * scale,
                y + bbox[1] * scale,
                x + bbox[2] * scale,
                y + bbox[3] * scale,
            ),
            outline=255,
            width=1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def build_patches(
    uatd_root: Path,
    output_dir: Path,
    split: str = "train",
    patch_size: int = 256,
    context_scale: float = 5.0,
    class_id: int = DEFAULT_CYLINDER_CLASS_ID,
    max_images: int | None = None,
    manifest_path: Path | None = None,
    contact_sheet: Path | None = None,
) -> int:
    """Create patches and return the number of cylinder instances written."""

    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if context_scale < 1.0:
        raise ValueError("context_scale must be at least 1.0 so the whole box is retained")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive")

    images_dir = uatd_root / "processed" / "images" / split
    labels_dir = uatd_root / "processed" / "labels" / split
    xml_dir = uatd_root / "labels_xml"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"label directory does not exist: {labels_dir}")

    label_files = sorted(labels_dir.glob("*.txt"))
    if max_images is not None:
        label_files = label_files[:max_images]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use a new directory to avoid stale patches"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if contact_sheet is not None:
        output_resolved = output_dir.resolve()
        contact_resolved = contact_sheet.resolve()
        if contact_resolved.is_relative_to(output_resolved):
            raise ValueError("--contact-sheet must be outside --out so dataset_tool.py does not ingest the preview")

    generated_previews: list[tuple[Path, list[float]]] = []
    count = 0
    missing_images = 0
    missing_xml = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for label_path in label_files:
            stem = label_path.stem
            image_path = _find_image(images_dir, stem)
            if image_path is None:
                missing_images += 1
                continue
            with Image.open(image_path) as source_image:
                image = source_image.convert("L")
            width, height = image.size
            boxes = _read_boxes(label_path, width, height, class_id)
            if not boxes:
                continue
            xml_path = xml_dir / f"{stem}.xml"
            metadata = _read_sonar_metadata(xml_path)
            if not xml_path.is_file():
                missing_xml += 1
            for object_index, box in enumerate(boxes):
                bbox = box["bbox_xyxy"]
                assert isinstance(bbox, list)
                x0, y0, x1, y1 = (float(value) for value in bbox)
                center_x = (x0 + x1) / 2.0
                center_y = (y0 + y1) / 2.0
                source_side = max(1, int(round(max(x1 - x0, y1 - y0) * context_scale)))
                patch = _edge_padded_crop(image, center_x, center_y, source_side)
                patch = patch.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
                window_left = center_x - source_side / 2.0
                window_top = center_y - source_side / 2.0
                patch_bbox = [
                    (x0 - window_left) / source_side * patch_size,
                    (y0 - window_top) / source_side * patch_size,
                    (x1 - window_left) / source_side * patch_size,
                    (y1 - window_top) / source_side * patch_size,
                ]

                output_name = f"{stem}_cylinder_{object_index:02d}.png"
                output_path = output_dir / output_name
                patch.save(output_path, format="PNG", compress_level=0)
                generated_previews.append((output_path, patch_bbox))
                record = {
                    "patch": output_name,
                    "patch_type": "target_context",
                    "channels": 1,
                    "patch_size": patch_size,
                    "source_split": split,
                    "source_image": str(image_path),
                    "source_width": width,
                    "source_height": height,
                    "class_id": class_id,
                    "object_index": object_index,
                    "bbox_xyxy": [round(value, 3) for value in (x0, y0, x1, y1)],
                    "bbox_patch_xyxy": [round(value, 3) for value in patch_bbox],
                    "crop_center_xy": [round(center_x, 3), round(center_y, 3)],
                    "source_side": source_side,
                    "context_scale": context_scale,
                    "edge_padding": "replicate",
                    "shadow_label_available": False,
                    "sonar_metadata_available": xml_path.is_file(),
                    "sonar": metadata,
                }
                manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    if contact_sheet is not None:
        _write_contact_sheet(generated_previews[:36], contact_sheet, patch_size=patch_size)
    print(json.dumps({
        "split": split,
        "label_files_checked": len(label_files),
        "patches_written": count,
        "missing_images": missing_images,
        "missing_xml": missing_xml,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "contact_sheet": str(contact_sheet) if contact_sheet is not None else None,
    }, ensure_ascii=False))
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uatd-root", type=Path, required=True, help="UATD root containing processed/ and labels_xml/")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for grayscale PNG patches")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--patch-size", type=int, default=256, help="Square output size (default: 256)")
    parser.add_argument("--context-scale", type=float, default=5.0, help="Window side / bbox long side (default: 5.0)")
    parser.add_argument("--class-id", type=int, default=DEFAULT_CYLINDER_CLASS_ID)
    parser.add_argument("--max-images", type=int, default=None, help="Limit checked label files, useful for smoke tests")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest path (default: <out>/manifest.jsonl)")
    parser.add_argument("--contact-sheet", type=Path, default=None, help="Optional contact sheet path for visual QA")
    args = parser.parse_args()
    build_patches(
        uatd_root=args.uatd_root,
        output_dir=args.out,
        split=args.split,
        patch_size=args.patch_size,
        context_scale=args.context_scale,
        class_id=args.class_id,
        max_images=args.max_images,
        manifest_path=args.manifest,
        contact_sheet=args.contact_sheet,
    )


if __name__ == "__main__":
    main()
