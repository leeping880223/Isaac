"""
將 Isaac Sim Replicator 輸出的 semantic_segmentation_*.png + semantic_segmentation_labels_*.json
轉換成 YOLO segmentation 格式的 images/labels 資料集（多邊形座標），並自動切分 train/val。

用法:
    python convert_to_yolo.py --config ../configs/dataset_config.yaml
"""

import argparse
import ast
import glob
import json
import os
import random
import shutil
import sys
import time

import cv2
import numpy as np
import yaml

# 路徑改用相對於專案根目錄，避免換機器跑不動（跟 capture_dataset.py 的 DATA_ROOT 慣例一致）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("DATA_ROOT", PROJECT_ROOT)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        raw = f.read()
    # 簡單支援 ${ENV_VAR} 展開，避免寫死絕對路徑
    raw = os.path.expandvars(raw)
    return yaml.safe_load(raw)


def parse_color_key(key: str) -> tuple:
    """ '(93, 220, 11, 255)' -> (93, 220, 11, 255)，對應 labels json 裡的 RGBA key"""
    return tuple(ast.literal_eval(key))


def extract_polygons(mask_path: str, labels_path: str, class_map: dict, min_area: float = 20.0) -> list:
    """讀取語意分割遮罩，依顏色找出每個 class 的輪廓，回傳 YOLO 多邊形標註（已正規化）"""
    with open(labels_path, "r") as f:
        labels = json.load(f)

    mask_bgra = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_bgra is None or mask_bgra.shape[-1] != 4:
        return []
    mask_rgba = cv2.cvtColor(mask_bgra, cv2.COLOR_BGRA2RGBA)
    img_h, img_w = mask_rgba.shape[:2]

    lines = []
    for color_key, info in labels.items():
        class_name = info.get("class", "")
        if class_name not in class_map:
            continue
        class_id = class_map[class_name]
        color = np.array(parse_color_key(color_key), dtype=mask_rgba.dtype)

        color_mask = np.all(mask_rgba == color, axis=-1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < min_area or len(contour) < 3:
                continue
            points = contour.reshape(-1, 2).astype(float)
            points[:, 0] /= img_w
            points[:, 1] /= img_h
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
            lines.append(f"{class_id} {coords}")

    return lines


def convert(cfg: dict):
    raw_dir = cfg["paths"]["raw_data_dir"]
    out_dir = cfg["paths"]["yolo_data_dir"]
    class_map = cfg["class_map"]
    val_ratio = cfg["split"]["val_ratio"]
    seed = cfg["split"]["seed"]

    mask_files = sorted(glob.glob(os.path.join(raw_dir, "**", "semantic_segmentation_[0-9]*.png"), recursive=True))
    if not mask_files:
        print(f"[WARN] 在 {raw_dir} 找不到任何 semantic_segmentation_*.png，請確認 raw_data_dir 設定")
        return

    # 先收集所有有效的 frame，再做 train/val 切分，避免邊處理邊分導致比例不準
    valid_frames = []
    for mask_file in mask_files:
        frame_dir = os.path.dirname(mask_file)
        frame_id = os.path.splitext(os.path.basename(mask_file))[0].split("_")[-1]
        labels_file = os.path.join(frame_dir, f"semantic_segmentation_labels_{frame_id}.json")
        img_src = os.path.join(frame_dir, f"rgb_{frame_id}.png")
        if os.path.exists(labels_file) and os.path.exists(img_src):
            # 用「class名稱_視角_frame_id」當輸出檔名，避免不同物件/視角的同編號互相覆蓋
            # 路徑結構：.../normal/<class>/<cam>/semantic_segmentation_*.png
            cam_name = os.path.basename(frame_dir)
            class_name_dir = os.path.basename(os.path.dirname(frame_dir))
            unique_id = f"{class_name_dir}_{cam_name}_{frame_id}".replace(" ", "_")
            valid_frames.append((unique_id, mask_file, labels_file, img_src))

    random.seed(seed)
    random.shuffle(valid_frames)
    val_count = int(len(valid_frames) * val_ratio)
    val_ids = set(uid for uid, *_ in valid_frames[:val_count])

    # 每次重跑前先清空舊的 images/labels，避免不同批次的來源資料交錯留下殘留檔案
    for split_name in ("train", "val"):
        shutil.rmtree(os.path.join(out_dir, "images", split_name), ignore_errors=True)
        shutil.rmtree(os.path.join(out_dir, "labels", split_name), ignore_errors=True)
        os.makedirs(os.path.join(out_dir, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split_name), exist_ok=True)

    move_count = 0
    write_count = 0
    total = len(valid_frames)
    start_time = time.time()

    for idx, (unique_id, mask_file, labels_file, img_src) in enumerate(valid_frames, start=1):
        split_name = "val" if unique_id in val_ids else "train"

        img_dst = os.path.join(out_dir, "images", split_name, f"{unique_id}.png")
        shutil.copy(img_src, img_dst)

        txt_dst = os.path.join(out_dir, "labels", split_name, f"{unique_id}.txt")
        lines = extract_polygons(mask_file, labels_file, class_map)

        with open(txt_dst, "w") as f_out:
            f_out.write("\n".join(lines) + ("\n" if lines else ""))

        if lines:
            write_count += 1
        move_count += 1

        elapsed = time.time() - start_time
        eta = elapsed / idx * (total - idx)
        sys.stdout.write(
            f"\r[進度] {idx}/{total} ({idx / total:.1%})  "
            f"已耗時 {elapsed:.0f}s，預估剩餘 {eta:.0f}s"
        )
        sys.stdout.flush()

    print()
    total_time = time.time() - start_time
    print("[DONE] 轉換完成！")
    print(f"-> 共處理了 {move_count} 幀影像（train: {len(valid_frames) - val_count}, val: {val_count}）")
    print(f"-> 成功寫入 {write_count} 張標註")
    print(f"-> 總耗時 {total_time:.1f}s")

    write_data_yaml(out_dir, class_map)


def write_data_yaml(out_dir: str, class_map: dict) -> None:
    """每次轉換都重新產生 data.yaml，避免手動維護導致路徑寫死、換機器/clone 後失效"""
    names = {class_id: name for name, class_id in class_map.items()}
    data_yaml = {
        "path": os.path.abspath(out_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": {i: names[i] for i in sorted(names)},
    }
    with open(os.path.join(out_dir, "data.yaml"), "w") as f_out:
        yaml.safe_dump(data_yaml, f_out, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "configs", "dataset_config.yaml"),
        help="dataset_config.yaml 的路徑",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    convert(cfg)


if __name__ == "__main__":
    main()
