"""
TOD3Cap dataset loader.

Expected layout (configurable via environment variables):
  $TOD3CAP_ROOT/
    annotations.json
    splits.json (optional, or splits inside annotations)
    pointclouds/
      <scene_id>.npy or <scene_id>.npz
"""
import os
import json
import random
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import clip.simple_tokenizer

import utils.pc_util as pc_util
from utils.pc_util import scale_points, shift_scale_points

IGNORE_LABEL = -100
BASE = "."
TOD3CAP_ROOT = os.environ.get("TOD3CAP_ROOT", os.path.join(BASE, "data", "tod3cap"))
ANNOTATION_PATH = os.environ.get("TOD3CAP_ANN", os.path.join(TOD3CAP_ROOT, "annotations.json"))
SPLIT_PATH = os.environ.get("TOD3CAP_SPLIT", os.path.join(TOD3CAP_ROOT, "splits.json"))

CAPTION_EVAL_MODE = "object"


def _load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _get_scene_list(annotations, split_set):
    splits = None
    if os.path.isfile(SPLIT_PATH):
        splits = _load_json(SPLIT_PATH)
    elif isinstance(annotations, dict) and "splits" in annotations:
        splits = annotations["splits"]
    if splits is None or split_set not in splits:
        raise ValueError(
            "TOD3Cap splits not found. Provide splits.json or 'splits' in annotations."
        )
    return splits[split_set]


def _extract_scenes(annotations):
    if isinstance(annotations, list):
        return annotations
    if isinstance(annotations, dict) and "scenes" in annotations:
        return annotations["scenes"]
    raise ValueError("Unsupported annotations format for TOD3Cap.")


def _extract_categories(annotations, scenes):
    if isinstance(annotations, dict) and "categories" in annotations:
        return [c["name"] if isinstance(c, dict) else c for c in annotations["categories"]]
    categories = set()
    for scene in scenes:
        for obj in scene.get("objects", []):
            cat = obj.get("category", obj.get("name", "unknown"))
            categories.add(cat)
    return sorted(list(categories))


def _corners_from_center_size(center, size):
    cx, cy, cz = center
    dx, dy, dz = size
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    corners = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )
    return corners


def _parse_box(obj):
    if "corners" in obj:
        corners = np.array(obj["corners"], dtype=np.float32)
        min_corner = corners.min(axis=0)
        max_corner = corners.max(axis=0)
        center = (min_corner + max_corner) / 2.0
        size = max_corner - min_corner
        return center, size, corners

    bbox = obj.get("bbox", obj.get("box"))
    if bbox is None:
        raise ValueError("Object missing bbox/box field.")

    if isinstance(bbox, dict):
        if "center" in bbox and "size" in bbox:
            center = np.array(bbox["center"], dtype=np.float32)
            size = np.array(bbox["size"], dtype=np.float32)
            corners = _corners_from_center_size(center, size)
            return center, size, corners
        if "min" in bbox and "max" in bbox:
            min_corner = np.array(bbox["min"], dtype=np.float32)
            max_corner = np.array(bbox["max"], dtype=np.float32)
            center = (min_corner + max_corner) / 2.0
            size = max_corner - min_corner
            corners = _corners_from_center_size(center, size)
            return center, size, corners

    bbox = np.array(bbox, dtype=np.float32).tolist()
    if len(bbox) >= 6:
        center = np.array(bbox[:3], dtype=np.float32)
        size = np.array(bbox[3:6], dtype=np.float32)
        corners = _corners_from_center_size(center, size)
        return center, size, corners

    raise ValueError("Unsupported bbox format for TOD3Cap.")


class DatasetConfig(object):
    def __init__(self):
        annotations = _load_json(ANNOTATION_PATH)
        scenes = _extract_scenes(annotations)
        categories = _extract_categories(annotations, scenes)

        self.type2class = {name: i for i, name in enumerate(categories)}
        self.class2type = {i: name for name, i in self.type2class.items()}
        self.num_semcls = len(self.type2class)
        self.num_angle_bin = 1
        self.max_num_obj = 256

    def angle2class(self, angle):
        raise ValueError("TOD3Cap uses axis-aligned boxes by default.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        # axis-aligned boxes; ignore angle
        hx = box_size[:, 0] / 2.0
        hy = box_size[:, 1] / 2.0
        hz = box_size[:, 2] / 2.0
        cx, cy, cz = box_center_unnorm[:, 0], box_center_unnorm[:, 1], box_center_unnorm[:, 2]
        corners = torch.stack(
            [
                torch.stack([cx - hx, cy - hy, cz - hz], dim=-1),
                torch.stack([cx + hx, cy - hy, cz - hz], dim=-1),
                torch.stack([cx + hx, cy + hy, cz - hz], dim=-1),
                torch.stack([cx - hx, cy + hy, cz - hz], dim=-1),
                torch.stack([cx - hx, cy - hy, cz + hz], dim=-1),
                torch.stack([cx + hx, cy - hy, cz + hz], dim=-1),
                torch.stack([cx + hx, cy + hy, cz + hz], dim=-1),
                torch.stack([cx - hx, cy + hy, cz + hz], dim=-1),
            ],
            dim=1,
        )
        return corners

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        corners = []
        for i in range(box_center_unnorm.shape[0]):
            corners.append(_corners_from_center_size(box_center_unnorm[i], box_size[i]))
        return np.stack(corners, axis=0)

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        dataset_config,
        split_set="train",
        num_points=40000,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        use_height=False,
        augment=False,
    ):
        self.dataset_config = dataset_config
        self.args = args
        self.max_des_len = args.max_des_len
        self.num_points = num_points
        self.use_height = use_height
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.augment = augment
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

        if args.vocabulary == "clib":
            self.tokenizer = clip.simple_tokenizer.SimpleTokenizer()
            self.clip_start_id = self.tokenizer.encoder.get("<|startoftext|>", 49406)
            self.clip_end_id = self.tokenizer.encoder.get("<|endoftext|>", 49407)
        else:
            raise NotImplementedError("TOD3Cap currently supports clib tokenizer only.")

        annotations = _load_json(ANNOTATION_PATH)
        scenes = _extract_scenes(annotations)
        scene_ids = _get_scene_list(annotations, split_set)
        self.scenes = [s for s in scenes if s.get("scene_id") in scene_ids]
        self.scene_id_to_idx = {s.get("scene_id"): i for i, s in enumerate(self.scenes)}

        if len(self.scenes) == 0:
            raise ValueError(f"No scenes found for split {split_set}.")

    def __len__(self):
        return len(self.scenes)

    def _load_point_cloud(self, path):
        full_path = path if os.path.isabs(path) else os.path.join(TOD3CAP_ROOT, path)
        if full_path.endswith(".npz"):
            data = np.load(full_path)
            for key in ["points", "point_cloud", "pc"]:
                if key in data:
                    return data[key]
            raise ValueError(f"npz file missing point cloud array: {full_path}")
        if full_path.endswith(".npy"):
            return np.load(full_path)
        raise ValueError(f"Unsupported point cloud format: {full_path}")

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        scene_id = scene.get("scene_id")
        pc_path = scene.get("pointcloud", scene.get("pointcloud_path"))
        if pc_path is None:
            raise ValueError(f"Scene {scene_id} missing pointcloud path.")

        point_cloud_raw = self._load_point_cloud(pc_path)
        point_cloud_raw = point_cloud_raw.astype(np.float32)
        point_cloud_dims_min = point_cloud_raw[:, :3].min(axis=0)
        point_cloud_dims_max = point_cloud_raw[:, :3].max(axis=0)

        if point_cloud_raw.shape[0] >= self.num_points:
            point_cloud = pc_util.random_sampling(point_cloud_raw, self.num_points)
        else:
            pad = np.zeros((self.num_points - point_cloud_raw.shape[0], point_cloud_raw.shape[1]), dtype=point_cloud_raw.dtype)
            point_cloud = np.concatenate([point_cloud_raw, pad], axis=0)

        objects = scene.get("objects", [])
        max_num_obj = self.dataset_config.max_num_obj

        box_centers = np.zeros((max_num_obj, 3), dtype=np.float32)
        box_sizes = np.zeros((max_num_obj, 3), dtype=np.float32)
        box_corners = np.zeros((max_num_obj, 8, 3), dtype=np.float32)
        box_present = np.zeros((max_num_obj,), dtype=np.float32)
        box_semcls = np.zeros((max_num_obj,), dtype=np.int64)
        box_object_ids = np.full((max_num_obj,), fill_value=-1, dtype=np.int64)
        box_captions = []

        raw_angles = np.zeros((max_num_obj,), dtype=np.float32)
        angle_classes = np.zeros((max_num_obj,), dtype=np.int64)
        angle_residuals = np.zeros((max_num_obj,), dtype=np.float32)

        for obj_idx, obj in enumerate(objects[:max_num_obj]):
            center, size, corners = _parse_box(obj)
            category = obj.get("category", obj.get("name", "unknown"))
            captions = obj.get("captions", obj.get("caption", ""))
            if isinstance(captions, str):
                captions = [captions]

            box_centers[obj_idx] = center
            box_sizes[obj_idx] = size
            box_corners[obj_idx] = corners
            box_present[obj_idx] = 1.0
            box_semcls[obj_idx] = self.dataset_config.type2class.get(category, 0)
            box_object_ids[obj_idx] = obj.get("object_id", obj_idx)
            box_captions.append(captions)

        if len(box_captions) < max_num_obj:
            box_captions.extend([[""]] * (max_num_obj - len(box_captions)))

        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        ).squeeze(0)
        box_centers_normalized = box_centers_normalized * box_present[..., None]

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            box_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        ).squeeze(0)

        reference_tokens = np.zeros((max_num_obj, self.max_des_len), dtype=np.int64)
        reference_masks = np.zeros((max_num_obj, self.max_des_len), dtype=np.float32)
        for i, captions in enumerate(box_captions[:max_num_obj]):
            caption = captions[0] if captions else ""
            token_ids = [self.clip_start_id] + self.tokenizer.encode(caption) + [self.clip_end_id]
            token_np = np.array(token_ids, dtype=np.int64)
            if token_np.shape[0] < self.max_des_len:
                pad_length = self.max_des_len - token_np.shape[0]
                token_np = np.concatenate([token_np, np.zeros((pad_length,), dtype=token_np.dtype)])
            elif token_np.shape[0] > self.max_des_len:
                token_np = token_np[:self.max_des_len]
            mask_np = (token_np != 0).astype(np.float32)
            reference_tokens[i] = token_np
            reference_masks[i] = mask_np

        ret_dict = {
            "point_clouds": point_cloud.astype(np.float32),
            "gt_box_corners": box_corners.astype(np.float32),
            "gt_box_centers": box_centers.astype(np.float32),
            "gt_box_centers_normalized": box_centers_normalized.astype(np.float32),
            "gt_angle_class_label": angle_classes.astype(np.int64),
            "gt_angle_residual_label": angle_residuals.astype(np.float32),
            "gt_box_sem_cls_label": box_semcls.astype(np.int64),
            "gt_box_present": box_present.astype(np.float32),
            "scan_idx": np.array(idx).astype(np.int64),
            "gt_box_sizes": box_sizes.astype(np.float32),
            "gt_box_sizes_normalized": box_sizes_normalized.astype(np.float32),
            "gt_box_angles": raw_angles.astype(np.float32),
            "point_cloud_dims_min": point_cloud_dims_min.astype(np.float32),
            "point_cloud_dims_max": point_cloud_dims_max.astype(np.float32),
            "gt_box_object_ids": box_object_ids.astype(np.int64),
            "reference_tokens": reference_tokens.astype(np.int64),
            "reference_masks": reference_masks.astype(np.float32),
            "scene_caption": box_captions[0][0] if len(box_captions) > 0 and len(box_captions[0]) > 0 else "",
            "scene_id": scene_id,
            "gt_box_captions": box_captions,
        }

        # placeholder votes (no point-level instance labels available)
        ret_dict["vote_label"] = np.zeros((self.num_points, 9), dtype=np.float32)
        ret_dict["vote_label_mask"] = np.zeros((self.num_points,), dtype=np.int64)

        return ret_dict
