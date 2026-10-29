#!/usr/bin/env python3
"""Render ego-centric nuScenes BEV map rasters for EIEA training.

Channels (5): drivable_area, lane, road_divider, ped_crossing, walkway.
Grid: 200x200, 0.5 m/cell, 100 m x 100 m centred at the ego pose and rotated
to ego yaw.  The output ``.npy`` files are referenced by the ``bev`` field of
each row, so ``training/train_eiea_30b.py`` can feed the BEV encoder.

Usage:
    python data/build_eiea_bev.py \
        --split train --data results/eiea_box/train.json \
        --out results/eiea_box

Environment:
    NUSCENES_ROOT  nuScenes v1.0-trainval root (default below)
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

LAYERS = ["drivable_area", "lane", "road_divider", "ped_crossing", "walkway"]
GRID = 200
RES = 0.5
SPAN = GRID * RES
DEFAULT_NS = "/media/nas_users/kaneqian/data/Nuscences"


def load_rows(path: str):
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in open(path) if line.strip()]
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--ns_root", default=os.environ.get("NUSCENES_ROOT", DEFAULT_NS))
    args = ap.parse_args()

    rows = load_rows(args.data)[: args.limit]
    bev_dir = os.path.join(args.out, "bev")
    os.makedirs(bev_dir, exist_ok=True)

    ns = args.ns_root
    sd = json.load(open(os.path.join(ns, "v1.0-trainval/sample_data.json")))
    samples = json.load(open(os.path.join(ns, "v1.0-trainval/sample.json")))
    scenes = json.load(open(os.path.join(ns, "v1.0-trainval/scene.json")))
    logs = json.load(open(os.path.join(ns, "v1.0-trainval/log.json")))
    poses = json.load(open(os.path.join(ns, "v1.0-trainval/ego_pose.json")))

    fn2sd = {}
    for r in sd:
        if "__CAM_" in r["filename"]:
            fn2sd[os.path.basename(r["filename"])] = (r["sample_token"], r["ego_pose_token"])
    st2sc = {s["token"]: s["scene_token"] for s in samples}
    sc2log = {s["token"]: s["log_token"] for s in scenes}
    log2loc = {l["token"]: l["location"] for l in logs}
    pose_by = {p["token"]: p for p in poses}

    from nuscenes.map_expansion.map_api import NuScenesMap
    from pyquaternion import Quaternion

    maps = {}
    written = 0
    for i, r in enumerate(rows):
        images = r.get("images") or ([r["image"]] if r.get("image") else [])
        if not images:
            continue
        # Front camera is images[1] in the 6-view Ego3D order.
        front = os.path.basename(images[1] if len(images) > 1 else images[0])
        sd_pair = fn2sd.get(front)
        if sd_pair is None:
            continue
        st, pt = sd_pair
        loc = log2loc.get(sc2log.get(st2sc.get(st)))
        if loc is None:
            continue
        if loc not in maps:
            maps[loc] = NuScenesMap(dataroot=ns, map_name=loc)

        ep = pose_by[pt]
        q = Quaternion(ep["rotation"])
        ego_x, ego_y, _ = ep["translation"]
        yaw_deg = math.degrees(math.atan2(q.rotation_matrix[1, 0],
                                         q.rotation_matrix[0, 0]))
        masks = maps[loc].get_map_mask(
            [ego_x, ego_y, SPAN, SPAN], yaw_deg, LAYERS, (GRID, GRID))
        bev = np.stack([(m > 0).astype(np.uint8) for m in masks], 0)
        path = os.path.join(bev_dir, f"{args.split}_{i}.npy")
        np.save(path, bev)
        r["bev"] = path
        written += 1
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)

    out_json = os.path.join(args.out, f"{args.split}_bev.json")
    json.dump(rows, open(out_json, "w"), ensure_ascii=False)
    print(f"wrote {written} BEV rasters to {bev_dir}; rows -> {out_json}")


if __name__ == "__main__":
    main()
