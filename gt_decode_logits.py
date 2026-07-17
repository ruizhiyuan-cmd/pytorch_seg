#!/usr/bin/env python
"""Export PEAKED GT 9-channel logits (from the ground-truth ridgepath target) for a decode sanity check.

Purpose: verify the turing instance-decode pipeline INDEPENDENTLY of the trained model. Instead of a
model forward, we take the exact GT target the model is trained against
(``RidgepathSegDataset`` with augment=False -> ``[10,H,W]`` = [semantic, row0:4, col0:4, weight]),
convert the direction distributions to PEAKED logits (argmax -> one-hot x scale; post.pb only uses the
softmax-argmax, so this is exactly what a "perfect" model would feed it), and write them in the SAME
format ``export_logits.py`` uses. Running turing's ``eval_external_logits.py`` on this manifest should
reproduce the GT instances at ~F1 1.0 if the decode + target-gen are correct.

  ch0 (semantic): (2*(sem>0.5) - 1) * scale        (foreground positive, background negative)
  ch1:5 (row):    (onehot(argmax row0:4)*2 - 1) * scale
  ch5:9 (col):    (onehot(argmax col0:4)*2 - 1) * scale

Outputs mirror export_logits.py: <out>/logits/<key>.npy (H,W,9 f32), <out>/gt/<key>.npy (H,W u32
instance ids = inst_ridge ch0), <out>/export_manifest.csv (key,npy_path,gt_npy_path,label_pb_path,H,W).

Usage:
  python gt_decode_logits.py --manifest cache/manifest_train_labeled.csv --out runs/gt_decode --n 20
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.seg_dataset import RidgepathSegDataset, DEFAULT_PARAMS  # noqa: E402


def _label_pb_path(image_path: str) -> str:
    """images/ -> labels/, .tif -> .pb (same pairing as export_logits.py; used only for overlays)."""
    return image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".pb"


def peak(onehot_argmax, n_bins, scale):
    """argmax index map [H,W] -> peaked logits [n_bins,H,W]: +scale at argmax, -scale elsewhere."""
    oh = np.eye(n_bins, dtype=np.float32)[onehot_argmax]      # [H,W,n_bins]
    return ((oh * 2.0 - 1.0) * scale).transpose(2, 0, 1)      # [n_bins,H,W]


def main():
    ap = argparse.ArgumentParser("export peaked GT 9-ch logits for decode verification")
    ap.add_argument("--manifest", default="cache/manifest_train_labeled.csv")
    ap.add_argument("--out", default="runs/gt_decode")
    ap.add_argument("--n", type=int, default=None, help="limit to first N tiles")
    ap.add_argument("--scale", type=float, default=20.0, help="peaked-logit magnitude (softmax-decisive)")
    ap.add_argument("--rot-k", type=int, default=0, help="np.rot90 k applied to inst_ridge (and GT mask) "
                    "before target-gen -- to test rotation-augmentation direction equivariance")
    args = ap.parse_args()

    ds = RidgepathSegDataset(args.manifest, augment=False, params=dict(DEFAULT_PARAMS))
    out = os.path.abspath(args.out)
    logits_dir, gt_dir = os.path.join(out, "logits"), os.path.join(out, "gt")
    os.makedirs(logits_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    n = len(ds) if args.n is None else min(args.n, len(ds))
    print(f"[gt-decode] {n} tiles | scale={args.scale} | out={out}", flush=True)

    rows = []
    for i in range(n):
        row = ds.rows[i]
        key = f"{row['subdir']}__{row['base']}"
        # Compute the target directly from inst_ridge (augment=False path) -- do NOT go through ds[i],
        # which also reads the .tif image from /mnt/deck (unneeded here, and the mount can stall).
        ir = np.load(row["inst_ridge_path"])             # (H,W,3) uint16; ch0 = instance ids
        if args.rot_k:                                   # rotate geometry BEFORE target-gen (aug parity)
            ir = np.ascontiguousarray(np.rot90(ir, args.rot_k, axes=(0, 1)))
        tgt = np.asarray(ds._target(ir.astype(np.uint16)))  # (H,W,10) float64
        sem = tgt[..., 0]
        row_arg = tgt[..., 1:5].argmax(-1)               # [H,W] in {0..3}
        col_arg = tgt[..., 5:9].argmax(-1)
        logits = np.empty((9, *sem.shape), dtype=np.float32)
        logits[0] = (2.0 * (sem > 0.5).astype(np.float32) - 1.0) * args.scale
        logits[1:5] = peak(row_arg, 4, args.scale)
        logits[5:9] = peak(col_arg, 4, args.scale)
        nhwc = logits.transpose(1, 2, 0).copy()          # [H,W,9]
        npy_path = os.path.join(logits_dir, f"{key}.npy")
        np.save(npy_path, nhwc)
        gt = ir[..., 0].astype(np.uint32)                # instance ids (already loaded)
        gt_path = os.path.join(gt_dir, f"{key}.npy")
        np.save(gt_path, gt)
        H, W = sem.shape
        rows.append(dict(key=key, npy_path=npy_path, gt_npy_path=gt_path,
                         label_pb_path=_label_pb_path(row["image_path"]), H=H, W=W))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    man = os.path.join(out, "export_manifest.csv")
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "npy_path", "gt_npy_path", "label_pb_path", "H", "W"])
        w.writeheader()
        w.writerows(rows)
    print(f"[gt-decode] wrote {len(rows)} tiles -> {logits_dir}\n[gt-decode] manifest: {man}", flush=True)


if __name__ == "__main__":
    main()
