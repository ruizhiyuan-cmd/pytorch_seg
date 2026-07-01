#!/usr/bin/env python
"""Standalone segmentation visualization -- decoupled from training.

Give it a seg config (for the model arch + data + run dir) and a checkpoint, and it runs inference
on N tiles and writes per-tile panels + raw arrays. No training, no DDP.

Panels per tile:  boundary | DAPI | pred semantic prob | target semantic | agreement (TP/FP/FN)

The model emits 9 logits with layout ``[semantic, row0:4, col0:4]`` (see losses/ridgepath_loss.py):
semantic -> sigmoid. Targets are ``[semantic, row0:4, col0:4, weight]``; we show the semantic target
for comparison. The raw row/col direction argmaps are saved to the .npz but not plotted -- their bare
argmax is not visually meaningful without the centerpath decode.

This shows the **raw network outputs**, not decoded instances. Turning the row/col/semantic maps into
labeled cells needs the centerpath decode (ridgepath_glue.cp_construct_instance) -- not wired here yet
(add an ``--instances`` path later for colored instance overlays).

Usage:
  python visualize_seg.py --config configs/tenxnet_small_dino_boundary2ch.yaml          # best.pth + val split
  python visualize_seg.py --config <cfg> --checkpoint runs/<run>/best.pth --manifest <csv> --n 8 --out viz_out
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.encoder_decoder import build_seg_model  # noqa: E402
from data.seg_dataset import RidgepathSegDataset  # noqa: E402
from train_seg import load_config  # noqa: E402 (DEFAULTS + YAML merge; no side effects on import)


def _stretch(x):
    """Per-channel 1-99 percentile contrast stretch -> [0, 1] for display."""
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser("standalone segmentation visualization")
    ap.add_argument("--config", required=True, help="seg YAML (model arch + data + out_dir)")
    ap.add_argument("--checkpoint", default=None,
                    help="path to .pth (default: <out_dir>/best.pth, else <out_dir>/checkpoint.pth)")
    ap.add_argument("--manifest", default=None, help="data manifest CSV (default: cfg val_manifest)")
    ap.add_argument("--n", type=int, default=6, help="number of tiles to visualize")
    ap.add_argument("--out", default=None, help="output dir (default: <out_dir>/viz)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for tile sampling")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="foreground prob threshold for the agreement map (the semantic head can be "
                         "under-confident, so a lower value may be more representative)")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit tile indices (overrides random sampling)")
    args = ap.parse_args()

    cfg = load_config(args.config, {})
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = args.checkpoint
    if ckpt is None:
        best = os.path.join(cfg["out_dir"], "best.pth")
        ckpt = best if os.path.exists(best) else os.path.join(cfg["out_dir"], "checkpoint.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    model = build_seg_model(encoder_name=cfg["encoder"], in_chans=cfg["in_chans"],
                            num_classes=cfg["num_classes"],
                            encoder_norm=cfg.get("encoder_norm", "bn"),
                            gn_max_groups=cfg.get("gn_max_groups", 32)).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])  # save_ckpt stores the raw (unwrapped) module state_dict
    model.eval()
    print(f"[viz] {ckpt} (epoch {ck.get('epoch')}) | encoder={cfg['encoder']} init={cfg.get('init')} "
          f"| device={device}")

    manifest = args.manifest or cfg["val_manifest"]
    ds = RidgepathSegDataset(manifest, augment=False, params=cfg["target_params"])
    out = args.out or os.path.join(cfg["out_dir"], "viz")
    os.makedirs(out, exist_ok=True)

    if args.indices is not None:
        idx = [i for i in args.indices if 0 <= i < len(ds)]
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist()

    for i in idx:
        img, tgt = ds[int(i)]                              # img [2,H,W], tgt [10,H,W]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))[0].cpu()  # [9,H,W]
        sem = torch.sigmoid(logits[0]).numpy()              # semantic prob
        row_pred = logits[1:5].argmax(0).numpy()            # 0..3 row dir (raw -> npz only; flat argmax)
        col_pred = logits[5:9].argmax(0).numpy()            # 0..3 col dir (raw -> npz only)
        tgt_sem = tgt[0].numpy()
        bd, dapi = _stretch(img[0].numpy()), _stretch(img[1].numpy())

        np.savez(os.path.join(out, f"viz_{int(i):05d}.npz"),
                 boundary=bd, dapi=dapi, pred_semantic=sem, target_semantic=tgt_sem,
                 pred_row=row_pred, pred_col=col_pred)

        # foreground agreement map (pred>0.5 vs target>0.5): green=correct (TP), red=false-positive,
        # blue=missed (FN), black=correct background -- localizes where the semantic head errs.
        pf, gf = sem > args.thresh, tgt_sem > 0.5
        agree = np.zeros((*pf.shape, 3), dtype=np.float32)
        agree[pf & gf] = (0.0, 1.0, 0.0)
        agree[pf & ~gf] = (1.0, 0.0, 0.0)
        agree[~pf & gf] = (0.0, 0.3, 1.0)

        fig, ax = plt.subplots(1, 5, figsize=(15, 3.1))
        ax[0].imshow(bd, cmap="gray");                       ax[0].set_title("boundary", fontsize=9)
        ax[1].imshow(dapi, cmap="gray");                     ax[1].set_title("DAPI", fontsize=9)
        ax[2].imshow(sem, cmap="magma", vmin=0, vmax=1);     ax[2].set_title("pred semantic prob", fontsize=9)
        ax[3].imshow(tgt_sem, cmap="magma", vmin=0, vmax=1); ax[3].set_title("target semantic", fontsize=9)
        ax[4].imshow(agree);  ax[4].set_title(f"agree@{args.thresh:g}: TP=g FP=r FN=b", fontsize=9)
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"viz_{int(i):05d}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

    print(f"[viz] wrote {len(idx)} tiles (PNG + npz) to {out}")


if __name__ == "__main__":
    main()
