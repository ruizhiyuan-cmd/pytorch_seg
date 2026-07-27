"""Stage 1 (torch env): run the pytorch_seg checkpoint on its tiles -> export 9-ch logits + 10-ch target
as .npz per tile, for tenxnet's decoder + instance_metric (stage 2, run in the efd215c env).

Usage: export_torch_logits.py <config> <ckpt> <out_dir> [n_tiles] [--split train|val]
       n_tiles=0 (or omitted) exports all tiles of the chosen split.
"""
import argparse, os, yaml, numpy as np, torch
from models.encoder_decoder import build_seg_model
from data.seg_dataset import RidgepathSegDataset

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("n", type=int, nargs="?", default=0, help="#tiles to export (0=all)")
ap.add_argument("--split", choices=["train", "val"], default="val", help="which manifest to export")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config)); ckpt = args.ckpt; out = args.out; N = args.n
manifest = cfg["manifest"] if args.split == "train" else cfg["val_manifest"]
os.makedirs(out, exist_ok=True)
dev = torch.device("cpu")
# model_kwargs match training; force pretrained=False for convnext (weights come from the ckpt below,
# not an HF download at decode time).
mk = dict(cfg.get("model_kwargs") or {})
if cfg["encoder"] in ("convnext_dino_hires", "hires_only"):
    mk["pretrained"] = False
model = build_seg_model(encoder_name=cfg["encoder"], in_chans=cfg["in_chans"],
                        num_classes=cfg["num_classes"], encoder_norm=cfg.get("encoder_norm", "bn"),
                        gn_max_groups=cfg.get("gn_max_groups", 32), **mk).to(dev)
ck = torch.load(ckpt, map_location=dev, weights_only=False)
model.load_state_dict(ck["model"]); model.eval()
print(f"loaded {ckpt} (epoch {ck.get('epoch','?')})")

ds = RidgepathSegDataset(manifest, augment=False, params=cfg["target_params"], seed=cfg["seed"])
ntiles = len(ds) if N <= 0 else min(N, len(ds))
print(f"split={args.split}  manifest={manifest}  ({len(ds)} tiles, exporting {ntiles})")
with torch.no_grad():
    for i in range(ntiles):
        img, tgt = ds[i]                                   # img [2,H,W], tgt [10,H,W]
        logits = model(img.unsqueeze(0).to(dev))[0]        # [9,H,W]
        L = logits.permute(1, 2, 0).cpu().numpy().astype(np.float32)   # [H,W,9]
        T = tgt.permute(1, 2, 0).cpu().numpy().astype(np.float32)      # [H,W,10]
        I = img.permute(1, 2, 0).cpu().numpy().astype(np.float32)      # [H,W,2] (normalized)
        np.savez(os.path.join(out, f"tile_{i}.npz"), logits=L, target=T, image=I)
        print(f"  tile {i}: logits {L.shape} |max|={np.abs(L).max():.2f}  fg_px={(T[...,0]>0.5).sum()}")
print(f"exported {ntiles} {args.split} tiles -> {out}")
