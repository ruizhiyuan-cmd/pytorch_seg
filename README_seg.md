# PyTorch Ridgepath segmentation — Phase 5 (training/eval scaffolding)

Status: **scaffolding only — sanity-checked, NOT trained.** Do not launch full training without
explicit approval. GPU runs go on **gpu1** (bespin1 has no GPU).

## Data / paths in use (large_cell_boundary)
- images: `/mnt/deck/1/ruizhi.yuan/tenxnet_deployed_data/large_cell_boundary_full/large_cell_boundary_v1/images/<subdir>/<base>.tif` — TIFF (512,512,2) uint16, **ch0=boundary, ch1=DAPI**
- labels: `.../large_cell_boundary_v1/labels/<subdir>/<base>.pb` — protobuf instance masks (817 pairs)
- cached inst_ridge: `~/pytorch_seg/cache/inst_ridge/<subdir>/<base>.npy` (all 817 precomputed offline, ref env)
- manifests in `~/pytorch_seg/cache/`:
  - `manifest_full.csv` — all 817 tiles
  - `manifest_train.csv` (694) / `manifest_val.csv` (123) — **deterministic, leakage-safe** split (grouped by source slide; seed 0, val_frac 0.15). **Configs use these.**
  - `manifest_smoke.csv` (16) — debug/smoke ONLY; not for real training
- DINO checkpoint: `~/ssl_dino/out_full/checkpoint.pth` (student backbone, 4-ch conv1, epoch 100)

## Channel mapping (DINO conv1 -> seg conv1), by marker NAME
- SSL 4-ch order (DINO conv1 inputs): `[DAPI, boundary, 18S, avim]`
- seg image order: `[boundary, DAPI]` => slice DINO conv1 `[1, 0]` (boundary<-idx1, DAPI<-idx0)
- Set in configs via `ssl_channels` + `seg_channels`; restore prints the resolved map and loaded/missing/unexpected keys.
- NOTE: the SSL order is user-provided (no ssl_4ch prep script exists in-repo to auto-verify); change `ssl_channels` if it turns out different.

## Step 0 — (re)build the inst_ridge cache (reference/bazel env; run on bespin1, NOT during GPU training)
```bash
BZPY=/mnt/bazelbuild/user/ruizhi.yuan/66d30966b7045ad5bca10aeb4ea3520e/execroot/com_github_10XDev_tenxnet/bazel-out/k8-fastbuild/bin/external/anaconda/bin/python3
RF=/mnt/bazelbuild/user/ruizhi.yuan/66d30966b7045ad5bca10aeb4ea3520e/execroot/com_github_10XDev_tenxnet/bazel-out/k8-fastbuild/bin/bin/train.runfiles/com_github_10XDev_tenxnet
cd ~/pytorch_seg/data
# targets use anno_names=[cell, large_cell] by default (boundary only; excludes 'nucleus').
# To regenerate an existing cache you MUST wipe it first (the launcher skips existing .npy):
rm -rf ~/pytorch_seg/cache/inst_ridge
# parallel (96-core box): 24 shards, then merges -> cache/manifest_full.csv
bash ~/pytorch_seg/cache/run_precompute_sharded.sh
# (or serial: PYTHONPATH="$RF" "$BZPY" precompute_inst_ridge.py --limit 817 --manifest ~/pytorch_seg/cache/manifest_full.csv)
# then split (deterministic, leakage-safe; torch_ssl env):
conda activate torch_ssl
python split_manifest.py --full ~/pytorch_seg/cache/manifest_full.csv --val_frac 0.15 --seed 0
```

## Step 1 — sanity (already verified; CPU ok)
```bash
conda activate torch_ssl
cd ~/pytorch_seg
python sanity_phase5.py                 # build + DINO restore + 1-batch fwd/bwd
python bench_seg_dataloader.py          # data pipeline sanity + throughput
python train_seg.py --config configs/r18_dino.yaml --smoke --save-viz   # ~20-iter loop + viz
```

## Step 2 — the three comparison runs (LAUNCH LATER, on gpu1, after approval)
Fair SSL ablation = **r18_scratch vs r18_dino** (same architecture).
`tenxnet_small_scratch` is the PyTorch-port baseline — do NOT compare it to r18_dino as "SSL benefit"
(architecture differs).

Single GPU:
```bash
conda activate torch_ssl
cd ~/pytorch_seg
python train_seg.py --config configs/tenxnet_small_scratch.yaml   # baseline (faithful tenxnet arch)
python train_seg.py --config configs/r18_scratch.yaml             # ablation control
python train_seg.py --config configs/r18_dino.yaml                # ablation treatment (SSL-init)
```

Optional multi-GPU (DDP, not required):
```bash
torchrun --nproc_per_node=4 train_seg.py --config configs/r18_dino.yaml --ddp
```

Useful flags: `--epochs N --batch-size N --num-workers N --out-dir DIR --resume PATH --save-viz --no-amp --max-iters N`.

## Notes for the real run
- Manifests are ready: configs train on `manifest_train.csv` (694) and eval on `manifest_val.csv` (123). `evaluate()` reports mean val loss (decode/instance metrics are Phase 6, deferred).
- AMP is on by default (CUDA only; auto-off on CPU). LR schedule = cosine + warmup (configurable).
- Eval currently = mean ridgepath loss. Instance AP/AR (COCO `maskApi`) + production `process_direct` decode are Phase 6.
