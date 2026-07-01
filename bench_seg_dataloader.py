"""Phase-4 CPU sanity checks + DataLoader throughput benchmark (torch_ssl).

  python bench_seg_dataloader.py

Sanity: item shapes/dtypes/ranges, the empty-label tile, geometric-aug effect, and one
end-to-end image->model->loss step (Phase 2 + Phase 4 together). Throughput: tiles/sec across
num_workers, with the pure-Cython online target-gen as the cost driver.
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.seg_dataset import RidgepathSegDataset, seg_worker_init_fn
from losses.ridgepath_loss import ridgepath_loss
from models.encoder_decoder import build_seg_model

MANIFEST = "/mnt/home/ruizhi.yuan/pytorch_seg/cache/manifest.csv"


def sanity():
    print("=== SANITY ===")
    ds = RidgepathSegDataset(MANIFEST, augment=False)
    print(f"dataset size: {len(ds)}")
    img, tgt = ds[0]
    print(f"item0: image {tuple(img.shape)} {img.dtype} | target {tuple(tgt.shape)} {tgt.dtype}")
    assert img.shape == (2, 512, 512) and tgt.shape == (10, 512, 512)
    assert torch.isfinite(img).all() and torch.isfinite(tgt).all()
    print(f"  image range [{img.min():.3f}, {img.max():.3f}] (positive-normalized)")
    sem = tgt[0]
    w = tgt[9]
    fg = sem > 0.5
    row_sum_fg = tgt[1:5].sum(0)[fg].mean().item() if fg.any() else float("nan")
    print(f"  semantic in {{0,1}}: {set(np.unique(sem.numpy()).tolist()) <= {0.0, 1.0}} | "
          f"weight range [{w.min():.3f}, {w.max():.3f}] | row-prob sum on fg ~ {row_sum_fg:.3f}")

    # empty-label tile (precompute reported tile #10 / idx 9 with 0 instances)
    empty_idx = next((i for i, r in enumerate(ds.rows) if "x_34373_y_21502" in r["base"]), None)
    if empty_idx is not None:
        ei, et = ds[empty_idx]
        print(f"  empty tile idx {empty_idx}: target finite={torch.isfinite(et).all().item()} "
              f"| fg pixels={(et[0] > 0.5).sum().item()}")
        assert torch.isfinite(et).all()

    # geometric aug actually transforms the data
    ds_aug = RidgepathSegDataset(MANIFEST, augment=True, seed=123)
    a0, _ = ds_aug[0]
    same = torch.equal(a0, img)
    print(f"  augmented item0 differs from un-augmented: {not same}")

    # end-to-end: image -> model -> loss (both encoders)
    for name in ("resnet18", "tenxnet_small"):
        model = build_seg_model(encoder_name=name, in_chans=2, num_classes=9)
        model.eval()
        with torch.no_grad():
            out = model(img.unsqueeze(0))
        loss, seg, row, col = ridgepath_loss(out, tgt.unsqueeze(0))
        print(f"  [{name}] out {tuple(out.shape)} | loss={loss.item():.4f} "
              f"(seg={seg.item():.4f} row={row.item():.4f} col={col.item():.4f}) "
              f"finite={torch.isfinite(loss).item()}")
        assert out.shape == (1, 9, 512, 512) and torch.isfinite(loss)
    print("  SANITY OK\n")


def throughput(workers_list=(0, 2, 4), batch_size=4, epochs=2):
    print("=== THROUGHPUT (online Cython target-gen is the cost) ===")
    ds = RidgepathSegDataset(MANIFEST, augment=True)
    n = len(ds)
    for nw in workers_list:
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=nw,
            worker_init_fn=seg_worker_init_fn if nw else None,
            persistent_workers=bool(nw), prefetch_factor=(4 if nw else None),
            pin_memory=False, drop_last=False,
        )
        # warmup one pass (spins up workers / loads .so)
        for _ in dl:
            pass
        t0 = time.time()
        seen = 0
        for _ in range(epochs):
            for img, tgt in dl:
                seen += img.shape[0]
        dt = time.time() - t0
        print(f"  num_workers={nw}: {seen} tiles in {dt:.2f}s -> {seen/dt:.1f} tiles/s "
              f"({1000*dt/seen:.1f} ms/tile)")


def main():
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}\n")
    sanity()
    throughput()
    print("\nPHASE-4 CPU CHECKS COMPLETE")


if __name__ == "__main__":
    main()
