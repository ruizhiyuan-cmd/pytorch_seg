"""Phase-5 sanity: imports, model build, DINO restore (+conv1 mapping correctness),
one real-batch forward/backward with shape + grad checks. Fast; no training.

  python sanity_phase5.py
"""
import torch

from data.seg_dataset import RidgepathSegDataset
from losses.ridgepath_loss import ridgepath_loss
from models.encoder_decoder import build_seg_model
from restore import restore_dino_into_resnet18

MANIFEST = "/mnt/home/ruizhi.yuan/pytorch_seg/cache/manifest.csv"
DINO_CKPT = "/mnt/home/ruizhi.yuan/ssl_dino/out_full/checkpoint.pth"
SEG_CHANNELS = ["boundary", "DAPI"]
SSL_CHANNELS = ["DAPI", "boundary", "18S", "avim"]


def test_build():
    print("=== build both encoders ===")
    for name in ("resnet18", "tenxnet_small"):
        m = build_seg_model(encoder_name=name, in_chans=2, num_classes=9)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {n/1e6:.3f}M params")


def test_restore():
    print("\n=== DINO restore into resnet18 seg encoder ===")
    model = build_seg_model(encoder_name="resnet18", in_chans=2, num_classes=9)
    info = restore_dino_into_resnet18(model, DINO_CKPT, SEG_CHANNELS, SSL_CHANNELS, which="student")

    # correctness: conv1 input positions must equal DINO channels by NAME
    ck = torch.load(DINO_CKPT, map_location="cpu", weights_only=False)
    dino_conv1 = ck["student"]["module.backbone.conv1.weight"]  # (64,4,7,7) [DAPI,boundary,18S,avim]
    seg_conv1 = model.encoder.backbone.conv1.weight.detach()    # (64,2,7,7) [boundary,DAPI]
    assert torch.equal(seg_conv1[:, 0], dino_conv1[:, 1]), "seg ch0 (boundary) != DINO idx1"
    assert torch.equal(seg_conv1[:, 1], dino_conv1[:, 0]), "seg ch1 (DAPI) != DINO idx0"
    # a non-conv1 layer must transfer verbatim
    assert torch.equal(model.encoder.backbone.layer1[0].conv1.weight.detach(),
                       ck["student"]["module.backbone.layer1.0.conv1.weight"])
    print("  conv1 name-mapping verified: seg[boundary]=DINO[1], seg[DAPI]=DINO[0]; "
          "layer1.0.conv1 transferred verbatim")
    assert not info["missing"] and not info["unexpected"], \
        f"unexpected key gaps: missing={info['missing']} unexpected={info['unexpected']}"
    print("  0 missing / 0 unexpected backbone keys")


def test_fwd_bwd():
    print("\n=== one real-batch forward/backward ===")
    ds = RidgepathSegDataset(MANIFEST, augment=True, seed=1)
    img0, tgt0 = ds[0]
    img1, tgt1 = ds[1]
    img = torch.stack([img0, img1])   # [2,2,512,512]
    tgt = torch.stack([tgt0, tgt1])   # [2,10,512,512]
    print(f"  batch image {tuple(img.shape)} | target {tuple(tgt.shape)}")
    assert img.shape == (2, 2, 512, 512) and tgt.shape == (2, 10, 512, 512)

    model = build_seg_model(encoder_name="resnet18", in_chans=2, num_classes=9)
    model.train()
    out = model(img)
    print(f"  prediction {tuple(out.shape)}")
    assert out.shape == (2, 9, 512, 512)
    loss, seg, row, col = ridgepath_loss(out, tgt)
    print(f"  loss {loss.item():.4f} (seg {seg.item():.4f} row {row.item():.4f} col {col.item():.4f}) "
          f"finite={torch.isfinite(loss).item()}")
    assert torch.isfinite(loss)
    model.zero_grad()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params without grad: {missing[:5]}"
    print(f"  backward OK, all {sum(1 for _ in model.parameters())} params have grad")


def main():
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    test_build()
    test_restore()
    test_fwd_bwd()
    print("\nPHASE-5 SANITY OK")


if __name__ == "__main__":
    main()
