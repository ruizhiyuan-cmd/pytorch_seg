"""Phase-2 verification: shapes, encoder specs, gradient flow, param counts, overfit sanity.

Run in torch_ssl:  python test_model.py
"""
import sys

import torch

from losses.ridgepath_loss import ridgepath_loss
from models.encoder_decoder import build_seg_model


def make_target(b, h, w, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.zeros(b, 10, h, w)
    t[:, 0] = (torch.rand(b, h, w, generator=g) > 0.4).float()
    t[:, 1:5] = torch.softmax(torch.randn(b, 4, h, w, generator=g), dim=1)
    t[:, 5:9] = torch.softmax(torch.randn(b, 4, h, w, generator=g), dim=1)
    t[:, 9] = torch.rand(b, h, w, generator=g) * 0.5 + 0.5
    return t.to(device)


def check_encoder(name, sizes=((2, 128, 128), (2, 96, 160))):
    print(f"\n========== encoder = {name} ==========")
    model = build_seg_model(encoder_name=name, in_chans=2, num_classes=9)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output_specs = {model.encoder.output_specs} | params = {n_params/1e6:.3f}M")

    for (c, h, w) in sizes:
        x = torch.randn(2, c, h, w)
        feats = model.encoder.forward_features(x)
        spec = {k: tuple(v.shape) for k, v in feats.items()}
        out = model(x)
        print(f"  in [2,{c},{h},{w}] -> features {spec}")
        print(f"                      -> out {tuple(out.shape)}")
        assert out.shape == (2, 9, h, w), f"bad output shape {tuple(out.shape)}"

    # gradient flow: every parameter must receive a grad.
    x = torch.randn(2, 2, 128, 128)
    target = make_target(2, 128, 128, x.device)
    out = model(x)
    total, seg, row, col = ridgepath_loss(out, target)
    model.zero_grad()
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    print(f"  loss total={total.item():.4f} (seg={seg.item():.4f} row={row.item():.4f} "
          f"col={col.item():.4f}) | params w/o grad: {len(missing)}")
    assert not missing, f"params without grad: {missing[:5]}..."
    return model


def overfit_sanity(name="resnet18", steps=60):
    print(f"\n========== overfit one batch ({name}) ==========")
    torch.manual_seed(0)
    model = build_seg_model(encoder_name=name, in_chans=2, num_classes=9)
    x = torch.randn(2, 2, 96, 96)
    target = make_target(2, 96, 96, x.device, seed=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    first = last = None
    for i in range(steps):
        out = model(x)
        loss, *_ = ridgepath_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
        if i % 15 == 0 or i == steps - 1:
            print(f"  step {i:3d}: loss = {loss.item():.4f}")
    print(f"  {first:.4f} -> {last:.4f}  ({'DECREASED' if last < first else 'NO DECREASE'})")
    assert last < first, "loss did not decrease on a single-batch overfit"


def main():
    print(f"torch {torch.__version__}")
    for name in ("resnet18", "tenxnet_small"):
        check_encoder(name)
    overfit_sanity("resnet18")
    overfit_sanity("tenxnet_small")
    print("\nALL PHASE-2 CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
