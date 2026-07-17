"""BN-robust eval of a pytorch_seg checkpoint on its (few) training tiles.

The built-in evaluate() uses model.eval() -> BatchNorm running stats, which are garbage after only a few
updates on ~5 tiles (val_loss explodes to millions). That makes val_fg_pred_bg=1.0 an ARTIFACT, not
necessarily collapse. Here we read fg_pred_bg / dir_acc under TWO modes:
  (A) eval  : model.eval()                        (running-stat BN  = the confounded path)
  (B) bnstat: eval() but BatchNorm forced to train (batch statistics; stochastic-depth stays OFF)
If (B) shows low fg_pred_bg / rising dir_acc while (A) is pinned at 1.0 -> it IS learning; the collapse
signal was a BN-eval artifact.
"""
import sys, yaml, torch, torch.nn as nn
from models.encoder_decoder import build_seg_model
from data.seg_dataset import RidgepathSegDataset
from losses.ridgepath_loss import ridgepath_loss

cfg = yaml.safe_load(open(sys.argv[1])); ckpt = sys.argv[2]
dev = torch.device(sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"device={dev}")
model = build_seg_model(encoder_name=cfg["encoder"], in_chans=cfg["in_chans"],
                        num_classes=cfg["num_classes"], encoder_norm=cfg.get("encoder_norm", "bn")).to(dev)
ck = torch.load(ckpt, map_location=dev, weights_only=False)
model.load_state_dict(ck["model"]); print(f"loaded {ckpt} (epoch {ck.get('epoch','?')})")

ds = RidgepathSegDataset(cfg["val_manifest"], augment=False, params=cfg["target_params"], seed=cfg["seed"])
imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(dev)      # [N,2,H,W]
tgts = torch.stack([ds[i][1] for i in range(len(ds))]).to(dev)      # [N,10,H,W]
print(f"{len(ds)} tiles")


def metrics(logits, tgt):
    total, seg, row, col = ridgepath_loss(logits, tgt)
    fg = tgt[:, 0] > 0.5
    pr, gr = logits[:, 1:5].argmax(1)[fg], tgt[:, 1:5].argmax(1)[fg]
    pc, gc = logits[:, 5:9].argmax(1)[fg], tgt[:, 5:9].argmax(1)[fg]
    fg_pred_bg = ((pr == 3) & (pc == 3)).float().mean().item()      # both row&col bg (== keras metric)
    dir_acc = ((pr == gr) & (pc == gc)).float().mean().item()
    return total.item(), fg_pred_bg, dir_acc


def bn_to_train(m):
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            mod.train()


with torch.no_grad():
    model.eval()
    loss_a, fb_a, da_a = metrics(model(imgs), tgts)
    model.eval(); bn_to_train(model)                                # SD off, BN batch-stats
    loss_b, fb_b, da_b = metrics(model(imgs), tgts)

print(f"  (A) eval   BN: loss={loss_a:12.3f}  fg_pred_bg={fb_a:.3f}  dir_acc={da_a:.3f}")
print(f"  (B) bnstat BN: loss={loss_b:12.3f}  fg_pred_bg={fb_b:.3f}  dir_acc={da_b:.3f}")
print("  ->", "LEARNING (BN-eval artifact hid it)" if fb_b < 0.9 and da_b > 0.05
      else "still collapsed under batch-stat BN too")
