"""READ-ONLY annotation-name inspection of large_cell_boundary .pb labels.

Run under the reference/bazel env (needs tenxnet protobuf + parser):
  RF=<...>/train.runfiles/com_github_10XDev_tenxnet
  PYTHONPATH="$RF" <bazel-py> inspect_anno.py

Does NOT touch the cache. Samples ~15 .pb files (sparse..dense via file size), reports unique
annotation names + counts, and compares instance counts from pb_file_to_mask with
anno_names=None vs anno_names=['cell','large_cell'].
"""
import collections
import os

import numpy as np

from tenxnet.vision.dataloaders.data_format import pb_label_pb2
from tenxnet.vision.dataloaders.data_format.pb_label_util import pb_file_to_mask

LABELS = ("/mnt/deck/1/ruizhi.yuan/tenxnet_deployed_data/large_cell_boundary_full/"
          "large_cell_boundary_v1/labels")
KEEP = ["cell", "large_cell"]


def all_pb():
    out = []
    for sub in sorted(os.listdir(LABELS)):
        d = os.path.join(LABELS, sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".pb"):
                    p = os.path.join(d, f)
                    out.append((sub, f[:-3], p, os.path.getsize(p)))
    return out


def pick_sample(files, n=15):
    s = sorted(files, key=lambda x: x[3])  # by size: small=sparse, large=dense
    idxs = sorted(set([0, 1, 2, len(s)-1, len(s)-2, len(s)-3]
                      + list(np.linspace(0, len(s)-1, n).astype(int))))
    return [s[i] for i in idxs]


def anno_names(pb_path):
    lbl = pb_label_pb2.VisionLabel()
    with open(pb_path, "rb") as f:
        lbl.ParseFromString(f.read())
    names = [a.name for a in lbl.annotations]
    poly_names = [a.name for a in lbl.annotations if a.WhichOneof("annotation") == "polygons"]
    return names, poly_names


def n_inst(mask):
    return int(len(np.unique(mask)) - 1)  # exclude background 0


def main():
    files = all_pb()
    sample = pick_sample(files, 15)
    print(f"total .pb in dataset: {len(files)} | inspecting {len(sample)} sample files\n")

    global_names = collections.Counter()       # all annotation names
    global_poly_names = collections.Counter()  # polygon-type annotation names only
    diffs = []
    print(f"{'subdir':<26}{'base':<48}{'kB':>6}{'#anno':>6}{'all':>6}{'filt':>6}  names")
    for sub, base, p, sz in sample:
        names, poly_names = anno_names(p)
        global_names.update(names)
        global_poly_names.update(poly_names)
        m_all = pb_file_to_mask(p, semantic=False, anno_names=None)
        m_filt = pb_file_to_mask(p, semantic=False, anno_names=KEEP)
        na, nf = n_inst(m_all), n_inst(m_filt)
        if na != nf:
            diffs.append((sub, base, na, nf))
        uniq = sorted(set(names))
        print(f"{sub:<26}{base[:46]:<48}{sz//1024:>6}{len(names):>6}{na:>6}{nf:>6}  {uniq}")

    print("\n=== unique annotation names (ALL) ===")
    for k, v in global_names.most_common():
        print(f"  {k!r}: {v}")
    print("=== unique annotation names (POLYGON-type only, these become mask instances) ===")
    for k, v in global_poly_names.most_common():
        print(f"  {k!r}: {v}")

    extra = sorted(set(global_names) - set(KEEP))
    print("\n=== SUMMARY ===")
    print(f"  names beyond {KEEP}: {extra if extra else 'NONE'}")
    print(f"  sample files where mask_all != mask_filtered: {len(diffs)}/{len(sample)}")
    for sub, base, na, nf in diffs[:8]:
        print(f"    {sub}/{base}: all={na} filt={nf} (delta={na-nf})")
    if extra or diffs:
        print("\n  RECOMMENDATION: regenerate cache with anno_names=['cell','large_cell'] "
              "(current cache used anno_names=None and likely includes extra objects).")
    else:
        print("\n  RECOMMENDATION: only cell/large_cell present and counts identical -> "
              "current cache (anno_names=None) is probably OK.")


if __name__ == "__main__":
    main()
