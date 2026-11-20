"""Derive a label-scarce split from data/UIEB without re-downloading.

The first N labeled pairs stay labeled; the remaining labeled pairs lose their GT and join the
unlabeled pool (their GT goes only to unlabeled_gt_analysis/, used by the bank-quality diagnostic).
Files use relative symlinks where available, with hard links or copies as a fallback.
Symlinks extracted as plain text by ZIP tools are repaired on rerun.

    python data/make_lowlabel_split.py --src data/UIEB --dst data/UIEB/L50 --n_labeled 50
"""
import argparse
import os
import shutil


def relative_target(src, directory):
    try:
        return os.path.relpath(src, directory)
    except ValueError:  # Windows paths on different drives cannot be relative.
        return None


def remove_archived_symlink(dst, target):
    """A flattened symlink contains exactly its original relative path."""
    if target is None or os.path.islink(dst) or not os.path.isfile(dst):
        return
    candidates = {os.fsencode(target), os.fsencode(target.replace(os.sep, "/"))}
    if os.path.getsize(dst) in {len(path) for path in candidates}:
        with open(dst, "rb") as file:
            archived_target = file.read()
        if archived_target in candidates:
            os.unlink(dst)


def link_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    target = relative_target(src, os.path.dirname(dst))
    remove_archived_symlink(dst, target)
    if os.path.lexists(dst):
        return
    if os.name != "nt" and target is not None:
        try:
            os.symlink(target, dst)
            return
        except OSError:
            pass
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/UIEB")
    ap.add_argument("--dst", default="data/UIEB/L50")
    ap.add_argument("--n_labeled", type=int, default=50)
    a = ap.parse_args()

    names = sorted(os.listdir(os.path.join(a.src, "labeled/input")))
    for n in names[:a.n_labeled]:
        for sub in ("input", "gt"):
            link_file(os.path.join(a.src, "labeled", sub, n), os.path.join(a.dst, "labeled", sub, n))
    for n in names[a.n_labeled:]:
        link_file(os.path.join(a.src, "labeled/input", n), os.path.join(a.dst, "unlabeled/input", n))
        link_file(os.path.join(a.src, "labeled/gt", n), os.path.join(a.dst, "unlabeled_gt_analysis", n))
    for n in sorted(os.listdir(os.path.join(a.src, "unlabeled/input"))):
        link_file(os.path.join(a.src, "unlabeled/input", n), os.path.join(a.dst, "unlabeled/input", n))
        link_file(os.path.join(a.src, "unlabeled_gt_analysis", n), os.path.join(a.dst, "unlabeled_gt_analysis", n))
    test_src = os.path.join(a.src, "test")
    test_dst = os.path.join(a.dst, "test")
    test_target = relative_target(test_src, a.dst)
    remove_archived_symlink(test_dst, test_target)
    if os.name != "nt" and test_target is not None and not os.path.lexists(test_dst):
        try:
            os.symlink(test_target, test_dst, target_is_directory=True)
        except OSError:
            pass
    if not os.path.islink(test_dst):
        for sub in ("input", "gt"):
            for n in sorted(os.listdir(os.path.join(test_src, sub))):
                link_file(os.path.join(test_src, sub, n), os.path.join(test_dst, sub, n))
    for d in ("labeled/input", "unlabeled/input", "unlabeled_gt_analysis"):
        print(d, len(os.listdir(os.path.join(a.dst, d))))


if __name__ == "__main__":
    main()
