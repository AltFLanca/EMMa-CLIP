from pathlib import Path
import argparse
import numpy as np

from emma import EmmaDataset, HDF5Source
import hdf5plugin

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-root",
    type=Path,
    required=True,
    help="Directory containing the EMMa dataset.",
)
parser.add_argument(
    "--augmentation-file",
    type=Path,
    required=True,
    help="Path to the CLIP category hierarchy embeddings HDF5 file.",
)
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
CLIP_EMBEDDING_PATH = args.augmentation_file

AUGMENTATION_DIR = CLIP_EMBEDDING_PATH.parent
VALID_ASINS_PATH = AUGMENTATION_DIR / "valid_asins.npy"

ds = EmmaDataset(DATASET_ROOT)

source = HDF5Source("clip_vit_b32_category_hierarchy_embeddings", CLIP_EMBEDDING_PATH, mode="r")
asins = set(source.keys())

def is_valid_embedding(source, asin):
    # No filtering
    return True

valid_asins = [
    asin
    for asin in asins
    if is_valid_embedding(source, asin)
]
valid_asins = np.asarray(valid_asins, dtype=str)

if not len(valid_asins):
    raise RuntimeError("No valid ASINs.")

print(f"Augmentation ASINs: {len(asins)}")
print(f"Excluded ASINs: {len(asins) - len(valid_asins)}")

AUGMENTATION_DIR.mkdir(parents=True, exist_ok=True)
np.save(VALID_ASINS_PATH, valid_asins)
print(f"Saved {len(valid_asins)} ASINs to: {VALID_ASINS_PATH}")