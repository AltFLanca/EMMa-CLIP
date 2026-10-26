
# EMMa-CLIP

Generate text embeddings from EMMa product category hierarchies using the frozen CLIP ViT-B/32 text encoder. For each ASIN, a raw 512-dimensional embedding is obtained from its category path, ordered from root to leaf. For example:

```text
Electronics > Computers & Accessories > Computer Accessories > Keyboards
```

Dataset access and augmentation storage are handled through [EMMa-Wrapper](https://github.com/AltFLanca/EMMa-Wrapper).

## Setup

Install the project dependencies using [uv](https://docs.astral.sh/uv/). From the repository root, run:

```bash
uv sync --locked
```

A prepared EMMa dataset is required. The dataset and CLIP model weights are not included in this repository.

## Repository Structure

- `src/test_single.ipynb` - Notebook for testing individual samples.
- `src/test_inference.ipynb` - Notebook for testing batched inference.
- `src/inference.py` - Generates CLIP text embeddings from product category hierarchies.
- `src/valid_output_filtering.py` - Produces a list of valid ASINs from the generated augmentations.
- `src/inspect_augmentation_output.ipynb` - Inspects embedding statistics, cosine similarities, and nearest neighbors.

## Generate Augmentations

Embedding generation requires an `.npy` file containing the ASINs to process, provided through the `--asins` argument.

The CLIP text encoder can be configured using the `--clip-model` argument, which accepts either a local model directory or a Hugging Face model identifier.

Run inference with:

```bash
uv run src/inference.py \
  --dataset-root /path/to/EMMa \
  --asins /path/to/augmented_asins.npy \
  --output /path/to/EMMa-CLIP/clip_vit_b32_category_hierarchy_embeddings.h5 \
  --clip-model /path/to/models/openai/clip-vit-base-patch32
```

Generated embeddings are stored in an HDF5 file at the specified output path. To inspect the generated embeddings, use [inspect_augmentation_output.ipynb](src/inspect_augmentation_output.ipynb).

**Note:** Inference automatically skips ASINs already present in the output file. When changing models or inference settings, use a new output path to avoid mixing results generated with different configurations.

## Validate Augmentations

The [valid_output_filtering.py](src/valid_output_filtering.py) script processes the generated augmentations and produces a `valid_asins.npy` file in the same directory as the input HDF5 file.

Run validation with:

```bash
uv run src/valid_output_filtering.py \
  --dataset-root /path/to/EMMa \
  --augmentation-file /path/to/EMMa-CLIP/clip_vit_b32_category_hierarchy_embeddings.h5
```

Because the output is always named `valid_asins.npy`, validating multiple HDF5 files in the same directory overwrites the previous list.

**Note:** The current implementation does not apply any filtering criteria. All processed ASINs are considered valid.