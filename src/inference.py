from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from typing import Any

from torch.utils.data import DataLoader
from emma import EmmaDataset
from emma.augment import HDF5Source
from emma.dataloader import EmmaMapDataset
import hdf5plugin
from transformers import AutoTokenizer, CLIPTextModelWithProjection

import utils.logging_ as logging
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode EMMa product category hierarchies with a frozen CLIP "
            "text encoder and store one raw 512-dimensional embedding per ASIN."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing the EMMa dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 file. Defaults to <dataset_root>/<augmentation-name>.h5.",
    )
    parser.add_argument(
        "--augmentation-name",
        default=None,
        help="Registered augmentation name. A source-specific default is used when omitted.",
    )
    parser.add_argument(
        "--asins",
        type=Path,
        required=True,
        help="Path to a .npy file containing the ASINs to process.",
    )
    parser.add_argument(
        "--clip-model",
        required=True,
        help="Hugging Face model ID or local directory for CLIP ViT-B/32.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="DataLoader and CLIP inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor used only when num_workers > 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N pending ASINs.",
    )
    return parser.parse_args()


def load_asins(path: Path) -> list[str]:
    asins = np.load(path, allow_pickle=False)
    return [str(asin) for asin in asins.reshape(-1)]


def make_transform(ds: EmmaDataset):
    def transform(asin: str, rec: dict) -> dict:
        return {
            "asin": asin,
            "categories": ds.get_category_names(rec=rec),
        }

    return transform


def collate_samples(samples: list[dict]) -> list[dict]:
    return samples


def make_writer(name: str, path: Path) -> HDF5Source:
    compression = hdf5plugin.Blosc(
        cname="lz4hc",
        clevel=5,
        shuffle=hdf5plugin.Blosc.SHUFFLE,
    )
    return HDF5Source(
        name,
        path,
        mode="a",
        compression=compression,
        shuffle=False,
        fletcher32=False,
    )


def missing_asins(writer: HDF5Source, asins: list[str]) -> list[str]:
    existing = set(writer.keys())
    return [asin for asin in asins if asin not in existing]


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def flatten_category_hierarchy(categories: Any) -> list[str]:
    if isinstance(categories, str):
        value = categories.strip()
        return [value]
    if isinstance(categories, (list, tuple)):
        hierarchy: list[str] = []
        for category in categories:
            hierarchy.extend(flatten_category_hierarchy(category))
        return hierarchy

    value = str(categories).strip()
    return [value]


def sample_text(sample: dict) -> str:
    hierarchy = flatten_category_hierarchy(sample["categories"])
    return " > ".join(hierarchy)


@torch.inference_mode()
def encode_texts(
    model: CLIPTextModelWithProjection,
    tokenizer: Any,
    device: torch.device,
    texts: list[str],
) -> torch.Tensor:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings,
        return_tensors="pt",
    ).to(device)

    embeddings = model(**inputs).text_embeds

    return embeddings.float().cpu()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")

    source_name = "category_hierarchy"
    augmentation_name = args.augmentation_name or f"clip_vit_b32_{source_name}_embeddings"

    dataset_root = args.dataset_root
    output_path = args.output or (dataset_root / f"{augmentation_name}.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving augmentation to {output_path}")

    device = select_device(args.device)
    print(f"Loading CLIP text encoder from {args.clip_model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.clip_model)
    model = CLIPTextModelWithProjection.from_pretrained(
        args.clip_model,
        use_safetensors=True,
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    ds = EmmaDataset(dataset_root)
    asins = load_asins(args.asins)
    print(f"Candidate ASINs: {len(asins)}")

    writer = make_writer(augmentation_name, output_path)
    pending_asins = missing_asins(writer, asins)
    if args.limit is not None:
        pending_asins = pending_asins[: args.limit]
    print(f"Pending ASINs: {len(pending_asins)}")

    if not pending_asins:
        writer.close()
        print("Nothing to do.")
        return

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": collate_samples,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(
        EmmaMapDataset(ds, pending_asins, transform=make_transform(ds)),
        **loader_kwargs,
    )

    start = time.time()
    for batch_idx, batch in enumerate(loader):
        texts = [sample_text(sample) for sample in batch]
        embeddings = encode_texts(model, tokenizer, device, texts).numpy()

        for sample, embedding in zip(batch, embeddings):
            writer.save(sample["asin"], embedding.astype(np.float32, copy=False))

        logging.progress_bar(batch_idx, len(loader), 1, 1)

    writer.close()
    print(f"Elapsed: {time.time() - start:.2f}s")
    print("Done")


if __name__ == "__main__":
    main()

