Inference run with command:

```
uv run src/inference.py --dataset-root /home/user/datasets/EMMa --asins /home/user/datasets/EMMa-ASINs/augmented_asins.npy --output /home/user/datasets/EMMa-CLIP/clip_vit_b32_category_hierarchy_embeddings.h5 --clip-model /home/user/models/openai/clip-vit-base-patch32
```

Then:

```
uv run src/valid_output_filtering.py --dataset-root /home/user/datasets/EMMa --augmentation-file /home/user/datasets/EMMa-CLIP/clip_vit_b32_category_hierarchy_embeddings.h5
```