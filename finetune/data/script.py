import os
import shutil
from datasets import load_dataset, Video

dataset_name = "Wild-Heart/Disney-VideoGeneration-Dataset"
split = "train"

# 1. Load the dataset (decoding by default).
dataset = load_dataset(dataset_name, split=split)

# 2. Cast the 'video' column to avoid decoding it.
dataset = dataset.cast_column("video", Video(decode=False))

# Now, sample["video"] is NOT a VideoReader. Instead, it's a dict with "path"/"bytes".

save_dir = "/home/royve/Github/CogVideo/finetune/data/raw"
os.makedirs(save_dir, exist_ok=True)

for idx, sample in enumerate(dataset):
    # Option A: Copy from 'path' if available
    video_path = sample["video"]["path"]
    out_path = os.path.join(save_dir, f"video_{idx}.mp4")
    shutil.copy(video_path, out_path)
    
    # OR Option B: If you really need the in-memory bytes
    # video_data = sample["video"]["bytes"]
    # with open(out_path, "wb") as f:
    #     f.write(video_data)

print(f"Done! Saved {len(dataset)} videos to {save_dir}")
