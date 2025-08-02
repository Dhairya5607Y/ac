import os
import random
import glob

AUG_DIR = "augmented_dataset"
KEEP_PER_CLASS = 3  # Keep only 4 out of 100

assert os.path.isdir(AUG_DIR), "❌ augmented_dataset folder not found."

total_deleted = 0

for pokemon in os.listdir(AUG_DIR):
    class_dir = os.path.join(AUG_DIR, pokemon)
    if not os.path.isdir(class_dir):
        continue

    images = glob.glob(os.path.join(class_dir, "*.png"))
    if len(images) <= KEEP_PER_CLASS:
        print(f"⏭️ Skipping {pokemon}: only {len(images)} images")
        continue

    to_delete = random.sample(images, len(images) - KEEP_PER_CLASS)
    for img_path in to_delete:
        os.remove(img_path)
    total_deleted += len(to_delete)
    print(f"🗑️ Deleted {len(to_delete)} from {pokemon}")

print(f"\n✅ Done. Total images deleted: {total_deleted}")
