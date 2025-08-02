import os
import logging
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# === Configuration ===
INPUT_DIR = "template_dataset"
OUTPUT_DIR = "augmented_dataset2"
AUG_PER_IMAGE = 100
IMAGE_SIZE = (224, 224)

# === Logging Setup ===
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "augmentation.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

logging.info("🔁 Starting Pokémon image augmentation...")
logging.info(f"Input dir: {INPUT_DIR} | Output dir: {OUTPUT_DIR} | Augmentations per image: {AUG_PER_IMAGE}")

# === Augmentation Pipeline ===
augment = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.ToPILImage(),
])

# === Ensure output folder exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Process Images ===
all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
success_count = 0
error_count = 0

for filename in tqdm(all_files, desc="Augmenting Pokémon images", dynamic_ncols=True):
    try:
        base = os.path.splitext(filename)[0].lower()
        if "shiny" in base:
            pokemon_name = base.replace("_shiny", "")
            class_name = f"{pokemon_name}_shiny"
        else:
            class_name = base.split("_")[0]

        src_path = os.path.join(INPUT_DIR, filename)
        dst_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        # Skip if already augmented
        existing_files = [
            f for f in os.listdir(dst_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if len(existing_files) >= AUG_PER_IMAGE:
            logging.info(f"⏭️  Skipping {filename} (already augmented)")
            continue

        original_img = Image.open(src_path).convert("RGB")
        base_name = os.path.splitext(filename)[0]

        for i in range(AUG_PER_IMAGE):
            aug_img = augment(original_img)
            save_name = f"{base_name}_aug_{i+1:03}.png"
            aug_img.save(os.path.join(dst_dir, save_name))

        logging.info(f"✅ Augmented: {filename}")
        success_count += 1

    except Exception as e:
        logging.error(f"❌ Error processing {filename}: {e}")
        error_count += 1

# === Summary ===
logging.info("🎉 Augmentation complete!")
logging.info(f"✅ Successful: {success_count}")
logging.info(f"❌ Failed: {error_count}")
