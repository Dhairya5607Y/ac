import requests
import os
from PIL import Image
from io import BytesIO
import time

BASE_URL = "https://pokeapi.co/api/v2/pokemon/"
SAVE_DIR = "sprites_all_forms"
IMAGE_SIZE = (96, 96)

os.makedirs(SAVE_DIR, exist_ok=True)

def download_and_resize_image(url, save_path):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGBA")
        try:
            resample_method = Image.Resampling.LANCZOS  # Pillow >= 10.0
        except AttributeError:
            resample_method = Image.ANTIALIAS  # Pillow < 10.0 fallback
        img = img.resize(IMAGE_SIZE, resample=resample_method)
        img.save(save_path)
        print(f"✅ Saved {save_path}")
    except Exception as e:
        print(f"⚠️ Failed to download {url}: {e}")

def main():
    # Get total number of Pokémon entries
    r = requests.get(BASE_URL + "?limit=1")
    total = r.json()['count']
    print(f"🔢 Total Pokémon entries: {total}")

    for poke_id in range(1, total + 1):
        try:
            r = requests.get(f"{BASE_URL}{poke_id}")
            if r.status_code == 404:
                print(f"🚫 Pokémon ID {poke_id} not found. Skipping.")
                continue

            r.raise_for_status()
            data = r.json()
            name = data['name']

            # Default sprite
            front_default = data['sprites']['front_default']
            default_path = os.path.join(SAVE_DIR, f"{name}_default.png")
            if front_default and not os.path.exists(default_path):
                download_and_resize_image(front_default, default_path)
            else:
                print(f"⏩ Skipping {default_path} (already exists)")

            # Shiny sprite
            front_shiny = data['sprites']['front_shiny']
            shiny_path = os.path.join(SAVE_DIR, f"{name}_shiny.png")
            if front_shiny and not os.path.exists(shiny_path):
                download_and_resize_image(front_shiny, shiny_path)
            else:
                print(f"⏩ Skipping {shiny_path} (already exists)")

            time.sleep(0.2)  # Respect API rate limits

        except Exception as e:
            print(f"💥 Error processing Pokémon ID {poke_id}: {e}")

if __name__ == "__main__":
    main()
