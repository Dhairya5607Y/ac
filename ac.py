import discord
import aiohttp
import aiofiles
import os
import torch
import cv2
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import numpy as np

TOKEN = "YOUR_DISCORD_USER_TOKEN"  # Use with caution
POKETWO_ID = 716390085896962058  # Pokétwo bot ID
TARGET_CHANNEL_IDS = [1234567890]  # Replace with actual channel IDs
MODEL_PATH = "models/efficientnetb0.pth"
CLASS_NAMES_PATH = "models/class_names.txt"
UNKNOWN_DIR = "unknown_pokemon"

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# === Load class names ===
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# === Load model ===
model = EfficientNet.from_name('efficientnet-b0', num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# === Image transform for AI ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Template Matching ===
def try_template_match(image_path, sprites_dir='sprites_all_forms', threshold=0.85):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for fname in os.listdir(sprites_dir):
        template = cv2.imread(os.path.join(sprites_dir, fname), 0)
        if template is None:
            continue
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return fname.split("_")[0].lower()
    return None

# === AI Prediction ===
def try_ai(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        predicted_idx = torch.argmax(outputs).item()
        return class_names[predicted_idx]

# === Save unknowns ===
def save_unknown(image_path):
    base_name = f"unknown_{len(os.listdir(UNKNOWN_DIR)) + 1}.png"
    save_path = os.path.join(UNKNOWN_DIR, base_name)
    os.rename(image_path, save_path)
    print(f"🕵️ Saved unknown Pokémon to {save_path}")

# === Download image from URL ===
async def download_image(url, save_path="current_pokemon.png"):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                f = await aiofiles.open(save_path, mode='wb')
                await f.write(await resp.read())
                await f.close()
                print(f"📥 Image downloaded to {save_path}")

# === Discord Setup ===
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"🤖 Logged in as {client.user}!")

@client.event
async def on_message(message):
    if message.channel.id not in TARGET_CHANNEL_IDS:
        return

    if message.author.id == POKETWO_ID:
        if message.embeds:
            embed = message.embeds[0]

            if "captcha" in embed.description.lower():
                await message.channel.send("@Pokétwo inc p all")
                print("⚠️ CAPTCHA detected! Sent bypass command.")
                return

            if "A wild pokémon has appeared!" in embed.title:
                image_url = embed.image.url
                await download_image(image_url)

                name = try_template_match("current_pokemon.png")
                if not name:
                    name = try_ai("current_pokemon.png")

                if name:
                    await message.channel.send(f"@Pokétwo c {name}")
                    print(f"✅ Caught Pokémon: {name}")
                else:
                    save_unknown("current_pokemon.png")

# === Run Bot ===
client.run(TOKEN, bot=False)


