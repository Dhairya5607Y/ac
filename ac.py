import discord
from discord.ext import commands
import requests
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pyautogui
import cv2
import numpy as np
import os
import time
from datetime import datetime
import io

# === CONFIG ===
TOKEN = "YOUR_DISCORD_ACCOUNT_TOKEN"
POKETWO_ID = 716390085896962058
CHANNEL_IDS = [123456789012345678, 234567890987654321]  # Allowed channel IDs

TEMPLATE_DIR = "sprites_all_forms"
MODEL_PATH = "models/efficientnetb0.pth"
CLASS_NAMES_PATH = "models/class_names.txt"
UNKNOWN_DIR = "unknown_pokemon"
CROP_REGION = (650, 300, 720, 420)  # Adjust based on screen resolution and bot layout

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# === AI Model Setup ===
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_name("efficientnet-b0")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Discord Client ===
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", self_bot=True, intents=intents)

# === Template Matching ===
def match_template(pokemon_img):
    best_score = 0
    best_name = None

    for filename in os.listdir(TEMPLATE_DIR):
        template = cv2.imread(os.path.join(TEMPLATE_DIR, filename))
        if template is None:
            continue

        template = cv2.resize(template, (96, 96))
        res = cv2.matchTemplate(np.array(pokemon_img), template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_score and max_val > 0.7:
            best_score = max_val
            best_name = filename.split("_")[0].lower()

    return best_name

# === AI Classification ===
def predict_pokemon(img: Image.Image):
    try:
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = probs.argmax().item()
            confidence = probs[0][pred_idx].item()
            return class_names[pred_idx] if confidence > 0.7 else None
    except Exception as e:
        print(f"❌ AI prediction failed: {e}")
        return None

# === Image Cropper ===
def crop_pokemon():
    x1, y1, x2, y2 = CROP_REGION
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    return screenshot

# === Save to Unknown ===
def save_unknown(img):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(UNKNOWN_DIR, f"unknown_{timestamp}.png")
    img.save(path)
    print(f"📁 Saved unknown Pokémon to {path}")

# === Discord Bot Events ===
@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.channel.id not in CHANNEL_IDS or message.author.id != POKETWO_ID:
        return

    # === Captcha Detection ===
    if "Please type the following command to verify that you're human" in message.content:
        await message.channel.send("@Pokétwo inc p all")
        print("⚠️ Captcha detected — verification sent.")
        return

    # === Spawned Pokémon Embed Handling ===
    if message.embeds:
        emb = message.embeds[0]
        if "A wild pokémon has appeared!" in emb.description.lower():
            print("🔍 Pokémon detected. Analyzing...")
            time.sleep(1.5)  # Wait for image to fully render

            img = crop_pokemon()
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # === Try Template Matching ===
            result = match_template(img_cv)
            if result:
                await message.channel.send(f"@Pokétwo c {result}")
                print(f"🎯 Caught using CV: {result}")
                return

            # === Try AI Model ===
            result = predict_pokemon(img)
            if result:
                await message.channel.send(f"@Pokétwo c {result}")
                print(f"🤖 Caught using AI: {result}")
                return

            # === Save Unknown Pokémon ===
            save_unknown(img)

# === Run Bot ===
bot.run(TOKEN, bot=False)
