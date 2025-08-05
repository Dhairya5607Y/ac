import discord
import cv2
import numpy as np
import pytesseract
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import requests
from io import BytesIO
import torch.nn as nn
import torch.nn.functional as F

# ----- Configuration -----
TOKEN = "YOUR_USER_TOKEN"  # ⚠️ Use your real user token here — BE CAREFUL
TARGET_CHANNEL_ID = 123456789012345678  # Replace with your target channel ID
POKETWO_ID = 716390085896962058  # Pokétwo bot ID

TEMPLATE_FOLDER = "./template_dataset"
MODEL_PATH = "./model.pth"

# ----- Discord Intents -----
intents = discord.Intents.default()
intents.message_content = True

# ----- Use discord.py-self Client -----
client = discord.Client(intents=intents, self_bot=True)

# ----- Model Definition -----
class PokemonNet(nn.Module):
    def __init__(self, num_classes=150):
        super(PokemonNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 62 * 62, num_classes)  # For 128x128 input

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 62 * 62)
        x = self.fc1(x)
        return x

# ----- Load Model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PokemonNet(num_classes=150).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ----- Utility Functions -----
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def crop_pokemon_area(image: Image.Image):
    w, h = image.size
    return image.crop((40, 40, w - 40, h - 40))

def remove_background(image: Image.Image):
    image_np = np.array(image.convert("RGBA"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
    _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    image_np[:, :, 3] = alpha
    return Image.fromarray(image_np)

def match_with_templates(pokemon_image: Image.Image):
    pokemon_cv = np.array(pokemon_image.convert("RGB"))
    pokemon_gray = cv2.cvtColor(pokemon_cv, cv2.COLOR_BGR2GRAY)

    for filename in os.listdir(TEMPLATE_FOLDER):
        template_path = os.path.join(TEMPLATE_FOLDER, filename)
        template = cv2.imread(template_path, 0)
        if template is None:
            continue
        res = cv2.matchTemplate(pokemon_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > 0.85:
            return os.path.splitext(filename)[0]
    return None

def predict_with_model(pokemon_image: Image.Image):
    image_tensor = transform(pokemon_image).unsqueeze(0).to(device)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
    return str(predicted.item())  # Replace with class-to-name mapping if needed

def extract_with_ocr(image: Image.Image):
    text = pytesseract.image_to_string(image)
    return text.strip().split("\n")[0]

def save_for_dataset(pokemon_image: Image.Image, name: str):
    if not os.path.exists(TEMPLATE_FOLDER):
        os.makedirs(TEMPLATE_FOLDER)
    out_path = os.path.join(TEMPLATE_FOLDER, f"{name}.png")
    pokemon_image.save(out_path)

# ----- Discord Events -----
@client.event
async def on_ready():
    print(f"[+] Logged in as {client.user} (ID: {client.user.id})")

@client.event
async def on_message(message):
    if message.channel.id != TARGET_CHANNEL_ID:
        return

    if message.author.id != POKETWO_ID:
        return

    if "Please use the command" in message.content and "to prove you’re human!" in message.content:
        await message.channel.send("@Pokétwo inc p")
        print("[!] Captcha detected — Sent inc p")
        return

    if message.embeds:
        embed = message.embeds[0]
        if embed.image:
            img_url = embed.image.url
            image = download_image(img_url)

            cropped = crop_pokemon_area(image)
            transparent = remove_background(cropped)

            name = match_with_templates(transparent)
            if not name:
                try:
                    name = predict_with_model(transparent)
                except Exception:
                    name = extract_with_ocr(transparent)

            if name:
                await message.channel.send(f"@Pokétwo c {name}")
                print(f"[+] Tried to catch: {name}")
                save_for_dataset(transparent, name)
            else:
                print("[!] Failed to identify Pokémon.")

# ----- Start Selfbot -----
client.run(TOKEN, bot=False)

