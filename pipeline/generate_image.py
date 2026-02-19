import os
import replicate
import requests
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = "outputs"

def generate_image(prompt, output_path):
    print("Calling Replicate prunaai...")

    output = replicate.run(
        "prunaai/p-image",
        input={
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "num_outputs": 1
        }
    )

    # Replicate now returns FileOutput object
    image_url = output.url if hasattr(output, "url") else output[0].url

    img_data = requests.get(image_url).content
    with open(output_path, "wb") as f:
        f.write(img_data)

    print("Image saved:", output_path)
