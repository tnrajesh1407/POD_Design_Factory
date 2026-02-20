import os
import replicate
import requests
from PIL import Image
from io import BytesIO
from pipeline.replicate_rate_limiter import wait_for_slot
from replicate.exceptions import ReplicateError
import time

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

from pipeline.replicate_rate_limiter import wait_for_slot
from replicate.exceptions import ReplicateError
import time

def generate_image(prompt, output_path):

    max_retries = 5
    delay = 10

    for attempt in range(max_retries):
        try:
            wait_for_slot()  # 🔥 throttle before request

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

            return

        except ReplicateError as e:
            if "429" in str(e):
                print("429 detected. Backing off...")
                time.sleep(delay)
                delay *= 1.5
            else:
                raise

    raise Exception("Failed after max retries")