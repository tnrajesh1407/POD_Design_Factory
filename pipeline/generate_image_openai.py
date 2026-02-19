from openai import OpenAI
import base64

client = OpenAI()

def generate_image(prompt, output_path):
    try:
        print("Calling OpenAI image generation...")

        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024"
        )

        image_base64 = result.data[0].b64_json

        with open(output_path, "wb") as f:
            f.write(base64.b64decode(image_base64))

        print("Image saved:", output_path)

    except Exception as e:
        print("OPENAI ERROR:", e)
        raise e

