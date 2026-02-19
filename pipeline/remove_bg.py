from rembg import remove
from PIL import Image

def remove_background(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    out = remove(img)                 # returns image with alpha
    out.save(output_path, "PNG")

