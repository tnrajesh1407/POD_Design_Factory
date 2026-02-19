from PIL import Image

def upscale_image(input_path,output_path):
    img = Image.open(input_path)
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    img.save(output_path)