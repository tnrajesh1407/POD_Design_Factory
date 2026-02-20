from rembg import remove
from PIL import Image

def remove_background(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    out = remove(img)                 # returns image with alpha
    out.save(output_path, "PNG")

def stabilize_alpha_edges(img: Image.Image, cutoff=22, feather=0.4):
    """
    Convert weak semi-transparent edge ramps into clean edges.
    Keeps interior shading intact.
    """
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)

    alpha = arr[:, :, 3].astype(np.float32)

    # Kill very low alpha
    alpha[alpha < cutoff] = 0

    # Optional slight blur to smooth harsh cut
    if feather > 0:
        alpha_img = Image.fromarray(alpha.astype(np.uint8), mode="L")
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(feather))
        alpha = np.array(alpha_img, dtype=np.float32)

    # Snap high values fully opaque
    alpha[alpha > 235] = 255

    arr[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(arr, "RGBA")
