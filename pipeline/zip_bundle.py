import zipfile, os

def create_zip_bundle(folder):
    license_txt = os.path.join(folder, "LICENSE.txt")

    with open(license_txt, "w") as f:
        f.write("Commercial Use License\nYou may sell unlimited products using this design.")

    zip_path = f"{folder}/bundle.zip"
    with zipfile.ZipFile(zip_path, 'w') as z:
        for file in os.listdir(folder):
            if file.endswith((".png", ".txt", ".json")):
                z.write(os.path.join(folder,file), file)

    return zip_path
