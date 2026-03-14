import os
import gdown
import zipfile

data_path = os.path.abspath("./")


def download_PST900():
    url = "https://drive.google.com/uc?id=1X9n2s8m7l3v5k6j8n9o0p1q2r3s4t5u6v7w8x9y0z"
    output = os.path.join(data_path, "PST900.zip")
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(data_path)
    os.remove(output)
    print("PST900 dataset downloaded and extracted successfully.")
