import kagglehub
import os
import shutil


file_path = os.path.abspath("./")
# Download latest version
path = kagglehub.dataset_download(
    "samdazel/teledyne-flir-adas-thermal-dataset-v2",
)
shutil.copytree(path, file_path, dirs_exist_ok=True)
os.remove(path)
file_name = os.path.basename(path)
print("Path to dataset files:", file_path, "\nDownloaded file:", file_name)
