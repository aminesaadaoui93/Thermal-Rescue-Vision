import kagglehub
import os
import shutil


file_path = os.path.abspath("./archive")
# Download latest version
path = kagglehub.dataset_download("samdazel/teledyne-flir-adas-thermal-dataset-v2")
file_name = os.path.basename(path)
print("Path to dataset files:", file_path, "\nDownloaded file:", file_name)
# imposing our structure on the downloaded dataset
"""wanted structure:/archive
/FLIR_ADAS_v2/
    /images_rgb_train/
    /images_rgb_val/
    /video_rgb_test
    /images_thermal_train/
    /images_thermal_val/
    /video_thermal_test"""
# knowing the downloaded structure, we can move the files to the wanted structure
the_downloaded_structure = ""
# we stop when we find images_rgb_train, because the rest of the structure is the same as the wanted one
for root, dirs, files in os.walk(path):
    if "images_rgb_train" in dirs:
        the_downloaded_structure = root
        break
else:
    raise Exception("The downloaded structure is not as expected.")
# we can create our wanted structure and move the files to it
os.makedirs(os.path.join(file_path, "FLIR_ADAS_v2"), exist_ok=True)
shutil.copytree(
    the_downloaded_structure,
    os.path.join(file_path, "FLIR_ADAS_v2"),
    dirs_exist_ok=True,
)
shutil.rmtree(path)
