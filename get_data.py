import shutil
import os

# Define the source directory path
source_dir = '/kaggle/input/wikiart/Cubism'

# Define the destination directory path where you want to save the dataset
destination_dir = '/kaggle/working/Cubism'

# Copy the entire directory from source to destination
shutil.copytree(source_dir, destination_dir)

# List the files in the destination directory to verify the copy
print("Files in the Cubism directory:")
for dirname, _, filenames in os.walk(destination_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))
