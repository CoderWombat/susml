import os
import numpy as np
import shutil

base_path = os.getcwd()
data_path = os.path.join(base_path, "../dataset/train")
categories = os.listdir(data_path)
if '.DS_Store' in categories:
    categories.remove('.DS_Store')
test_path = os.path.join(base_path, "../dataset/val")

for cat in categories:
    image_files = os.listdir(os.path.join(data_path, cat))
    choices = np.random.choice([0, 1], size=(len(image_files),), p=[.85, .15])
    files_to_move = np.compress(choices , image_files)

    for _f in files_to_move:
        origin_path = os.path.join(data_path, cat, _f)
        dest_dir = os.path.join(test_path, cat)
        dest_path = os.path.join(test_path, cat, _f)
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        shutil.move(origin_path, dest_path)