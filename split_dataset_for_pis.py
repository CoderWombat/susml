import os
import numpy as np
import shutil

base_path = '/Users/Emanuel/Documents/Studium/Master/SusML/susml/'
data_path = os.path.join(base_path, "dataset/train")
categories = os.listdir(data_path)
if '.DS_Store' in categories:
    categories.remove('.DS_Store')
test_path = os.path.join(base_path, "dataset/val")

for cat in categories:
    train_files = os.listdir(os.path.join(data_path, cat))
    test_files = os.listdir(os.path.join(test_path, cat))
    train_choices = np.random.choice(range(12), size=(len(train_files)))
    test_choices = np.random.choice(range(12), size=(len(test_files)))

    for i in range(len(train_choices)):
        origin_path = os.path.join(data_path, cat, train_files[i])
        dest_dir = os.path.join(base_path, "pi-data/", str(train_choices[i]), "dataset/train", cat)
        dest_path = os.path.join(dest_dir, train_files[i])
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        shutil.copyfile(origin_path, dest_path)

    for i in range(len(test_choices)):
        origin_path = os.path.join(test_path, cat, test_files[i])
        dest_dir = os.path.join(base_path, "pi-data/", str(test_choices[i]), "dataset/val", cat)
        dest_path = os.path.join(dest_dir, test_files[i])
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        shutil.copyfile(origin_path, dest_path)