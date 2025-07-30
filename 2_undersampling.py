from torchvision import transforms
import os
import pickle
from tqdm import tqdm
import pandas as pd
from collections import Counter
import numpy as np
from imblearn.under_sampling import OneSidedSelection
import cv2
import shutil
import glob

# get dataset

data_dir = "all_images/"
images = [data_dir + a for a in os.listdir(data_dir)]

image_info = pd.read_csv("data/images.txt", sep=" ", header=None)
image_info.columns = ["id", "name"]

class_info = pd.read_csv("data/image_class_labels.txt", sep=" ", header=None)
class_info.columns = ["id", "label"]

transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),  # implicitly divides by 255
    ]
)

all_imgs, all_labels, all_ids = [], [], []
get_id = image_info.set_index("name")["id"].to_dict()
get_label = class_info.set_index("id")["label"].to_dict()

for im in tqdm(images):
    filename = im.rsplit("/", 1)[-1]
    img = cv2.resize(cv2.imread(im), (299, 299))
    all_imgs.append(img)

    id_val = get_id.get(filename)
    if id_val is None:
        continue
    all_ids.append(id_val)

    label_val = get_label.get(id_val)
    if label_val is not None:
        all_labels.append(label_val)

# start with 43447
all_imgs = np.array(all_imgs)
all_labels = np.array(all_labels)
all_ids = np.array(all_ids)
assert len(all_imgs) == len(all_labels) == len(all_ids)
print(len(all_ids))


##########################################################################
# length = 71213


def undersample(imgs, labels, ids, idx):
    # summarize class distribution
    print(Counter(labels))
    # define the undersampling method
    undersample = OneSidedSelection(random_state=0, n_seeds_S=50)
    # transform the dataset
    imgs = imgs.reshape(imgs.shape[0], -1)
    X, y = undersample.fit_resample(imgs, labels)
    samples_used = ids[undersample.sample_indices_]
    # summarize the new class distribution
    print(Counter(y))
    correct_x = [im.reshape(299, 299, 3) for im in X]

    pickle.dump(correct_x, open(f"undersampling/imgs_{idx}.pkl", "wb"))
    pickle.dump(y, open(f"undersampling/labels_{idx}.pkl", "wb"))
    pickle.dump(samples_used, open(f"undersampling/ids_{idx}.pkl", "wb"))


idx = 0
step = 5000
limit = len(images) - 4000  # TODO: TOGGLE

while idx * step < limit:
    i = idx * step
    end = min(i + step, len(images))
    print(f"Range: {i} - {end}")

    all_imgs_cut = all_imgs[i:end]
    all_labels_cut = all_labels[i:end]
    all_ids_cut = all_ids[i:end]

    undersample(all_imgs_cut, all_labels_cut, all_ids_cut, idx)
    idx += 1

### COMBINING
from glob import glob


def combine_pickle_files(directory_path, output_file, start):

    combined_df = []

    for file_path in tqdm(glob(os.path.join(directory_path, f"{start}*.pkl"))):
        with open(file_path, "rb") as f:
            combined_df.extend(pickle.load(f))

    print(len(combined_df))
    with open(output_file, "wb") as out:
        pickle.dump(combined_df, out, protocol=pickle.HIGHEST_PROTOCOL)


print("Done. Combining files...")

combine_pickle_files("undersampling/", "undersampling/full_x.pkl", "imgs")
combine_pickle_files("undersampling/", "undersampling/full_y.pkl", "labels")
combine_pickle_files("undersampling/", "undersampling/full_ids.pkl", "ids")

out = pickle.load(open("undersampling/full_x.pkl", "rb"))
labels = pickle.load(open("undersampling/full_y.pkl", "rb"))
ids = pickle.load(open("undersampling/full_ids.pkl", "rb"))

# assert len(ids) == len(set(ids))

if not os.path.exists("undersampling/images/"):
    os.makedirs("undersampling/images/")
# files = glob.glob("undersampling/images/*")
# for f in files:
#     os.remove(f)

# NOTE: these are the original and incorrect labels
print("Generating images...")
from pathlib import Path

images_dir = Path("undersampling/images")
labels_dir = Path("undersampling")
images_dir.mkdir(parents=True, exist_ok=True)

existing_files = set(os.listdir(images_dir))

label_prefixes = {
    "Pneumothorax": "pthrx",
    "Pneumonia": "pneumo",
    "Cardiomegaly": "cardio",
}

for i in tqdm(range(len(out))):
    file_row = image_info.loc[image_info["id"] == ids[i]].iloc[0]
    label_row = class_info.loc[class_info["id"] == ids[i]].iloc[0]

    filename = file_row["name"]
    label = label_row["label"]

    label_path = labels_dir / label
    label_path.mkdir(parents=True, exist_ok=True)

    # Check duplication efficiently
    if filename in existing_files:
        prefix = label_prefixes.get(label, label[:4])
        filename = f"{prefix}_{filename}"

    shutil.copy(f"all_images/{file_row['name']}", label_path / filename)
    shutil.copy(f"all_images/{file_row['name']}", images_dir / filename)

    existing_files.add(filename)
