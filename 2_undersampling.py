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

data_dir = "all_data/"
images = [data_dir + a for a in os.listdir(data_dir)]

image_info = pd.read_csv("images.txt", sep=" ", header=None)
image_info.columns = ["id", "name"]

class_info = pd.read_csv("image_class_labels.txt", sep=" ", header=None)
class_info.columns = ["id", "label"]

# transform = transforms.Compose(
#     [
#         transforms.Resize((299, 299)),
#         transforms.ToTensor(),  # implicitly divides by 255
#     ]
# )

# all_imgs, all_labels, all_ids = [], [], []
# for im in tqdm(images):
#     filename = im.split("/")[-1]

#     # img = transform(Image.open(im).convert('RGB'))
#     img = cv2.imread(im)
#     img = cv2.resize(img, (299, 299))
#     # img = img[np.newaxis, :]
#     all_imgs.append(np.array(img))

#     id = image_info[image_info["name"] == filename]["id"].values[0]
#     label = class_info[class_info["id"] == id]["label"].values[0]
#     all_labels.append(label)
#     all_ids.append(id)

# # start with 43447
# all_imgs = np.array(all_imgs)
# all_labels = np.array(all_labels)
# all_ids = np.array(all_ids)


##########################################################################
# length = 71213


def undersample(imgs, labels, ids, idx):
    # summarize class distribution
    counter = Counter(labels)
    print(counter)
    # define the undersampling method
    undersample = OneSidedSelection(random_state=0, n_seeds_S=50)
    # transform the dataset
    imgs = imgs.reshape(imgs.shape[0], -1)
    X, y = undersample.fit_resample(imgs, labels)
    samples_used = undersample.sample_indices_
    samples_used = ids[samples_used]
    # summarize the new class distribution
    counter = Counter(y)
    print(counter)
    correct_x = []
    for im in X:
        reshaped = np.reshape(im, (299, 299, 3))
        correct_x.append(reshaped)
    pickle.dump(correct_x, open(f"undersampling/imgs_{idx}.pkl", "wb"))
    pickle.dump(y, open(f"undersampling/labels_{idx}.pkl", "wb"))
    pickle.dump(samples_used, open(f"undersampling/ids_{idx}.pkl", "wb"))


# idx = 0
# for i in range(0, len(images) - 4000, 5000):
#     print(f"Range: {i} - {i + 5000}")
#     if i != 65000:
#         all_imgs_cut, all_labels_cut, all_ids_cut = (
#             all_imgs[i : i + 5000],
#             all_labels[i : i + 5000],
#             all_ids[i : i + 5000],
#         )
#     else:
#         all_imgs_cut, all_labels_cut, all_ids_cut = (
#             all_imgs[i:],
#             all_labels[i:],
#             all_ids[i:],
#         )
#     undersample(all_imgs_cut, all_labels_cut, all_ids_cut, idx)
#     idx += 1

### COMBINING


def combine_pickle_files(directory_path, output_file, start):
    combined_df = []  # Initialize an empty DataFrame to store the merged data

    for file_name in tqdm(os.listdir(directory_path)):
        if file_name.endswith(".pkl") and file_name.startswith(start):
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, "rb") as f:
                content = pickle.load(f)
                for thing in content:
                    combined_df.append(thing)

    print(len(combined_df))
    with open(output_file, "wb") as out:
        pickle.dump(combined_df, out, protocol=pickle.HIGHEST_PROTOCOL)


# print("Done. Combining files...")

# combine_pickle_files("undersampling/", "undersampling/full_x.pkl", "imgs")
# combine_pickle_files("undersampling/", "undersampling/full_y.pkl", "labels")
# combine_pickle_files("undersampling/", "undersampling/full_ids.pkl", "ids")

out = pickle.load(open("undersampling/full_x.pkl", "rb"))
labels = pickle.load(open("undersampling/full_y.pkl", "rb"))
ids = pickle.load(open("undersampling/full_ids.pkl", "rb"))

# assert len(ids) == len(set(ids))

# if not os.path.exists("undersampling/images/"):
#     os.makedirs("undersampling/images/")
# files = glob.glob("undersampling/images/*")
# for f in files:
#     os.remove(f)

# NOTE: these are the original and incorrect labels
print("Generating images...")
for i in tqdm(range(len(out))):
    filename = image_info[image_info["id"] == ids[i]]["name"].values[0]
    label = class_info[class_info["id"] == ids[i]]["label"].values[0]
    if not os.path.exists(f"undersampling/{label}/"):
        os.makedirs(f"undersampling/{label}/")
    # duplication handling
    if filename in os.listdir("undersampling/images/"):
        if label == "Pneumothorax":
            prefix = "pthrx"
        elif label == "Pneumonia":
            prefix = "pneumo"
        elif label == "Cardiomegaly":
            prefix = "cardio"
        else:
            prefix = label[:4]
        new_filename = prefix + "_" + filename
        #shutil.copy(f"all_data/{filename}", f"undersampling/{label}/{new_filename}")
        #shutil.copy(f"all_data/{filename}", f"undersampling/images/{new_filename}")
    else:
        #shutil.copy(f"all_data/{filename}", f"undersampling/{label}/{filename}")
        #shutil.copy(f"all_data/{filename}", f"undersampling/images/{filename}")
