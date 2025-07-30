import pandas as pd
import os

ims = pd.read_csv('data/images.txt', sep=' ', header=None)
print(len(ims))

labels = pd.read_csv('data/image_class_labels.txt', sep=' ', header=None)
print(len(labels))

print(len(os.listdir('all_images/')))