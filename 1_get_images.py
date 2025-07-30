import pandas as pd
from tqdm import tqdm
import shutil
import os
import numpy as np
import pickle

meta = pd.read_csv('data/mimic-cxr-2.0.0-metadata.csv')

# 96161 PA instances
meta = meta[meta['ViewPosition'] == 'PA']

# unique studies - 85872 PA instances
meta = meta.drop_duplicates('study_id')

studies = meta['study_id'].values
label = pd.read_csv('data/mimic-cxr-2.0.0-chexpert.csv')
print(label.columns)

# NOTE: use existing labels for now.

path_to_all_imgs = '../concept_bottleneck_clustered/1 - Concept Extraction/all_90k_PA!!/'
all_imgs = os.listdir(path_to_all_imgs)
num_healthy, num_cancer, num_pneumothorax, num_cardiomegaly, num_effusion, num_pneumonia, all = 0, 0, 0, 0, 0, 0, 0
labels, filenames = [], []
failed = 0

conditions = {
    'Cancer': ('Lung Lesion', 'c_', num_cancer),
    'Healthy': ('No Finding', 'h_', num_healthy),
    'Pneumothorax': ('Pneumothorax', 'pth_', num_pneumothorax),
    'Cardiomegaly': ('Cardiomegaly', 'car_', num_cardiomegaly),
    'Effusion': ('Pleural Effusion', 'e_', num_effusion),
    'Pneumonia': ('Pneumonia', 'p_', num_pneumonia)
}
#for i in tqdm(range(len(studies))):
for study in tqdm(studies):
    study_label = label[label['study_id'] == study]
    if study_label.empty:
        failed += 1
        continue

    filename_series = meta.loc[meta['study_id'] == study, 'dicom_id']
    if filename_series.empty:
        failed += 1
        continue

    filename = f"{filename_series.values[0]}.jpg"
    if filename not in all_imgs:
        failed += 1
        continue

    used = False
    for label_name, (col, prefix, counter) in conditions.items():
        if study_label[col].values[0] == 1.0:
            globals()[f"num_{label_name.lower()}"] += 1
            filenames.append((prefix if used else '') + filename)
            labels.append(label_name)
            used = True
            all += 1
        
print(f'Failed: {failed}')
assert len(filenames) == len(set(filenames)) # all filenames unique
print(f'All: {len(filenames)}')
assert len(filenames) == len(labels)

print('Writing files...')

with open('data/images.txt', 'w') as img_f, open('data/image_class_labels.txt', 'w') as lbl_f:
    for i, (file, cls) in enumerate(zip(filenames, labels)):
        target_file = file.split('_')[1] if '_' in file else file
        shutil.copy(path_to_all_imgs + target_file, 'all_images/' + file)
        img_f.write(f'{i} {file}\n')
        lbl_f.write(f'{i} {cls}\n')

print(f'Healthy: {num_healthy}')
print(f'Cancer: {num_cancer}')
print(f'Pneumothorax: {num_pneumothorax}') 
print(f'Cardiomegaly: {num_cardiomegaly}') 
print(f'Effusion: {num_effusion}')
print(f'Pneumonia: {num_pneumonia}') 
