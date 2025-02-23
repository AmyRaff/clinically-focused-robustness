import pandas as pd
from tqdm import tqdm
import shutil
import os
import numpy as np
import pickle

meta = pd.read_csv('mimic-cxr-2.0.0-metadata.csv')

# 96161 PA instances
meta = meta[meta['ViewPosition'] == 'PA']

# unique studies - 85872 PA instances
meta = meta.drop_duplicates('study_id')


studies = meta['study_id'].values
label = pd.read_csv('mimic-cxr-2.0.0-chexpert.csv')
print(label.columns)

# NOTE: use existing labels for now.

path_to_all_imgs = '../concept_bottleneck_clustered/1 - Concept Extraction/all_90k_PA!!/'

num_healthy, num_cancer, num_pneumothorax, num_cardio, num_effusion, num_pneumonia, all = 0, 0, 0, 0, 0, 0, 0
labels, filenames = [], []
failed = 0
for i in tqdm(range(len(studies))):
    study = studies[i]
    if len(label[label['study_id'] == study]) > 0: 
        filename = meta[meta['study_id'] == study]['dicom_id']
        if len(filename) > 0:
            filename = filename.values[0] + '.jpg'
            if filename not in os.listdir(path_to_all_imgs):
                failed +=1
                continue
            all+=1
            is_healthy = label[label['study_id'] == study]['No Finding'].values[0]
            is_cancer = label[label['study_id'] == study]['Lung Lesion'].values[0]
            is_pneumothorax = label[label['study_id'] == study]['Pneumothorax'].values[0]
            is_cardio = label[label['study_id'] == study]['Cardiomegaly'].values[0]
            is_effusion = label[label['study_id'] == study]['Pleural Effusion'].values[0]
            is_pneumonia = label[label['study_id'] == study]['Pneumonia'].values[0]
            used = False  # NOTE: some images belong to more than one class
            if is_cancer == 1.0: 
                num_cancer +=1
                if used:
                    filenames.append('c_' + filename)
                else:
                    filenames.append(filename)
                used = True
                labels.append('Cancer')
            if is_healthy == 1.0: 
                num_healthy +=1
                if used:
                    filenames.append('h_' + filename)
                else:
                    filenames.append(filename)
                used = True
                labels.append('Healthy')
            if is_pneumothorax == 1.0: 
                num_pneumothorax +=1
                if used:
                    filenames.append('pth_' + filename)
                else:
                    filenames.append(filename)
                used = True
                labels.append('Pneumothorax')
            if is_cardio == 1.0: 
                num_cardio +=1
                if used:
                    filenames.append('car_' + filename)
                else:
                    filenames.append(filename)
                used = True
                labels.append('Cardiomegaly')
            if is_effusion == 1.0: 
                num_effusion +=1
                if used:
                    filenames.append('e_' + filename)
                else:
                    filenames.append(filename)
                used = True
                labels.append('Effusion')
            if is_pneumonia == 1.0: 
                num_pneumonia +=1
                if used:
                    filenames.append('p_' + filename)
                else:
                    filenames.append(filename)
                used = True
                labels.append('Pneumonia')
        
# print(f'Failed: {failed}')
assert len(filenames) == len(set(filenames))
print(f'All: {len(filenames)}')
assert len(filenames) == len(labels)

print('Writing files...')

for i in tqdm(range(len(filenames))):
    file, cls = filenames[i], labels[i]
    if '_' in file:
        shutil.copy(path_to_all_imgs + file.split('_')[1], 'all_data/' + file)
        # file = file.split('_')[1]
    else:
        shutil.copy(path_to_all_imgs + file, 'all_data/' + file)
        
    with open('images.txt', 'a') as f:
            f.write(f'{i} {file}\n')
            
    with open('image_class_labels.txt', 'a') as f:
            f.write(f'{i} {cls}\n')

print(f'Healthy: {num_healthy}')
print(f'Cancer: {num_cancer}')
print(f'Pneumothorax: {num_pneumothorax}') 
print(f'Cardiomegaly: {num_cardio}') 
print(f'Effusion: {num_effusion}')
print(f'Pneumonia: {num_pneumonia}') 
