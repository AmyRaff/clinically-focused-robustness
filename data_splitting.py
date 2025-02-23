from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast

# NOTE: to handle duplicates, do separately. not all are in 'images/'

with open('concept_representations.txt', 'r') as f:
    data = f.readlines()
    
# get filenames, original labels, and correct labels
filenames, full_report_concepts = [], []
for i in tqdm(range(len(data))):
    line = data[i]
    
    if line.startswith('FILENAME:'):
        filename = line.split('FILENAME: ')[1].split(' ')[0]
        filenames.append(filename)
    
    elif line.startswith('CONCEPTS:'):
        report_concs = line.split('CONCEPTS: ')[1]
        full_report_concepts.append(ast.literal_eval(report_concs))
    
# NOTE: indices correspond to classes.txt
labels = ['Healthy', 'Cancer', 'Cardiomegaly', 'Effusion',  'Pneumonia', 'Pneumothorax']

def generate_files(label, i):

    for im in os.listdir(f'undersampling/{label}/'):
        
        with open('undersampling/images.txt', 'a') as f:
                f.write(f'{i} {im}\n')
                
        with open('undersampling/image_class_labels.txt', 'a') as f:
                f.write(f"{i} {labels.index(label)}\n")
        
        idx = filenames.index(im)
        concs = full_report_concepts[idx]
        
        with open('undersampling/image_attribute_labels.txt', 'a') as f:
            for c in range(len(concs)):
                f.write(f"{i} {c} {concs[c]}\n")
                
        i+=1
        
    return i

i = 0
for l in labels:
    i = generate_files(l, i)
 
    
im_info = pd.read_csv('undersampling/images.txt', header=None)

cls_info = pd.read_csv('undersampling/image_class_labels.txt', header=None)

attr_info = pd.read_csv('undersampling/image_attribute_labels.txt', header=None)

# delete files before running!!!
assert len(im_info) == len(cls_info) == len(attr_info) / 17

image_ids = pd.read_csv('undersampling/images.txt', sep=' ', header=None)
image_ids.columns=['Im_ID', 'Im_Name']

label_data = pd.read_csv('undersampling/image_class_labels.txt', sep=' ', header=None)
label_data.columns = ['X', 'y']
X = np.array(label_data['X'].tolist())
y = np.array(label_data['y'].tolist())

print(len(y)) # 35892
for i in range(6):
    print(list(y).count(i)) # 16203, 1429, 5737, 7428, 3228, 1867

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1)
print(len(X_train)) # 32302
print(len(X_test)) # 3590

for i in range(6):
    print(list(y_train).count(i)) # 14583, 1286, 5163, 6685, 2905, 1680
    print(list(y_test).count(i)) # 1620, 143, 574, 743, 323, 187

written = 0
with open('undersampling/train_test_split.txt', 'w') as f:
    for i in tqdm(X_train):
        is_training = 1
        f.write('{} {}\n'.format(i, is_training))
    for i in tqdm(X_test):
        is_training = 0
        written +=1
        f.write('{} {}\n'.format(i, is_training))
        
print(len(X_test))
print(written)