import ast
from tqdm import tqdm 
import shutil
import os

with open('concept_representations.txt', 'r') as f:
    data = f.readlines()

# concepts are related to certain classes
# 0 = healthy, 1 = cancer, 2 = cardiomegaly, 3 = effusion, 4 = pneumonia, 5 = pneumothorax

# all_concepts = [unremarkable, mass, nodule, irregular_hilum, adenopathy, irregular_parenchyma,
#                 pneumonitis, consolidation, infection, opacities, effusion, fluid, meniscus_sign, 
#                 costophrenic_angle, enlarged_heart, absent_lung_markings, irregular_diaphragm]

concept_classes = [0, 1, 1, 1, 1, 1, 4, 4, 4, 4, 3, 3, 3, 3, 2, 5, 5]

# get filenames, original labels, and correct labels
filenames, full_report_concepts, sentence_concepts, orig_labels, present_labels = [], [], [], [], []
for i in tqdm(range(len(data))):
    line = data[i]
    
    if line.startswith('FILENAME:'):
        filename = line.split('FILENAME: ')[1].split(' ')[0]
        label = line.split('ORIGINAL LABEL: ')[1][:-1]
        filenames.append(filename)
        orig_labels.append(label)
    
    elif line.startswith('CONCEPTS:'):
        report_concs = line.split('CONCEPTS: ')[1]
        full_report_concepts.append(ast.literal_eval(report_concs))
        positives = [i for i, x in enumerate(ast.literal_eval(report_concs)) if x == 1]
        concepts = []
        for i in positives:
            if i == 0: concepts.append('Healthy')
            if i in [1,2,3,4,5]: concepts.append('Cancer')
            if i in [6,7,8]: concepts.append('Pneumonia')
            # opacities effusion unless infection present
            if i == 9 and 'Pneumonia' not in concepts:
                concepts.append('Effusion')
            if i in [10,11,12,13]: concepts.append('Effusion')
            if i == 14: concepts.append('Cardiomegaly')
            if i in [15,16,17]: concepts.append('Pneumothorax')
        
        present_labels.append(set(concepts))
        
    elif line.startswith('SENTENCE'):
        sentence_concs = line.split('SENTENCE CONCEPTS: ')[1]
        sentence_concepts.append(ast.literal_eval(sentence_concs))
        
print(len(filenames))
num_different = 0
with open('replaced_labels.txt', 'w') as f:
    for i in tqdm(range(len(filenames))):
        filename = filenames[i]
        orig_label = orig_labels[i]
        present = present_labels[i]
        changed = False
        for label in present:
            # IMPORTANT: put images in correct class folders, can be more than one
            shutil.copy(f'undersampling/images/{filename}', f'undersampling/{label}/{filename}')
            if label != orig_label:
                changed = True
        if changed:
            # keep track of images whose labels have changed
            f.write(f'{filename} -- {orig_label} -> {present}\n\n')
            if filename in os.listdir(f'undersampling/{orig_label}/'):
                os.remove(f'undersampling/{orig_label}/{filename}')
            num_different +=1

print(num_different)
