import ast
from tqdm import tqdm 
import os
import random
import pandas as pd

random.seed(2)

def check_label(concs):

    positives = [i for i, x in enumerate(concs) if x == 1]
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
    return list(set(concepts))

with open('concept_representations.txt', 'r') as f:
    data = f.readlines()

# concepts are related to certain classes
# 0 = healthy, 1 = cardiomegaly, 2 = cancer, 3 = effusion, 4 = pneumonia, 5 = pneumothorax

# all_concepts = [unremarkable, mass, nodule, irregular_hilum, adenopathy, irregular_parenchyma,
#                 pneumonitis, consolidation, infection, opacities, effusion, fluid, meniscus_sign, 
#                 costophrenic_angle, enlarged_heart, absent_lung_markings, irregular_diaphragm]

all_labels = ['Healthy', 'Cancer', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax']
concept_classes = [0, 1, 1, 1, 1, 1, 4, 4, 4, 4, 3, 3, 3, 3, 2, 5, 5]

image_info = pd.read_csv("undersampling/images.txt", sep=" ", header=None)
image_info.columns = ["id", "name"]

train_info = pd.read_csv("undersampling/train_test_split.txt", sep=" ", header=None)
train_info.columns = ["id", "is_train"]

class_info = pd.read_csv("undersampling/image_class_labels.txt", sep=" ", header=None)
class_info.columns = ["id", "label"]

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
        present_labels.append(check_label(ast.literal_eval(report_concs)))
        
    elif line.startswith('SENTENCE'):
        sentence_concs = line.split('SENTENCE CONCEPTS: ')[1]
        sentence_concepts.append(ast.literal_eval(sentence_concs))
        
print(len(filenames)) # 34532

def get_inter_class_perturbs(label, num):
    label_indices = []
    for id in train_info['id']:
        im = image_info[image_info['id'] == id]['name'].values[0]
        cls = class_info[class_info['id'] == id]['label'].values[0]
        # filter only for images of the current class that are in the test set
        if all_labels[cls] == label:
            is_train = train_info[train_info['id'] == id]['is_train'].values[0]
            if im in filenames and is_train == 0:
                idx = filenames.index(im)
                label_indices.append(idx)
            else:
                label_indices.append(None)
    
    used_filenames, all_perturbs = [], []
    for i in label_indices:
        if i is not None:
            filename = filenames[i]
            used_filenames.append(f'{label}/{filename}')
            report_concs = full_report_concepts[i]
            correct_labels = present_labels[i]
            #print(f'Perturbing for {label}')
            # if class only has one concept, cannot do this perturbation.
            indices = get_label_indices(label)
            original = report_concs
            inter_class_perturbs = []
            # get n random perturbations of the class concepts
            while len(inter_class_perturbs) < num:
                a = [random.getrandbits(1) for a in range(len(indices))]
                # NOTE: don't want too many positives
                if a not in inter_class_perturbs and a != [0] * len(indices) and sum(a) < 3:
                    concs = original.copy()
                    idx = 0
                    for c in range(len(concs)):
                        if c in indices:
                            concs[c] = a[idx]
                            idx+=1
                    if concs != original and concs not in inter_class_perturbs:
                        inter_class_perturbs.append(concs)
            
            # print(f'Inter: {inter_class_perturbs}')
            # check perturbations are of the same class as original
            for perturbation in inter_class_perturbs:
                for l in correct_labels:
                    assert l in check_label(perturbation)
                # assert correct_labels == check_label(perturbation)
            
            all_perturbs.append(inter_class_perturbs)
        
    return all_perturbs, used_filenames

def get_label_indices(label):
    if label == 'Healthy': 
        indices = [0]
    elif label == 'Cardiomegaly':
        indices = [14]
    elif label == 'Cancer':
        indices = [1, 2, 3, 4, 5]# NOTE!!!
    elif label == 'Pneumothorax':
        indices = [15, 16]
    elif label == 'Pneumonia':
        indices = [6, 7, 8]
    elif label == 'Effusion':
        # opacities (9) effusion unless infection present - ignore
        indices = [10, 11, 12, 13]
    return indices

def get_fp_perturbs(label, num):
    label_indices = []
    for id in train_info['id']:
        im = image_info[image_info['id'] == id]['name'].values[0]
        cls = class_info[class_info['id'] == id]['label'].values[0]
        # filter only for images of the current class that are in the test set
        if all_labels[cls] == label:
            is_train = train_info[train_info['id'] == id]['is_train'].values[0]
            if im in filenames and is_train == 0:
                idx = filenames.index(im)
                label_indices.append(idx)
            else:
                label_indices.append(None)
    
    used_filenames, all_perturbs = [], []

    for i in label_indices:
        if i is not None:
            filename = filenames[i]
            report_concs = full_report_concepts[i]
            correct_labels = present_labels[i]
            if len(correct_labels) > 1: # NOTE
                continue
            # num_perturbations = 2 # NOTE: num
            original = report_concs
            outer_class_perturbs = []
            new_labels = []
            # get n random perturbations of the concepts of another class
            while len(outer_class_perturbs) < num:
                new_label = random.choice(all_labels)
                if new_label != label:
                    original_indices = get_label_indices(label)
                    indices = get_label_indices(new_label)
                    a = [random.getrandbits(1) for a in range(len(indices))]
                    # NOTE: dont want too many positive
                    if a != [0] * len(indices) and sum(a) < 3:
                        concs = original.copy()
                        # NOTE: REMOVE - set concepts for original class to 0
                        #for c in range(len(concs)):
                        #    if c in original_indices:
                        #        concs[c] = 0
                        for i in indices:
                            # check no instances of new class are already positive
                            if concs[i] == 1:
                                continue 
                        # update concepts for new class
                        idx = 0
                        for c in range(len(concs)):
                            if c in indices:
                                concs[c] = a[idx]
                                idx+=1
                        if concs != original and concs not in outer_class_perturbs:
                            new_labels.append(new_label)
                            outer_class_perturbs.append(concs)
            used_filenames.append(f'{label}/{filename}')
            assert len(outer_class_perturbs) == num
            all_perturbs.append(outer_class_perturbs)
            # check perturbations belong to the correct random classes
            for i in range(len(outer_class_perturbs)):
                assert new_labels[i] in check_label(outer_class_perturbs[i])
        
    return all_perturbs, used_filenames


def get_fn_perturbs(label, num):
    label_indices = []
    for id in train_info['id']:
        im = image_info[image_info['id'] == id]['name'].values[0]
        cls = class_info[class_info['id'] == id]['label'].values[0]
        # filter only for images of the current class that are in the test set
        if all_labels[cls] == label:
            is_train = train_info[train_info['id'] == id]['is_train'].values[0]
            if im in filenames and is_train == 0:
                idx = filenames.index(im)
                label_indices.append(idx)
            else:
                label_indices.append(None)
    
    used_filenames, all_perturbs = [], []

    for i in label_indices:
        if i is not None:
            filename = filenames[i]
            report_concs = full_report_concepts[i]
            correct_labels = present_labels[i]
            if len(correct_labels) < 2: # NOTE
                continue
            # num_perturbations = 2 # NOTE: num
            original = report_concs
            outer_class_perturbs = []
            rem_labels = []
            # get n random perturbations of the concepts of another class
            while len(outer_class_perturbs) < num:
                # randomly select a single true label
                single_label = random.choice(correct_labels)
                rem_labels.append(single_label)
                #if new_label != label:
                indices = get_label_indices(single_label)
                #a = [random.getrandbits(1) for a in range(len(indices))]
                # NOTE: dont want too many positive
                #if a != [0] * len(indices) and sum(a) < 3:
                concs = original.copy()
                # set everything for single label to 0
                for c in range(len(concs)):
                    if c in indices:
                        concs[c] = 0
                outer_class_perturbs.append(concs)
            used_filenames.append(f'{label}/{filename}')
            assert len(outer_class_perturbs) == num
            all_perturbs.append(outer_class_perturbs)
            # check perturbations belong to the correct random classes
            for i in range(len(outer_class_perturbs)):
                #print(rem_labels[i])
                #print(outer_class_perturbs[i])
                #print(check_label(outer_class_perturbs[i]))
                assert rem_labels[i] not in check_label(outer_class_perturbs[i])
        
    return all_perturbs, used_filenames
            
                    
every_file, every_perturbation = [], []
for l in tqdm(all_labels):
    print(l)
    # NOTE: need 6, do more here so can capture later
    # num_perturbations = 10
    # print('Generating inter-class perturbations...')
    if l != 'Healthy' and l != 'Cardiomegaly':
        print('Generating inter-class perturbations...')
        inner_perturbs, inner_files = get_inter_class_perturbs(l, 2)
        print('Generating fp perturbations...')
        fp_perturbs, fp_files = get_fp_perturbs(l, 10)
        print('Generating fn perturbations...')
        fn_perturbs, fn_files = get_fn_perturbs(l, 10)
        
    else:
        inner_files, inner_perturbs = None, None
        print('Generating fp perturbations...')
        fp_perturbs, fp_files = get_fp_perturbs(l, 11)
        print('Generating fn perturbations...')
        fn_perturbs, fn_files = get_fn_perturbs(l, 11)
    
    
    print('Merging and writing perturbations...')
    pairs = []
    if inner_perturbs is not None:
        for i in range(len(inner_perturbs)):
            pairs.append((inner_files[i], inner_perturbs[i]))
    for i in range(len(fp_perturbs)):
        pairs.append((fp_files[i], fp_perturbs[i]))
    for i in range(len(fn_perturbs)):
        pairs.append((fn_files[i], fn_perturbs[i]))
    
    #print(pairs)
    #pairs = set(pairs)
    all_files, all_perturbs = [], []
    for file, perturb in pairs:
        perturbs = []
        for p in range(len(pairs)):
            if pairs[p][0] == file:
                curr = pairs[p][1]
                for c in curr:
                    perturbs.append(c)
        all_files.append(file)
        all_perturbs.append(perturbs)
    
    # if inner_perturbs is not None:
    #     assert inner_files == fp_files

    #     all_perturbations = []
    #     for i in range(len(inner_files)):
    #         perturbations = []
    #         for inner in inner_perturbs[i]:
    #             perturbations.append(inner)
    #         for fp in fp_perturbs[i]:
    #             perturbations.append(fp)
    #         #assert len(perturbations) == 20 # NOTE
    #         all_perturbations.append(perturbations)
    # else:
    #     all_perturbations = fp_perturbs
    
    # print(len(all_files))
    assert len(all_perturbs) == len(all_files)
    for file in all_files:
        every_file.append(file)
    if all_perturbs is not None:
        for perturb in all_perturbs:
            every_perturbation.append(perturb)
    
print(len(every_perturbation))
with open('aaa.txt', 'w') as f:
    for i in range(len(every_perturbation)):
        f.write(f'{every_file[i]}\n')
        f.write(f'{every_perturbation[i]}\n\n')
