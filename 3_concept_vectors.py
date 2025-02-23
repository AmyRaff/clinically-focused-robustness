import re
import pandas as pd
import os
from tqdm import tqdm


def clean_report(str):

    # Remove empty lines
    filtered = [a for a in str if a != " \n"]  # remove empty lines
    # Only want FINDINGS and IMPRESSION - remove everything before
    impression_idx, findings_idx = None, None
    start_idx = 0
    for line in range(len(filtered)):
        if "IMPRESSION:" in filtered[line]:
            impression_idx = line
        elif "FINDINGS:" in filtered[line] or ("CHEST" in filtered[line] and 'PA' not in filtered[line]):
            findings_idx = line
    if impression_idx is not None:
        # if impression but no findings, start at impression
        start_idx = impression_idx
    if findings_idx is not None:
        # if findings but no impression, start at findings
        if impression_idx == None:
            start_idx = findings_idx
        # if both findings and impression, start at one that appears first
        elif findings_idx < impression_idx:
            start_idx = findings_idx
    filtered = filtered[start_idx:]
    # get relevant report region as a string for output analysis
    text = ''
    for line in filtered:
        text += line
    # Regex cleaning - spaces, newlines, tabs
    out = ""
    for line in filtered:
        line = re.sub(r"\b[A-Z]+\b", " ", line)  # remove title lines (all uppercase)
        line = re.sub(r"_|:", " ", line)
        line = re.sub("\n", " ", line)
        line = re.sub(",", " ", line)
        line = line.replace("\t", " ")
        # line = line.strip()
        line = re.sub(r"\s+", " ", line)  # remove multiple spaces and tabs
        while line.startswith(" "):
            line = line[1:]
        out = out + line
    # Split into sentences, make lowercase
    out = out.split(".")
    out = [a[1:] if a.startswith(" ") else a for a in out]
    out = [a.lower() for a in out]
    # get list of all cleaned sentences for output analysis
    all_sentences = out
    out = [a for a in out if len(a) > 2]
    out = [a for a in out if len(a.split()) > 1]
    out = [a for a in out if re.search(r"[a-zA-Z]+", a)]
    # Remove unwanted lines that confuse the algorithm
    out = [a for a in out if not a.startswith("no ")]
    out = [a for a in out if not a.startswith("there is no ")]
    out = [a for a in out if not a.startswith("no evidence of")]
    out = [a for a in out if not "nipple" in a]
    out = [a for a in out if not "comparison" in a]
    out = [a for a in out if not "assess" in a]
    out = [a for a in out if not "clinical setting" in a]
    out = [a for a in out if not "not enlarged" in a]
    out = [a for a in out if not "not visible" in a]
    out = [a for a in out if not "evaluation" in a]
    out = [a for a in out if not "not evident" in a]
    out = [a for a in out if not "exclude" in a]
    out = [a for a in out if not "resolved" in a]
    # maybe - not a pathology
    out = [a for a in out if not "granuloma" in a]
    # find whether each original sentence is used for output analysis
    is_present = [True if a in out else False for a in all_sentences]
    # Handling negative mentions
    filtered_out = []
    for line in out:
        if "without" in line:
            idx = line.index("without")
            line = line[:idx]
        if " no " in line:
            idx = line.index(" no ")
            line = line[:idx]
        if "rule out" in line:
            idx = line.index("rule out")
            line = line[:idx]
        if "clear of" in line:
            idx = line.index("clear of")
            line = line[:idx]
        if "check for" in line:
            idx = line.index("check for")
            line = line[:idx]
        if "should not be mistaken for" in line:
            idx = line.index("should not be mistaken for")
            line = line[:idx]
        filtered_out.append(line)
    return filtered_out, text, all_sentences, is_present

def extract_concepts(report):
    
    occurences = [0] * len(all_concepts)
    
    for line in report:
        # concepts are either full words/lines or collections of words that can happen in any order
        for i in range(len(all_concepts)):
            cluster = all_concepts[i]
            for j in range(len(cluster)):
                concept = cluster[j]
                if all_concepts_type[i][j] == True:
                # if concept is a full word/line which just needs detecting
                    if concept in line.split(' '):
                        # print(concept)
                        occurences[i] = 1
                else:
                # if concept is collection of words in any order
                    words = concept.split(' ')
                    if 'hilus' in line:
                        line += ' hilum'
                        line += ' hilar'
                    elif 'hilum' in line:
                        line += ' hilus'
                        line += ' hilar'
                    elif 'hilar' in line:
                        line += ' hilus'
                        line += ' hilum'
                    elif 'heart' in line:
                        line += ' size'
                        line += ' cardiac'
                        line += ' shadow'
                        line += ' silhouette'
                        line += ' contour'
                    elif 'cardiac' in line and 'silhouette' in line:
                        line += ' size'
                        line += ' heart'
                        line += ' shadow'
                        line += ' contour'
                    elif 'enlarged' in line: 
                        line += ' larger'
                        line += ' enlargement'
                    num_present = 0
                    for word in words:
                        if word in line.split(' '):
                            num_present +=1
                    if num_present == len(words):
                        # print(concept)
                        occurences[i] = 1
                        
    # if any pathologies present, remove healthy concepts
    for i in range(1, len(occurences)):
        if occurences[i] == 1:
            occurences[0] = 0
    return occurences

#### Metadata
# NOTE: TOGGLE FOR FULL AND TEST

metadata = pd.read_csv("mimic-cxr-2.0.0-metadata.csv")

# image_info = pd.read_csv("undersampling/images.txt", sep=" ", header=None)
image_info = pd.read_csv("images.txt", sep=" ", header=None)
image_info.columns = ["id", "name"]

#class_info = pd.read_csv("undersampling/image_class_labels.txt", sep=" ", header=None)
class_info = pd.read_csv("image_class_labels.txt", sep=" ", header=None)
class_info.columns = ["id", "label"]

# train_info = pd.read_csv("undersampling/train_test_split.txt", sep=" ", header=None)
# train_info.columns = ["id", "is_train"]

#### Concept Lists

# Healthy
unremarkable = ['normal', 'unremarkable', 'lungs clear', 'no evidence', 'no interval change',
                  'no acute cardiopulmonary abnormality', 'normal hilar contours', 'no acute process']
unremarkable_full = [True, True, False, True, False, False, False, False]

# Cancer
mass = ['mass', 'cavitary lesion', 'carcinoma', 'neoplasm', 'tumor', 'tumour', 
                   'rounded opacity', 'lung cancer', 'apical opacity', 'lump', 'triangular opacity',
                   'malignant', 'malignancy']
mass_full = [True, False, True, True, True, True, False, True, True, True, True, True, True]
nodule = ['nodular density', 'nodular densities', 'nodular opacity', 'nodular opacities',
          'nodular opacification', 'nodule']
nodule_full = [True, True, True, True, True, True]
irregular_hilum = ['hilar mass', 'hilar opacity', 'hilus enlarged', 'hilus fullness', 
                   'hilus bulbous']
irregular_hilum_full = [True, True, False, False, False]
adenopathy = ['hilar adenopathy', 'hilar lymphadenopathy', 'mediastinal lymphadenopathy', 
              'mediastinal adenopathy']
adenopathy_full = [True, True, True, True]
irregular_parenchyma = ['pulmonary metastasis', 'carcinomatosis', 'metastatic disease']
irregular_parenchyma_full = [True, True, True]

# Pneumonia
pneumonitis = ['pneumonia', 'pneumonitis', 'bronchopneumonia', 'airspace disease', 
               'air bronchograms', 'cavitation']
pneumonitis_full = [True] * 6
consolidation = ['consolidation']
consolidation_full = [True]
infection = ['infection', 'infectious process', 'infectious']
infection_full = [True, True, True]
opacities = ['airspace opacities', 'airspace opacity', 'homogeneous opacity', 'homogeneous opacities',
             'patchy opacities', 'patchy opacity', 'ground-glass opacities', 'ground-glass opacity',
             'alveolar opacities', 'alveolar opacity', 'ill-defined opacities', 
             'reticulonodular pattern']
opacities_full = [True] * 12

# Pleural Effusion
effusion = ['effusion', 'effusions', 'pleural effusion']
effusion_full = [True, True, True]
fluid = ['pleural fluid', 'fluid collection', 'layering fluid']
fluid_full = [True, True, False]
meniscus_sign = ['meniscus sign']
meniscus_sign_full = [True]
costophrenic_angle = ['costophrenic angle blunting']
costophrenic_angle_full = [False]
opacities = opacities # NOTE: Handled in next script

# Cardiomegaly
enlarged_heart = ['cardiomegaly', 'borderline cardiac silhouette', 'heart enlarged', 
                  'prominent cardiac silhouette', 'top-normal heart']
enlarged_heart_full = [True, False, False, False, False]

# Pneumothorax
absent_lung_markings = ['apical pneumothorax', 'basilar pneumothorax', 'hydro pneumothorax', 
                  'hydropneumothorax', 'lateral pneumothorax', 'pneumothorax', 'pneumothoraces',
                  'absent lung markings']
absent_lung_full = [True, True, True, True, True, True, True, False]
irregular_diaphragm = ['flattening ipsilateral diaphragm', 'inversion ipsilateral diaphragm']
diaphragm_full = [False, False]

all_concepts = [unremarkable, mass, nodule, irregular_hilum, adenopathy, irregular_parenchyma,
                pneumonitis, consolidation, infection, opacities, effusion, fluid, meniscus_sign, 
                costophrenic_angle, enlarged_heart, absent_lung_markings, irregular_diaphragm]
all_concepts_type = [unremarkable_full, mass_full, nodule_full, irregular_hilum_full,
                     adenopathy_full, irregular_parenchyma_full, pneumonitis_full,
                     consolidation_full, infection_full, opacities_full, effusion_full,
                     fluid_full, meniscus_sign_full, costophrenic_angle_full, enlarged_heart_full,
                     absent_lung_full, diaphragm_full]

assert len(all_concepts) == len(all_concepts_type)
for i in range(len(all_concepts)):
    assert len(all_concepts[i]) == len(all_concepts_type[i])
    
###  Get reports and ORIGINAL labels
reports, labels, filenames = [], [], []

training = 0
# for i in tqdm(range(len(train_info['id']))):
for i in tqdm(range(len(set(class_info['id'])))): # NOTE: set
    # get test set - NOTE: TOFFLE FOR FULL AND TEST
    # id = train_info['id'][i]
    id = class_info['id'][i]
    # is_train = train_info['is_train'][i]
    is_train = 0
    image = image_info[image_info['id'] == id]['name'].values[0]
    if is_train == 0 and image in os.listdir('undersampling/images/'):
        training +=1
        filenames.append(image)
        # duplication handling
        if "_" not in image:
            filename = image.split(".")[0]
        else:
            filename = image.split("_")[1].split(".")[0]
        data = metadata[metadata["dicom_id"] == filename]
        subject = str(data["subject_id"].values[0])
        study = str(data["study_id"].values[0])
        dir = "p" + subject[:2]
        path = (
            "../concept_bottleneck_clustered/1 - Concept Extraction/files/"
            + dir
            + "/p"
            + subject
            + "/s"
            + study
            + ".txt"
        )
        reports.append(path)
        # get ground truth label
        id = image_info[image_info["name"] == filename + '.jpg']["id"].values[0]
        label = class_info[class_info["id"] == id]["label"].values[0]
        labels.append(label)


### Write to output file
with open('concept_representations.txt', 'w') as f:
    written, with_zero = 0, 0
    for i in tqdm(range(len(reports))):
        file, ground_label, name = reports[i], labels[i], filenames[i]
        info = open(file, "r")
        info = info.readlines()
        
        str_report, orig_text, all_sentences, is_present = clean_report(info)

        # get concepts for whole report
        report_concepts = extract_concepts(str_report)
        # get concepts for each sentence of report
        sentence_concepts = []
        curr_sentence_idx = 0
        for i in range(len(all_sentences)):
            if is_present[i] == True: 
                out = extract_concepts([str_report[curr_sentence_idx]])
                sentence_concepts.append(out)
                curr_sentence_idx+=1
            
        
        # dont want instances with no concepts
        with_zero +=1
        if report_concepts != [0] * len(report_concepts):
            
            # get total of sentences and check same as original report
            total = [sum(i) for i in zip(*sentence_concepts)]
            total = [1 if i > 0 else 0 for i in total]
            for i in range(1, len(total)):
                if total[i] == 1:
                    total[0] = 0
            # print(report_concepts)
            # print(sentence_concepts)
            assert total == report_concepts

            # write to file for analysis
            f.write('-------------------------------\n')
            f.write('-------------------------------\n\n')
            f.write(str(orig_text))
            f.write(f'\nFILENAME: {name} ORIGINAL LABEL: {ground_label}\n')
            f.write(f'CONCEPTS: {report_concepts}\n')
            f.write(f'SENTENCE CONCEPTS: {sentence_concepts}\n\n')
            written +=1

print(written) # out of 4100
print(with_zero)

# 34532