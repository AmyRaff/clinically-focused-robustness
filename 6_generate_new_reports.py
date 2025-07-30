import ast
from tqdm import tqdm
import re
import random

random.seed(2)

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
                        
    # NOTE: REMOVE - if any pathologies present, remove healthy concepts
    #for i in range(1, len(occurences)):
    #    if occurences[i] == 1:
    #        occurences[0] = 0
    return occurences

def clean_report(str):

    # Remove empty lines
    filtered = [a + '.' for a in str if a != " \n"]  # remove empty lines
    # Only want FINDINGS and IMPRESSION - remove everything before
    impression_idx, findings_idx = None, None
    start_idx = 0
    for line in range(len(filtered)):
        if "IMPRESSION:" in filtered[line]:
            impression_idx = line
        elif "FINDINGS:" in filtered[line] or "CHEST" in filtered[line]:
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

# get concept perturbations
with open('aaa.txt', 'r') as f:
    perturbation_data = f.readlines()
    
filenames, correct_labels, perturbations = [], [], []
for line in perturbation_data: # TODO: TOGGLE
    if len(line) > 3:
        filenames.append(line.split(',')[0][2:-1])
        correct_labels.append(line.split(',')[0][2:-1].split('/')[0])
        #print(ast.literal_eval(''.join(str(line.split(',')[2:])).replace("', '", ',')[3:-5]))
        #print(line.split(',')[1][2:-1])
        perturbations.append((line.split(',')[1][2:-1], ast.literal_eval(''.join(str(line.split(',')[2:])).replace("', '", ',')[3:-5])))
        
assert len(filenames) == len(perturbations) == len(correct_labels)
print(len(filenames)) # 40997

# get original representations
# with open('concept_representations.txt', 'r') as f:
#     original_data = f.readlines()
    
# NOTE: lists will be shorter as this is before class duplication 
# orig_filenames, full_report_concepts, sentence_concepts, orig_labels, present_labels, orig_reports = [], [], [], [], [], []
# for i in tqdm(range(len(original_data))):
#     line = original_data[i]
#     report_str = ''
    
#     if line != '-------------------------------\n':
    
#         if line.startswith('FILENAME:'):
#             filename = line.split('FILENAME: ')[1].split(' ')[0]
#             label = line.split('ORIGINAL LABEL: ')[1][:-1]
#             orig_filenames.append(filename)
#             orig_labels.append(label)
        
#         elif line.startswith('CONCEPTS:'):
#             report_concs = line.split('CONCEPTS: ')[1]
#             full_report_concepts.append(ast.literal_eval(report_concs))        
#             present_labels.append(check_label(ast.literal_eval(report_concs)))
            
#         elif line.startswith('SENTENCE'):
#             sentence_concs = line.split('SENTENCE CONCEPTS: ')[1]
#             sentence_concepts.append(ast.literal_eval(sentence_concs))
            
#         else:
#             if len(line) > 1:
#                 report_str += line
#             if len(report_str) > 1:
#                 orig_reports.append(report_str)
                
import re

with open('concept_representations.txt', 'r') as f:
    text = f.read()

# Split by the report delimiter
entries = [e.strip() for e in text.split('-------------------------------') if e.strip()]

orig_reports, orig_filenames, orig_labels, present_labels, full_report_concepts, sentence_concepts = [], [], [], [], [], []

for entry in entries:
    # Extract report text (everything before 'FILENAME:')
    report_match = re.split(r'FILENAME:', entry, maxsplit=1)
    if len(report_match) < 2:
        continue
    report_text = report_match[0].strip()
    orig_reports.append(report_text)

    # Extract filename
    filename_match = re.search(r'FILENAME:\s*(\S+)', entry)
    orig_filenames.append(filename_match.group(1) if filename_match else None)

    # Extract label
    label_match = re.search(r'ORIGINAL LABEL:\s*(\w+)', entry)
    orig_labels.append(label_match.group(1) if label_match else None)

    # Extract concepts list
    concepts_match = re.search(r'CONCEPTS:\s*\[(.*?)\]', entry)
    if concepts_match:
        full_report_concepts.append([int(x.strip()) for x in concepts_match.group(1).split(',')])
        present_labels.append(check_label([int(x.strip()) for x in concepts_match.group(1).split(',')]))
    else:
        full_report_concepts.append([])
        present_labels.append([])

    # Extract sentence concepts (list of lists)
    sentence_concepts_match = re.search(r'SENTENCE CONCEPTS:\s*\[(.*)\]', entry, re.DOTALL)
    if sentence_concepts_match:
        sent_concepts_str = sentence_concepts_match.group(1)
        sentence_concepts.append(eval('[' + sent_concepts_str.strip() + ']'))
    else:
        sentence_concepts.append([])

# format reports
print('Formatting reports...')
# breaks = [i for i, x in enumerate(orig_reports) if x == "-------------------------------\n"]
# reports = []
# for r in tqdm(range(len(orig_reports))):
#     if r in breaks and r + 1 in breaks:
#         report_start_idx = r + 2
#         if breaks.index(r + 1) + 1 < len(breaks):
#             report_end_idx = breaks[breaks.index(r + 1) + 1]
#             report = orig_reports[report_start_idx:report_end_idx]
#         else:
#             report = orig_reports[report_start_idx:]
#         report_str = ''
#         for line in report:
#             report_str += line
#         reports.append(report_str)
        
# breaks = [i for i, x in enumerate(orig_reports) if x == "-------------------------------\n"]
# print(len(orig_reports))
# reports = [
#     "".join(orig_reports[start + 2:end])
#     for start, end in zip(breaks, breaks[1:] + [len(orig_reports)])
# ]

print(len(orig_filenames))
assert len(orig_reports) == len(orig_filenames) 

all_correct_labels = []
# link representations to perturbations
fucked = 0
print('Linking data...')
# all_full_report_concs, all_sentence_concs, all_orig_labels, all_reports = [], [], [], []
# for f in tqdm(range(len(filenames))):
#     file = filenames[f]
#     correct_label = correct_labels[f]
#     all_correct_labels.append(correct_label)
#     perturbs = perturbations[f]
#     for i in range(len(orig_filenames)):
#         if file == orig_filenames[i]:
#             idx = i
#     report = reports[idx]
#     all_reports.append(report)
#     sentence_concs = sentence_concepts[idx]
#     all_sentence_concs.append(sentence_concs)
#     report_concs = full_report_concepts[idx]
#     all_full_report_concs.append(report_concs)
#     orig_label = orig_labels[idx]
#     all_orig_labels.append(orig_label)
#     if correct_label not in present_labels[idx]:
#         print(file)
#     assert correct_label in present_labels[idx]

index_map = {fname: i for i, fname in enumerate(orig_filenames)}

all_full_report_concs, all_sentence_concs, all_orig_labels, all_reports = [], [], [], []
for f, file in enumerate(tqdm(filenames)):
    idx = index_map[file.split('/')[1]]
    correct_label = correct_labels[f]
    all_correct_labels.append(correct_label)

    all_reports.append(orig_reports[idx])
    all_sentence_concs.append(sentence_concepts[idx])
    all_full_report_concs.append(full_report_concepts[idx])
    all_orig_labels.append(orig_labels[idx])

    #if correct_label not in present_labels[idx]:
    #    print(file)
    #    print(correct_label)
    #    print(present_labels[idx])
        
    #assert correct_label in present_labels[idx]

# OTE: deleted 40500 from text files
assert len(filenames) == len(all_sentence_concs) == len(all_reports)

# filenames, perturbations, correct_labels, all_reports, all_sentence_concs, 
# all_full_report_concs, all_orig_labels


# Get mapping of sentences to concept vectors - 5 sentences per vector
print('Generating mapping...')
# def generate_mapping():
#     sentences, concepts = [], []
#     concept_counts = {}
#     # randomly sample reports NOTE
#     random_idxs = set(random.sample(range(len(all_reports) - 1), int(len(all_reports) - 2)))
#     for i in tqdm(random_idxs):
#         # get relevant report sentences
#         cleaned_report, _, _, _ = clean_report(all_reports[i].split('.'))
#         for j in range(len(cleaned_report)):
#             sentence = cleaned_report[j]
#             concs = all_sentence_concs[i][j]
#             if concs != [0]*17 and sentence not in sentences:
#                 if str(concs) in concept_counts.keys():
#                     if concept_counts[str(concs)] < 5:
#                         concept_counts[str(concs)] +=1
#                         concepts.append(concs)
#                         sentences.append(sentence)
#                 else:
#                     concept_counts[str(concs)] = 1
#                     concepts.append(concs)
#                     sentences.append(sentence)
        
#     # get dict of sentences and concepts for next step
#     sentences_to_concepts = dict.fromkeys(sentences)
#     sentences_to_concepts.update(zip(sentences_to_concepts, concepts))

#     return sentences_to_concepts

def generate_mapping():
    concept_counts = {}
    sentences_to_concepts = {}

    for i in tqdm(random.sample(range(len(all_reports) - 1), len(all_reports) - 2)):
        cleaned_report, _, _, _ = clean_report(all_reports[i].split('.'))

        for sentence, concs in zip(cleaned_report, all_sentence_concs[i]):
            if concs == [0] * 17 or sentence in sentences_to_concepts:
                continue

            key = tuple(concs)
            if concept_counts.get(key, 0) < 5:
                concept_counts[key] = concept_counts.get(key, 0) + 1
                sentences_to_concepts[sentence] = concs

    return sentences_to_concepts



# TODO: when using smaller testing set, keep original mapping
import pickle
#pickle.dump(generate_mapping(), open('sentence_concepts.pkl', 'wb'))

sentences_to_concepts = pickle.load(open('sentence_concepts.pkl', 'rb'))
print(type(sentences_to_concepts))

##### Replacement Algorithm

print('Generating adversarial reports...')
adversarial_reports = []
adv_filenames = []
new_labels = []
failed = {}
report_map = []
worked = 0
#for r in tqdm(range(20)):
for r in tqdm(range(len(all_reports))):
    test_report = all_reports[r]
    orig_vector = all_full_report_concs[r]
    sentence_vectors = [a for a in all_sentence_concs[r] if a != [0] * 17]
    type, target = perturbations[r]

    # for each perturbation of each report
    # NOTE: to ensure can capture correct vectors, generate 20 perturbations in previous file
    # save first 10 that work here
    # ^^^^^^ ignore for now
    num = 6 # NOTE
    num_in, num_fp, num_fn = 0, 0, 0
    fail = 0
    success = []
    curr_adv_reports = []
    curr_goal_labels = []
    for i in range(1):
        if len(success) < num:
            goal = target
            new_label = check_label(goal)
            go_ahead = False
            if new_label != ['Healthy'] and new_label != ['Healthy']:
                if i < 3:
                    if num_in < 2:
                        go_ahead = True
                        num_in +=1
                elif 3 <= i < 13:
                    if num_fp < 2:
                        go_ahead = True
                        num_fp +=1
                else:
                    if num_fn < 2:
                        go_ahead = True
                        num_fn+=1
            else:
                if i < 11:
                    if num_fp < 2:
                        go_ahead = True
                        num_fp +=1
                else:
                    if num_fn < 2:
                        go_ahead = True
                        num_fn+=1
            if go_ahead:
                current = orig_vector
                # get sentences that correspond to sentence vectors
                cleaned_report, _, _, _ = clean_report(test_report.split('\n'))#.split('.'))
                # get original report sentences in lowercase
                orig_sentences = test_report.split('.')
                # pick out and delete sentences which have concepts we want to remove
                filtered_sentences = orig_sentences
                for j in range(len(sentence_vectors)):
                    if sentence_vectors[j] == current:
                        sentence_to_delete = cleaned_report[j]
                        for k in range(len(orig_sentences) - 1):
                            if k < len(orig_sentences):
                                if sentence_to_delete in orig_sentences[k].lower():
                                    filtered_sentences.remove(orig_sentences[k])
                # add sentences with concepts we want - randomly select from list
                import numpy as np
                diff = list(np.subtract(goal, current))
                replacements = [a for a,b in sentences_to_concepts.items() if b == diff]
                successful = True
                if len(replacements) > 0:
                    # if goal already exists in mapping, use
                    replacement = random.choice(replacements)
                    filtered_sentences.insert(-2, replacement)
                    worked +=1
                else:
                    # otherwise need more than one sentence
                    needed = [z for z, x in enumerate(diff) if x == 1]
                    separate_vectors = []
                    for n in needed:
                        vec = [0] * len(goal)
                        vec[n] = 1
                        separate_vectors.append(vec)
                    for vector in separate_vectors:
                        replacements = [a for a,b in sentences_to_concepts.items() if b == vector]
                        if len(replacements) < 1:
                            successful = False
                        else:
                            replacement = random.choice(replacements)
                            filtered_sentences.insert(-2, replacement)
                            worked +=1
                # if fails, skip to next perturbation
                if successful:
                    # check new report has desired concept vector
                    filtered_sentences = [a.lower() + '. ' for a in filtered_sentences]
                    cleaned_filtered, _, _, _ = clean_report(filtered_sentences)
                    new_vector = extract_concepts(cleaned_filtered)
                    if new_vector == goal:
                        # get formatted adversarial report for next step
                        #filtered_report = 'pa chest xray. ordinary spine and 12 rib pairs. '
                        filtered_report = ''
                        #if 'Cardiomegaly' not in new_label:
                        #    filtered_report += 'ordinary human heart. '
                        for line in filtered_sentences:
                            filtered_report += line
                            filtered_report += '. '
                        adversarial_reports.append(filtered_report)
                        success.append(filtered_report)
                        new_labels.append(new_label)
                        curr_goal_labels.append(new_label)
                        adv_filenames.append(filenames[r])
                        curr_adv_reports.append(filtered_report)
                        
    if len(success) < num:
        fail +=1
    if len(success) != num:
        failed[filenames[r]] = len(success)
        # NOTE: if no perturbations, use original report 10 times
        if len(success) == 0:
            for fails in range(num):
                success.append(test_report)
                adversarial_reports.append(test_report)
                new_labels.append(check_label(orig_vector))
                curr_goal_labels.append(check_label(orig_vector))
                adv_filenames.append(filenames[r])
                curr_adv_reports.append(test_report)
        # NOTE: if less than 10, duplicate reports randomly
        else:
            for fails in range(num - len(success)):
                duplicate = random.choice(success)
                idx = success.index(duplicate)
                success.append(duplicate)
                adversarial_reports.append(duplicate)
                new_labels.append(curr_goal_labels[idx])
                adv_filenames.append(filenames[r])
                curr_adv_reports.append(duplicate)
    assert len(success) == num
    #print(len(set(adversarial_reports)))
    for rep in set(curr_adv_reports):
        if rep != test_report:
            report_map.append((filenames[r], test_report, type, rep))
    # with open('check.txt', 'a') as f:
    #     f.write(test_report)
    #     f.write('\n-----------------------\n')
    #     f.write(filenames[r])
    #     f.write('\n-----------------------\n')
    #     for rep in set(adversarial_reports):
    #         f.write(rep)
    #         f.write('\n')
    #     f.write('-----------------------\n')
    #     f.write('-----------------------\n\n')

# def generate_adversarial_reports():
#     adversarial_reports, new_labels, failed = [], [], {}
#     worked = 0
#     num = 6

#     for r, test_report in enumerate(tqdm(all_reports)):
#         orig_vector = all_full_report_concs[r]
#         sentence_vectors = [a for a in all_sentence_concs[r] if a != [0] * 17]
#         targets = perturbations[r]

#         success, curr_goal_labels = [], []
#         counts = {"in": 0, "fp": 0, "fn": 0}

#         for i, goal in enumerate(targets):
#             if len(success) >= num:
#                 break

#             new_label = check_label(goal)
#             go_ahead = (
#                 (new_label != ['Healthy'] and ((i < 3 and counts["in"] < 2) or (3 <= i < 13 and counts["fp"] < 2) or (counts["fn"] < 2)))
#                 or (new_label == ['Healthy'] and ((i < 11 and counts["fp"] < 2) or counts["fn"] < 2))
#             )

#             if not go_ahead:
#                 continue

#             if i < 3:
#                 counts["in"] += 1
#             elif i < 13:
#                 counts["fp"] += 1
#             else:
#                 counts["fn"] += 1

#             cleaned_report, _, _, _ = clean_report(test_report.split('\n'))
#             orig_sentences = test_report.split('.')

#             filtered_sentences = [s for s in orig_sentences]
#             current = orig_vector

#             for j, vec in enumerate(sentence_vectors):
#                 if vec == current:
#                     sentence_to_delete = cleaned_report[j]
#                     filtered_sentences = [s for s in filtered_sentences if sentence_to_delete not in s.lower()]

#             diff = [a - b for a, b in zip(goal, current)]
#             needed_vectors = [[(1 if i == idx else 0) for i in range(len(goal))] for idx, v in enumerate(diff) if v == 1]

#             successful = True
#             for vector in ([diff] if tuple(diff) in sentences_to_concepts.values() else needed_vectors):
#                 replacements = [s for s, v in sentences_to_concepts.items() if v == vector]
#                 if not replacements:
#                     successful = False
#                     break
#                 replacement = random.choice(replacements)
#                 filtered_sentences.insert(-2, replacement)
#                 worked += 1

#             if not successful:
#                 continue

#             filtered_sentences = [s.lower() + '. ' for s in filtered_sentences]
#             cleaned_filtered, _, _, _ = clean_report(filtered_sentences)

#             if extract_concepts(cleaned_filtered) == goal:
#                 report_text = ''.join(s + '. ' for s in filtered_sentences)
#                 adversarial_reports.append(report_text)
#                 success.append(report_text)
#                 new_labels.append(new_label)
#                 curr_goal_labels.append(new_label)

#         if len(success) < num:
#             failed[filenames[r]] = len(success)
#             while len(success) < num:
#                 duplicate = random.choice(success or [test_report])
#                 label = curr_goal_labels[success.index(duplicate)] if success else check_label(orig_vector)
#                 success.append(duplicate)
#                 adversarial_reports.append(duplicate)
#                 new_labels.append(label)

#         assert len(success) == num

#     return adversarial_reports, new_labels, failed, worked


#adversarial_reports, new_labels, failed, worked = generate_adversarial_reports()
# NOTE: should be 10 * len(all_reports) --- use to link later
print(len(adversarial_reports))
print(f'Unique: {len(set(adversarial_reports))}')
print(len(all_reports))
print(len(adv_filenames))
print(len(new_labels))
assert len(adversarial_reports) == 6 * len(all_reports)
assert len(adversarial_reports) == len(adv_filenames)

adv_orig_labels = []
for l in all_correct_labels:
    for i in range(6): # NOTE
        adv_orig_labels.append(l)
print(len(adv_orig_labels))

pickle.dump(report_map, open('report_map.pkl', 'wb'))

pickle.dump(adversarial_reports, open('mimic_six_adversarial_reports.pkl', 'wb'))
pickle.dump(new_labels, open('six_adv_report_new_labels.pkl', 'wb'))
pickle.dump(adv_orig_labels, open('six_adv_report_orig_labels.pkl', 'wb'))
pickle.dump(adv_filenames, open('mimic_six_adv_filenames.pkl', 'wb'))