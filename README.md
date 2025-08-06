# Supporting information and reproducibility for CoMAS: Clinical Concept Manipulation for Robustness Evaluation in Chest X-ray Diagnosis

- Original report phrases contributing to each of the consolidated clinical concepts can be found in **concepts_to_phrases.png**

- Evaluation of image realism and representation of correct clinical concepts undertaken by an expert radiologist for both CaRMA (our work) and CoRPA (existing work). CoRPA was found to largely fail to represent the correct clinical concepts, while CaRMA succeeded in most cases. Radiologist notes presented in **expert_analysis_carma.xlsx** and **expert_analysis_corpa.xlsx**.

- Code associated with generating vector perturbations and adversarial reports can be found in this repository. Adversarial image generation requires use of a GPU - code is omitted here but the generative pipeline can be found at **https://huggingface.co/microsoft/radedit**
