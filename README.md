# EpiModX


## 🚀 Abstract

Epigenetic modifications play a vital role in the pathogenesis of human diseases, particularly neurodegenerative disorders such as Alzheimer's disease (AD), where dysregulated histone modifications are strongly implicated in disease mechanisms. While recent advances underscore the importance of accurately identifying these modifications to elucidate their contribution to AD pathology, existing computational methods remain limited by their generic approaches that overlook disease-specific epigenetic signatures. To bridge this gap, we developed a novel large language model (LLM)-based deep learning framework tailored for disease-contextual prediction of histone modifications and variant effects. Focusing on AD as a case study, we integrated epigenomic data from multiple patient samples to construct a comprehensive, disease-specific histone modification dataset, enabling our model to learn AD-associated molecular signatures. A key innovation of our approach is the incorporation of a Mixture of Experts (MoE) architecture, which effectively distinguishes between disease and healthy epigenetic states, allowing for precise identification of AD-relevant epigenetic modification patterns. Our model demonstrates robust performance in disease-specific histone modification prediction, achieving mean area under receiver-operating curves (AUROCs) ranging from 0.8170 to 0.9142, significantly outperforming existing state-of-the-art methods that lack disease context. Beyond accurate modification site prediction, our framework provides important biological insights by successfully prioritizing AD-associated genetic variants, which show significant enrichment in disease-relevant pathways, supporting their potential functional role in AD pathogenesis. These findings suggest that differential modification loci identified by our model may represent key regulatory elements in AD. Our framework establishes a powerful new paradigm for epigenetic research that can be extended to other complex diseases, offering both a valuable tool for variant effect interpretation and a promising strategy for uncovering novel disease mechanisms through epigenetic profiling.

![Framework](https://github.com/user-attachments/assets/9bca6eb1-86b8-4337-82f2-3409780119e5)


## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/project-name.git
cd project-name


pip install -r requirements.txt

conda create -n project-env python=3.10
conda activate project-env
pip install -r requirements.txt

```

## ✅ Requirements

• Python = 3.9

• numpy

• pandas

• scikit-learn
