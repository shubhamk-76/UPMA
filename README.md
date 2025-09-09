# [CVIP 2025] UPMA: Unsupervised Pseudo Mask Attention for Camouflaged Object Detection using Foundation Models and Cue-Guided Refinement

![Framework](figure/Architecture_Diagram.png)

---
## 📌 Overview
**UPMA** introduces an unsupervised approach for **Camouflaged Object Detection (COD)** that leverages **foundation models** to generate high-quality pseudo masks, followed by **cue-guided refinement**.  

This pipeline avoids the need for expensive manual annotations and it is highly interpreteble than other UCOD methods.
---

## 📂 Datasets
We evaluated UPMA on four standard COD datasets:  

- **COD10K**: [Link](https://dengpingfan.github.io/pages/COD.html)  
- **CAMO**: [Link](https://sites.google.com/view/ltnghia/research/camo)  
- **NC4K**: [Link](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment)  
- **CHAMELEON**: [Link](https://drive.google.com/drive/folders/1LN4sP2DRtWcWHcgDcaZcWBVZfoJKccJU?usp=drive_link)  

**Training Split:**  
- 3,040 images from **COD10K**  
- 1,000 images from **CAMO**
- Rest all images we used for testing.

---

## ⚙️ How to Reproduce

Getting started:  
### 1. Generate Pseudo Masks -> 2. Generate Cue mask(S)  -> 3. Training
```bash
cd PseudoMaskGenerator/scripts
python generate_pseudo_mask.py
python Cue_gen.py
python train.py
python test.py
PySODEvalToolkit/results.txt
```

## 📊 Our Results

### Quantitative Results
![Result](figure/Result.png)  

- Our results are **highlighted in yellow**.  
- We present two variants of our **Pseudo Mask Generator**:  
  - **w/B** → with bounding box guidance  
  - **wo/B** → without bounding box guidance  

---

### Qualitative Results
![Qualitative Result](figure/Qualititative_Result.png)  

**Qualitative comparison on CAMO dataset:**  
- Competing methods (**UCOS-DA**, **FOUND**) sometimes produce sharper boundaries but often introduce **background artifacts and false positives** (red, Rows 2, 4, 6).  
- **UPMA** achieves **smoother masks** that more consistently capture the **entire camouflaged object** with fewer spurious regions (green, Rows 2, 4, 6).  


### Acknowledgement

**Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W., Dollár, P., & Girshick, R. (2023).** *Segment Anything.* arXiv. https://arxiv.org/abs/2304.02643



