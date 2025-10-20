# 📚 Paper Reviews Workflow

Hello!  
This folder is a record of my process of **systematically reading, summarizing, implementing, and applying research papers**. Below is the list of papers I have reviewed, along with the overall **pipeline I follow for paper reviews**.


---
## 📑 Paper List  


### 🔎 Correspondence Matching
| No. | Title | Links |
|-----|-------|-------|
| 1 | [***KeyNet*** : *Keypoint Detection by Handcrafted and Learned CNN filters*](./KeyNet/README.md) | [정리](./KeyNet/README.md) · [구현](./KeyNet/구현) |
| 2 | [***SuperPoint*** : *Self-Supervised Interest Point Detection*](./SuperPoint/README.md) | [정리](./SuperPoint/README.md) · [구현](./SuperPoint/구현) |
| 3 | [***SuperGlue*** : *Feature Matching with Graph Neural Nets*](./SuperGlue/README.md) | [정리](./SuperGlue/README.md) · [구현](./SuperGlue/구현) |
| 4 | [***LoFTR*** : *Detector-Free Local Feature Matching*](./LoFTR/README.md) | [정리](./LoFTR/README.md) · [구현](./LoFTR/구현) |
| 5 | [***GeoCNN*** : *CNN Architecture for Geometric Matching*](./GeoCNN/README.md) | [정리](./GeoCNN/README.md) · [구현](./GeoCNN/구현) |
| 6 | [***WeakAlign*** : *Weakly-supervised Semantic Alignment*](./Weakalign/README.md) | [정리](./Weakalign/README.md) · [구현](./Weakalign/구현) |
| 7 | [***NC-Net*** : *Neighbourhood Consensus Networks*](./NCnet/README.md) | [정리](./NCnet/README.md) · [구현](./NCnet/구현) |
| 8 | [***Hyperpixel Flow*** : *Semantic Correspondence with Multi-layer Features*](./HyperpixelFlow/README.md) | [정리](./HyperpixelFlow/README.md) · [구현](./HyperpixelFlow/구현) |
| 9 | [***SD4Match*** : *Stable Diffusion Features for Semantic Matching*](./SD4Match/README.md) | [정리](./SD4Match/README.md) · [구현](./SD4Match/구현) |
| 10 | [***DistillDIFT*** : *Distillation of Diffusion Features*](./DistillDIFT/README.md) | [정리](./DistillDIFT/README.md) · [구현](./DistillDIFT/구현) |
| 11 | [***GeoAware-SC*** : *Geometry-Aware Semantic Correspondence*](./GeoAware-SC/README.md) | [정리](./GeoAware-SC/README.md) · [구현](./GeoAware-SC/구현) |



### 🧩 3D Classification & Segmentation
| No. | Title | Links |
|-----|-------|-------|
| 1 | [***PointNet*** : *Deep Learning on Point Sets for 3D Classification and Segmentation*](./PointNet/README.md) | [정리](./PointNet/README.md) · [구현](./PointNet/구현) |



### 🏗 3D Reconstruction
| No. | Title | Links |
|-----|-------|-------|



### 🎨 Others
| No. | Title | Links |
|-----|-------|-------|
| 1 | [*Image Style Transfer Using CNNs*](./ImageStyleTransfer_CNN/README.md) | [정리](./ImageStyleTransfer_CNN/README.md) · [구현](./ImageStyleTransfer_CNN/구현) · [응용](./ImageStyleTransfer_CNN/응용) |

---

## 1. 📖 Finding Papers (What to Read?)

- Focused primarily on papers from top-tier conferences  
  (e.g., CVPR, ECCV, ICCV, AAAI, NeurIPS, WACV, etc.)  
- Actively utilized curation platforms such as **Arxiv Digest** and **HuggingFace Papers**

---

## 2. 👓 Reading Papers (How to Read?)

- **Reading order**  
  1. First pass: grasp the overall structure (abstract → introduction → conclusion)  
  2. Second pass: deep dive (method [formulas, architecture] → experiments → related work)  

- **Key points to check while reading**  
  - What is the core idea of this paper? / What problem does it aim to solve?  
  - Why is this research important?  
  - What are the limitations or gaps in prior work? (How has related research evolved?)  
  - How does this paper address those limitations? (Key differences and novelties)  
  - What does the proposed structure/algorithm look like?  
  - What results or improvements were achieved with the new method?  
  - What datasets or resources were used? (public vs. collected data, scale, characteristics)  
  - What limitations remain unresolved in this work?

---

## 3. 📝 Digesting Papers (How to Digest?)

**Paper Title:**  
**Authors:**  
**Conference/Journal:**  

- Motivation for selecting this paper

1. What is the core idea of this paper?  
2. Why is this research important?  
3. What are the limitations of prior work?  
4. How does this paper address those limitations?  
5. What is the structure/algorithm of the proposed method?  
6. What results were achieved?  
7. What datasets were used?  
8. 🔎 **Critical Reflection (my own analysis & critique)**  


**Appendix**  
1. Building an intuitive understanding of the structure  
2. Questions I had & how they were resolved  


---

## 4. 🛠 Implementation & Reproduction (Reproduce)

- First, I **independently implemented the core algorithms** and ran small-scale experiments to ensure I understood the underlying mechanics.  
- Next, I explored official or re-implemented codes available on **Papers with Code**, cloning the repositories and carefully stepping through the codebase to verify that I could run and fully understand each component.  
- Finally, I conducted **hands-on modifications**, such as altering network structures or adapting the code to my own datasets, to test whether the methods generalize and to deepen my comprehension.  

> Through this process, I aim not only to reproduce results but also to develop the ability to **critically analyze and extend existing implementations**.


---

## 5. 🚀 Application & Extension

Building on the process of **reviewing, implementing, and reproducing papers**, I further explored the following directions:

- Applying existing methods to **different tasks**  
- **Combining ideas** from multiple papers to discover new possibilities  
- Experimenting with methods on problems aligned with my **own research interests**  
- Testing approaches on **real-world datasets** (e.g., Kaggle, Dacon competitions)  
- Conducting **additional research** when limitations were identified, and extending them into potential research topics  

> Rather than stopping at understanding and reproduction, I aimed to **apply and extend research ideas** to practical tasks, competitive challenges, and even new research directions.


---

This repository documents my journey of **studying papers, implementing ideas, and extending them into new directions**.  
I will continue to update it with deeper explorations and further developments. ✨
