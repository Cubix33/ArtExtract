# 🎨 ArtExtract - GSoC 2026 Evaluation Tasks
**Organization:** HumanAI Umbrella Organization  
**Applicant:** Anshika Singh ([@Cubix33](https://github.com/Cubix33))

This repository contains the completed evaluation tasks for the prospective GSoC 2026 application for the **ArtExtract** project. 

---

## 📂 Repository Structure

```text

ArtExtract_AnshikaSingh/
├── TASK1/
│   ├── artextract_task1.ipynb
│   ├── artextract_task1.pdf
│   ├── cross_validation.py
│   └── training_script.py
├── TASK2/
│   ├── artextract_task2.ipynb
│   ├── artextract_task2.pdf
│   └── feature_extraction.py
|___________________________

```

---

## 📌 Task 1: Convolutional-Recurrent Architectures
**Objective:** Build a model based on convolutional-recurrent architectures to classify Style, Artist, Genre, and other attributes using the ArtGAN (WikiArt) dataset, and identify outliers.

This directory contains the pipeline for classification and outlier detection. Due to computational constraints, heavy model training was executed externally on a college server. The notebook includes the complete strategy discussion, evaluation metrics, and the methodology used to find paintings that do not visually fit their assigned artist or genre.

* 📓 **`artextract_task1.ipynb`** — Main Jupyter notebook detailing strategy, model architecture, evaluation, and outlier detection.
* 📄 **`artextract_task1.pdf`** — Static PDF export of the notebook with outputs.
* ⚙️ **`training_script.py`** — Standalone training script executed on the server.
* 🔄 **`cross_validation.py`** — Rigorous cross-validation script executed on the server.

> 💾 **[Download Best Model Weights Here](https://drive.google.com/file/d/1MKGXfsLC0q-Yb-6rj51rQ6vGFBaRra54/view?usp=sharing)**

---

## 📌 Task 2: Similarity (Semantic Image Retrieval)
**Objective:** Build a model to find similarities in paintings (e.g., portraits with a similar face or pose) using the National Gallery of Art (NGA) open dataset.

This section contains the Content-Based Image Retrieval pipeline. It utilizes self-supervised Vision Transformers (DINOv2) to extract features across 127,000+ images, enabling zero-shot retrieval based on semantic pose and structural composition rather than text metadata. The notebook thoroughly discusses the chosen strategy and the evaluation metrics (mAP, MRR) used to validate performance.

* 📓 **`artextract_task2.ipynb`** — Main Jupyter notebook containing the DINOv2 vector similarity search, metric evaluation, and visual galleries demonstrating pose matching.
* 📄 **`artextract_task2.pdf`** — Static PDF export of the notebook with outputs.
* ⚙️ **`feature_extraction.py`** — DINOv2 extraction script executed on the server to process the massive NGA dataset.

> 🧠 **[Download 127k Feature Vector Index (.npy) Here](https://drive.google.com/file/d/1HgA-DcAij_IBHBqFDYeQ-cBGd1uieS1R/view?usp=sharing)**

---

##  Shared Data Files (Task 2)
The following NGA Open Data files are placed in the root directory and are required for the Task 2 notebook to run:
* **`objects.csv`**: Primary metadata (Title, Artist, Classification).
* **`published_images.csv`**: Standardized IIIF Image URLs.
* **`objects_terms.csv`**: Semantic tags used for targeted evaluation filtering (e.g., isolating portraits).

---

## 🚀 Execution Instructions
1. Clone this repository to your local environment.
2. Download the heavy model weights (Task 1) and the `.npy` feature index (Task 2) from the Google Drive links provided above.
3. Place the downloaded files in their respective task folders, or update the file path variables directly inside the Jupyter notebooks.
4. Run the notebooks to reproduce the evaluations and view the interactive components.

```
