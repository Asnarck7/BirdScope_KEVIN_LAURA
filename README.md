# 🐦 BirdScope — Intelligent Bird Dataset Pipeline

<div align="center">

### Bird Detection, Dataset Cleaning and Deep Learning Training Pipeline

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,50:203A43,100:2C5364&height=180&section=header&text=BirdScope&fontSize=48&fontColor=ffffff"/>

<br>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=for-the-badge">
<img src="https://img.shields.io/badge/OpenCV-ComputerVision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">

</div>

---

# 📌 Overview

BirdScope is a preprocessing and training pipeline for bird image classification.

The project automatically detects birds, cleans datasets, removes low-quality samples and prepares balanced datasets for Deep Learning workflows.

---

# ✨ Main Features

* 🐦 Automatic bird detection with YOLOv8
* ✂️ Intelligent bird cropping
* 🧹 Dataset cleaning
* 🔍 Duplicate filtering using Perceptual Hash
* 🌫 Blur detection
* 🌑 Dark image filtering
* ☀️ Overexposed image filtering
* 📏 Resize pipeline (224x224)
* ⚖️ Dataset balancing
* 🧠 Parent → Child classification workflow
* ⚡ CUDA support

---

# 🔄 Pipeline

```mermaid
graph TD;

A[Raw Images] --> B[YOLOv8 Detection]
B --> C[Crop Bird]
C --> D[Quality Validation]
D --> E[Duplicate Removal]
E --> F[Dataset Cleaning]
F --> G[Training Dataset]
G --> H[Deep Learning Model]
```

---

# 📂 Repository Structure

```text
Proyecto_Aves/
│
├── 1_detectar_recortar.py
├── Modelo_Entrenamiento_Padre_Hijo_Aves.ipynb
├── README.md
├── requirements.txt
│
├── dataset_original/     (not included)
├── cleaned_dataset/      (generated automatically)
├── training_padre_hijo/  (not included)
```

---

# ⚠️ Dataset

The dataset is NOT included in this repository due to storage limitations.

Expected structure:

```text
dataset_original/
├── species_1/
├── species_2/
├── species_3/
```

Add your images inside each species folder before executing the pipeline.

---

# 🧹 Cleaning Strategy

The system evaluates all images before selecting the best samples.

Applied validations:

| Validation           | Purpose                    |
| -------------------- | -------------------------- |
| Blur Detection       | Remove blurry images       |
| Dark Detection       | Remove underexposed images |
| Bright Detection     | Remove overexposed images  |
| Duplicate Removal    | Perceptual Hash filtering  |
| Bird Size Validation | Remove tiny detections     |
| Quality Ranking      | Preserve best samples      |

---

# ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/Asnarck7/Proyecto_Aves.git
cd Proyecto_Aves
```

Create environment:

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Linux:

```bash
source venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Pipeline

```bash
python 1_detectar_recortar.py
```

---

# 🧠 Technologies

* Python
* YOLOv8
* OpenCV
* PyTorch
* NumPy
* Pillow
* ImageHash
* Jupyter Notebook

---

# 👨‍💻 Authors

Kevin Julian Guerrero Penagos
Laura

Computer Vision • Deep Learning • Artificial Intelligence
