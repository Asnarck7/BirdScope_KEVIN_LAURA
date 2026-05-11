# 🐦 Bird Dataset Cleaner using YOLOv8

Automatic bird dataset preprocessing pipeline using YOLOv8.

This project detects birds in images, crops them automatically, removes low-quality samples, filters duplicates, and generates a balanced dataset ready for deep learning training.

---

# ✨ Features

- Bird detection with YOLOv8
- Automatic image cropping
- Duplicate image removal using perceptual hashing
- Blur detection
- Dark/bright image filtering
- Minimum bird size validation
- Dataset balancing
- Automatic resize to 224x224
- Ready for CNN training

---

# 📂 Input Structure

Place your dataset inside:

```bash
AVES/
```

Example:

```bash
AVES/
├── species_1/
├── species_2/
├── species_3/
```

---

# 📦 Installation

Clone repository:

```bash
git clone https://github.com/your-user/Bird-Dataset-Cleaner.git
cd Bird-Dataset-Cleaner
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Run:

```bash
python bird_dataset_cleaner.py
```

Processed images will be saved automatically in:

```bash
AVES_PROCESADAS_2/
```

---

# ⚙️ Technologies

- Python
- YOLOv8
- OpenCV
- PyTorch
- NumPy
- Pillow

---

# 🧠 Pipeline Overview

1. Detect bird
2. Select best bounding box
3. Expand crop margins
4. Remove low-quality images
5. Remove duplicates
6. Resize image to 224x224
7. Save best images per class

---

# 📊 Output

The script prints statistics such as:

- Total processed images
- Saved images
- Removed duplicates
- Blur removals
- Dark/bright removals
- Final dataset quality percentage

---

# 🚀 Author

Developed for computer vision and deep learning dataset preprocessing.