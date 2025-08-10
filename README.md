# Urban Issue Classification with Privacy Protection

## Project Overview

This project implements a complete pipeline for urban issue detection with privacy protection:

*   **Privacy Protection:** Automatically detects and anonymizes faces and license plates in urban images.
*   **Urban Issue Classification:** Identifies common urban issues like potholes, garbage, and damaged signs.

---

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv env

# Activate on Windows
env\Scripts\activate

# Activate on Linux/Mac
source env/bin/activate
```

### 2. Install Requirements

```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Download Pre-trained Models

*   **Urban Issue Classifier:**
    *   Download `final_model.h5` from: [https://dms.uom.lk/s/RPnXHnJ4AgjzPF6](https://dms.uom.lk/s/RPnXHnJ4AgjzPF6)
    *   Place it in the `stage2_Classifier` folder.

*   **Face and License Plate Detection Models:**
    *   The notebook expects the following models:
        *   Face model: `best.pt`
        *   License plate model: `best_plate.pt`
    *   These should be part of your repository or downloaded separately.

### 4. Create Input/Output Directory Structure

```bash
# Create directories for test images and results
mkdir -p Final/test/input
mkdir -p Final/test/output
mkdir -p Final/test/anonymized
```

### 5. Add Test Images

Place your test images in the `input` directory:

```bash
# Example: Copy some test images
cp path/to/your/images/*.jpg Final/test/input/
```

---

## Running the Notebook

1.  Open `Final.ipynb` in the Jupyter interface.
2.  Run all cells in order.
3.  View the results in the `output` directory.
    *   Comparison images showing original vs. anonymized
    *   Confidence scores for each urban issue class
    *   Summary of detections and classifications

---

## Viewing Results

The notebook generates:

*   **Anonymized Images:** Privacy-protected versions of your input images.
*   **Comparison Visualizations:** Side-by-side views of original and anonymized images.
*   **Classification Charts:** Bar charts showing confidence scores for each urban issue class.
*   **Summary Table:** Overview of all processed images with detection counts.