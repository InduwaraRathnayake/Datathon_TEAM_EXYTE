# Stage 2 Instructions

This guide covers only the steps required to run Stage 2 of the project.

## Prerequisites

- Python 3.x installed
- Virtual environment set up and activated
- Required dependencies installed (see `requirements.txt`)

## Steps to Run Stage 2

1. **Activate the virtual environment**

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies** (if not already done)

   ```powershell
   pip install -r requirements.txt
   ```

3. **Navigate to the Stage 2 directory**

   ```powershell
   cd stage2
   ```

4. **Create the multi-label CSV**

   ```powershell
   python create_labels_csv.py
   ```

   - This will generate `multi_label_dataset.csv` from the images in the dataset folders.

5. **Encode the labels for training**

   ```powershell
   python label_encoding.py
   ```

   - This will create `multi_label_encoded.csv` with multi-hot encoded labels for each image.

6. **Preprocess images and generate numpy arrays for training**

   ```powershell
   python data_loader.py
   ```

   - This will load images, preprocess them, and save numpy arrays (`X_train.npy`, `y_train.npy`, etc.) for fast model training.

7. **Train the model**

   ```powershell
   python train_model.py
   ```

   - This will train the model and save it as `final_model.h5` in the `stage2` directory.

8. **Test the trained model**

   ```powershell
   python evaluate_test.py
   ```

   - This will evaluate the trained model on the test set and print accuracy, precision, recall, and F1-score (overall and per class).

9. **Run the Stage 2 application**
   ```powershell
   python app.py
   ```

## Deactivate the Virtual Environment

```powershell
deactivate
```

---

For any issues, check error messages and ensure all required files are present in the `stage2` directory.
