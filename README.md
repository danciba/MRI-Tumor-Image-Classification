# MRI-Tumor-Image-Classification

This project implements a deep learning model for the classification of brain tumor images using the **EfficientNetB0** architecture, with custom data augmentation and preprocessing. The dataset is sourced from Kaggle's [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).

## Features
- **Model**: Transfer learning with EfficientNetB0.
- **Augmentation**: Custom data augmentation with TensorFlow.
- **Loss Function**: Focal loss to handle class imbalances.
- **Metrics**: Accuracy, Precision, Recall, and AUC.
- **Evaluation**: Classification report and confusion matrix for test data.

---

## Dataset
The dataset contains MRI images labeled as:
- `yes`: Images showing a brain tumor.
- `no`: Images without a brain tumor.

The dataset is split into:
- **Training Set**: 80% of data.
- **Validation Set**: 10% of data.
- **Test Set**: 10% of data.

---

## Installation

### Prerequisites
Ensure you have Python 3.12+ installed.

### Clone the Repository
```bash
git clone https://github.com/danciba/MRI-Tumor-Image-Classification.git
cd MRI-Tumor-Image-Classification
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Kaggle API
To download the dataset, set up the Kaggle API:
1. Create an account on [Kaggle](https://www.kaggle.com/).
2. Generate an API token from your account settings.
3. Place the downloaded `kaggle.json` file in `~/.kaggle/`.

---

## Usage

### Train the Model
Run the Python script to train the model:
```bash
MRI-Tumor-Image-Classification.py
```

### Test the Model
Modify the script to evaluate the test set (refer to the `test` section in the code).

---

## Results

### Validation Set
- **Validation Loss**: 1.6621
- **Validation Accuracy**: 0.9200
- **Validation Precision**: 0.9333
- **Validation Recall**: 0.9333
- **Validation AUC**: 0.9933

However, the confusion matrix for the validation set revealed some inconsistencies:
```
              precision    recall  f1-score   support

           0       0.40      0.40      0.40        10
           1       0.60      0.60      0.60        15

    accuracy                           0.52        25
   macro avg       0.50      0.50      0.50        25
weighted avg       0.52      0.52      0.52        25
```

### Test Set
- **Test Loss**: 1.6662
- **Test Accuracy**: 0.8462
- **Test Precision**: 0.8333
- **Test Recall**: 0.9375
- **Test AUC**: 0.9469

The confusion matrix for the test set showed:
```
              precision    recall  f1-score   support

           0       0.50      0.40      0.44        10
           1       0.67      0.75      0.71        16

    accuracy                           0.62        26
   macro avg       0.58      0.57      0.58        26
weighted avg       0.60      0.62      0.61        26
```

These results indicate that while the overall accuracy metrics appear high, the confusion matrices highlight performance issues in classifying the individual classes, particularly for the minority class.

---

## File Structure
```
MRI-Tumor-Image-Classification/
├── brain_tumor_detection.py   # Main Python script
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file
```

---

## Acknowledgements
- [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- TensorFlow and Keras documentation.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.