# Brain MRI Tumor Detection

This project demonstrates the use of deep learning to classify brain MRI images as indicating the presence or absence of a tumor. It uses a fine-tuned ResNet50 model and custom data preprocessing to achieve accurate predictions. The dataset is sourced from Kaggle and consists of labeled MRI images categorized into "Tumor" and "No Tumor" classes.

---

## Project Structure

```
Brain-MRI-Tumor-Detection/
|-- data/                    # Directory for the dataset
|-- scripts/                 # Python scripts for training, evaluation, and utilities
|-- models/                  # Saved trained models
|-- README.md                # Project documentation
|-- requirements.txt         # Python dependencies
|-- train_model.py           # Main script for training the model
|-- evaluate_model.py        # Script for evaluating the trained model
|-- visualize_results.py     # Script for visualizing predictions and performance metrics
```

---

## Features

- **Dataset Handling**: Automated download and organization of the dataset.
- **Data Augmentation**: Preprocessing pipeline includes augmentation techniques like flipping, brightness adjustment, and contrast adjustment.
- **Model Training**: Fine-tuning ResNet50 for binary classification.
- **Custom Generators**: Custom `Sequence`-based data generator for efficient data loading and augmentation.
- **Evaluation**: Metrics such as accuracy, loss, confusion matrix, and classification report.
- **Visualization**: Displays predictions, training history, and confusion matrix.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Brain-MRI-Tumor-Detection.git
   cd Brain-MRI-Tumor-Detection
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Use Docker:
   ```bash
   docker build -t brain-mri-tumor-detection .
   docker run -it brain-mri-tumor-detection
   ```

---

## Usage

### Dataset Preparation

The dataset will be automatically downloaded and organized when you run the training script. Ensure your Kaggle API credentials are set up.

### Train the Model

Run the following command to train the model:
```bash
python train_model.py
```

### Evaluate the Model

To evaluate the trained model on the test set:
```bash
python evaluate_model.py
```

### Visualize Results

To display predictions and metrics:
```bash
python visualize_results.py
```

---

## Results

- **Validation Accuracy**: ~XX%  
- **Test Accuracy**: ~XX%  

### Example Predictions

| Actual       | Predicted    | Image                              |
|--------------|--------------|------------------------------------|
| Tumor        | Tumor        | ![Example1](path/to/image1.png)   |
| No Tumor     | Tumor        | ![Example2](path/to/image2.png)   |

---

## Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection). It consists of two categories:

- **Yes**: MRI images with brain tumors.
- **No**: MRI images without brain tumors.

---

## Model Architecture

- **Base Model**: Pre-trained ResNet50 with weights from ImageNet.
- **Custom Head**:
  - GlobalAveragePooling2D layer
  - Dense layer with 256 units and ReLU activation
  - Dropout layer with a rate of 0.5
  - Final Dense layer with sigmoid activation

---

## Contributions

Contributions are welcome! If you'd like to add features or fix bugs, feel free to submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- The dataset creators on Kaggle.
- TensorFlow and Keras for the deep learning framework.
- Scikit-learn for metrics and utilities.

