# Plant Disease Detection using CNN

A TensorFlow/Keras-based image classification project for detecting plant diseases from leaf images. The model classifies images into **38 plant/disease categories** and additionally reports whether a plant is **healthy or unhealthy**.

The workflow is implemented for **Google Colab** and uses the **New Plant Diseases Dataset (Augmented)** from Kaggle.

## Overview

The project performs the following steps:

1. Imports the required Python libraries.
2. Downloads the Kaggle plant disease dataset.
3. Extracts and reorganizes the dataset into training, validation, and test directories.
4. Loads and preprocesses images by resizing them to `128 × 128` pixels and scaling pixel values to `[0, 1]`.
5. Builds a custom convolutional neural network (CNN).
6. Trains the CNN with early stopping.
7. Evaluates the model using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Saves the trained model.
9. Provides a function for predicting the disease/health status of an individual leaf image.
10. Generates a visualized prediction grid for sample test images.

## Dataset

The project uses the Kaggle dataset:

**New Plant Diseases Dataset (Augmented)**  
Kaggle: `vipoooool/new-plant-diseases-dataset`

The downloaded archive is approximately **2.70 GB**.

The dataset contains:

- **70,295 training images**
- **17,572 validation images**
- **38 classes**

The classes include healthy leaves as well as several diseases affecting apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, raspberry, soybean, squash, strawberry, and tomato plants.

### Example classes

- `Apple___Apple_scab`
- `Apple___healthy`
- `Corn_(maize)___Common_rust_`
- `Grape___Black_rot`
- `Potato___Late_blight`
- `Tomato___Early_blight`
- `Tomato___healthy`

## Requirements

The notebook is designed to run in Google Colab.

Main dependencies include:

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

The code also uses the Kaggle API to download the dataset.

## Running the Project

### 1. Open the notebook in Google Colab

Upload/open the Python notebook in Google Colab.

### 2. Configure Kaggle

The original notebook uploads a `kaggle.json` file manually and places it in:

```text
~/.kaggle/kaggle.json
```

Make sure your Kaggle credentials are configured securely. **Do not commit `kaggle.json` or any Kaggle API key to GitHub.**

### 3. Download the dataset

The notebook downloads:

```text
vipoooool/new-plant-diseases-dataset
```

and saves the resulting archive as:

```text
new-plant-diseases-dataset.zip
```

### 4. Extract and organize the dataset

The dataset is extracted into:

```text
plant_disease_data/
```

The notebook locates the nested dataset directory and moves the relevant folders to:

```text
/content/train
/content/valid
/content/test
```

The training and validation generators use the directory structure to infer class labels.

### 5. Image preprocessing

Images are resized to:

```text
128 × 128 × 3
```

Pixel values are normalized using:

```python
rescale=1./255
```

A batch size of:

```text
32
```

is used.

The project uses categorical labels because this is a multi-class classification problem.

## CNN Architecture

The model is a custom sequential CNN with three convolutional blocks:

```text
Input: 128 × 128 × 3

Conv2D: 32 filters, 3 × 3, ReLU
MaxPooling2D: 2 × 2

Conv2D: 64 filters, 3 × 3, ReLU
MaxPooling2D: 2 × 2

Conv2D: 128 filters, 3 × 3, ReLU
MaxPooling2D: 2 × 2

Flatten

Dense: 128, ReLU
Dropout: 0.5

Dense: 38, Softmax
```

The model is compiled using:

- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy
- **Metric:** Accuracy

## Training

The model is configured to train for up to **20 epochs**.

Early stopping is applied using validation loss:

```python
EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

This allows training to stop when validation performance stops improving and restores the best model weights observed during training.

### Training results

The recorded training run reached:

| Epoch | Training Accuracy | Validation Accuracy |
|------:|------------------:|--------------------:|
| 1 | 91.26% | 90.90% |
| 2 | 91.88% | 90.09% |
| 3 | 92.19% | 88.58% |
| 4 | 92.81% | 90.87% |

The notebook ultimately reports:

- **Training Accuracy:** 99.11%
- **Test/Validation Accuracy:** 90.90%
- **Macro Precision:** 91.11%
- **Macro Recall:** 90.89%

The classification report gives an overall:

- **Accuracy:** 89%
- **Macro-average F1-score:** 0.89
- **Weighted-average F1-score:** 0.89

> Note: the notebook uses the directory named `valid` for the generator called `test_gen`. Therefore, references to "test accuracy" in the notebook correspond to this validation/test split used during evaluation.

## Evaluation

The project evaluates the classifier using:

### Confusion Matrix

A confusion matrix is generated with:

```python
confusion_matrix(true_labels, pred_labels)
```

and displayed using `ConfusionMatrixDisplay`.

### Classification Report

Scikit-learn's `classification_report` is used to obtain:

- Precision
- Recall
- F1-score
- Support

for each of the 38 classes.

### Overall metrics

The notebook also calculates:

```text
Train Accuracy
Test Accuracy
Precision
Recall
```

using the corresponding Scikit-learn and Keras evaluation functions.

## Plant Health Prediction

The function:

```python
predict_plant_health(img_path)
```

takes an image path, preprocesses the image, runs inference through the trained CNN, and identifies the predicted class.

The logic then checks whether `"healthy"` appears in the predicted class name.

For example:

```text
Plant is Healthy
```

or:

```text
Plant is Unhealthy — Disease: Late_blight
```

This makes the classifier useful both as a **disease identification model** and as a simple **healthy/unhealthy detector**.

## Example Predictions

The notebook demonstrates predictions such as:

```text
Late_blight
Tomato_Yellow_Leaf_Curl_Virus
Early_blight
Cedar_apple_rust
Bacterial_spot
Spider_mites Two-spotted_spider_mite
Septoria_leaf_spot
Common_rust_
```

It also identifies some samples as healthy.

## Visualization

The project creates:

- Training vs. validation loss plots
- Training vs. validation accuracy plots
- A confusion matrix
- A grid of test images with predicted labels

For the final visualization, unhealthy images are given a red overlay while healthy images are displayed without the disease overlay.

## Saved Model

The trained model is saved as:

```text
plant_disease_model.h5
```

The notebook reports a warning that the HDF5 format is considered a legacy Keras format. A newer implementation can instead save the model using:

```python
model.save("plant_disease_model.keras")
```

## Project Structure

A simplified view of the working directory is:

```text
project/
│
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___healthy/
│   ├── ...
│
├── valid/
│   ├── Apple___Apple_scab/
│   ├── Apple___healthy/
│   ├── ...
│
├── test/
│   ├── sample images
│   └── ...
│
├── new-plant-diseases-dataset.zip
├── plant_disease_model.h5
└── README.md
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | CNN construction and training |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| OpenCV | Image processing and visualization |
| Scikit-learn | Model evaluation |
| Kaggle API | Dataset download |
| Google Colab | Development and execution environment |

## Results

The custom CNN achieved approximately **91% test/validation accuracy** in the recorded run, with macro precision and recall both around **91%**.

The per-class performance varies. Several classes achieved F1-scores above 0.95, while some tomato disease classes showed lower performance. For example:

- `Corn_(maize)___Common_rust_`: F1 ≈ 0.99
- `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`: F1 ≈ 0.99
- `Tomato___Early_blight`: F1 ≈ 0.69
- `Tomato___Late_blight`: F1 ≈ 0.70
- `Tomato___Septoria_leaf_spot`: F1 ≈ 0.69

This indicates that the model performs strongly for many categories but has difficulty distinguishing some visually similar diseases.

## Limitations

The current implementation has several practical limitations:

- It relies on a large augmented Kaggle dataset.
- The model is a relatively simple custom CNN rather than a modern transfer-learning architecture.
- The training and evaluation generators use `shuffle=False`.
- The notebook uses the validation set as the main evaluation set.
- The model can overfit: the recorded training accuracy reaches 99.11% while evaluation accuracy is around 90.90%.
- The healthy/unhealthy decision is derived directly from the predicted class name rather than using a separate binary classifier.
- The model has not been demonstrated to generalize to field images with different lighting, backgrounds, camera conditions, or disease severity.

## Possible Improvements

Potential improvements to the project include:

- Transfer learning with models such as MobileNet, EfficientNet, ResNet, or Xception.
- Data augmentation during training.
- Learning-rate scheduling.
- Class balancing if required.
- A dedicated held-out test set.
- Grad-CAM or other explainability methods to visualize the leaf regions influencing predictions.
- Deployment as a web or mobile application.
- Saving class labels alongside the trained model for easier inference.
- Using the native `.keras` format instead of `.h5`.

## License and Dataset Attribution

The dataset downloaded by the notebook is the **New Plant Diseases Dataset (Augmented)** from Kaggle and is attributed in the notebook as having a `copyright-authors` license.

Refer to the original Kaggle dataset page and its license terms before redistributing the dataset or using it in a public project.

## Author

**Sanjay**

Plant Disease Detection using Convolutional Neural Networks.
