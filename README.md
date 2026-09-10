# TFM – Chest X-ray Analysis

Deep Learning project developed as part of my Master's Thesis in Computer Engineering at the University of Jaén, Spain.

##  Master's Thesis

**Clasificación de imágenes médicas para diagnóstico automatizado**

The project investigates Deep Learning and Computer Vision techniques for the automated classification of chest X-ray images, with a focus on model performance, generalization and visual interpretability.

##  Research Objectives

- Develop Deep Learning models for automated chest X-ray classification.
- Compare different CNN architectures based on performance and computational cost.
- Investigate the effect of input image resolution on classification performance.
- Evaluate binary and multiclass classification scenarios.
- Apply Explainable AI techniques to analyze the regions influencing model predictions.

##  Methodology

The experimental pipeline includes:

**Data Preparation → Preprocessing → Data Augmentation → Transfer Learning → Fine-Tuning → Model Evaluation → Explainable AI**

The study evaluates several pretrained CNN architectures:

- ResNet50
- ResNet101
- DenseNet121
- DenseNet201
- EfficientNetB2
- EfficientNetB3

Transfer learning was performed using ImageNet-pretrained models, followed by full fine-tuning. Data augmentation, class weighting, dropout, Early Stopping, ReduceLROnPlateau and ModelCheckpoint were used to improve generalization and training stability.

##  Classification Tasks

### Binary Classification

The first experimental stage distinguishes between:

- **Finding** – pathological finding present
- **No Finding** – no pathological finding

This stage was used to evaluate the ability of the models to distinguish normal from pathological chest X-rays.

### Multiclass Classification

The study was subsequently extended to multiple pulmonary conditions, including:

- Atelectasis
- Effusion
- Emphysema
- No Finding
- Nodule
- Pneumonia
- Pneumothorax

Different class configurations were investigated to analyze the effect of class imbalance and diagnostic difficulty.

##  Model Selection

Among the evaluated architectures, **DenseNet201** achieved the best overall performance in the binary classification experiment.

| Architecture | Accuracy | AUC | Macro F1 |
|--------------|----------|-----|----------|
| ResNet50 | 88% | 0.966 | 0.87 |
| ResNet101 | 90% | 0.970 | 0.90 |
| DenseNet121 | 90% | 0.959 | 0.90 |
| **DenseNet201** | **92%** | **0.985** | **0.92** |
| EfficientNetB2 | 89% | 0.948 | 0.89 |
| EfficientNetB3 | 88% | 0.954 | 0.87 |

DenseNet201 achieved **92% accuracy, 0.985 AUC and 0.92 macro F1-score** on the test set. For the Finding class, precision and recall both reached 0.94. :contentReference[oaicite:1]{index=1}

##  Experimental Configuration

The models were evaluated using independent training, validation and test sets, together with cross-validation and comparative experiments.

Key techniques included:

- Image preprocessing
- Data augmentation
- Transfer learning with ImageNet
- Full fine-tuning
- Class weighting
- Dropout regularization
- Early Stopping
- ReduceLROnPlateau
- ModelCheckpoint
- Label smoothing

For the advanced multiclass experiments, DenseNet201 was evaluated at different image resolutions: **224×224, 512×512 and 768×768 pixels**. The 768×768 configuration provided the best overall performance and more localized Grad-CAM activations. :contentReference[oaicite:2]{index=2}

##  Multiclass Results

The multiclass experiments investigated the effect of the number of diagnostic categories.

With seven classes, performance was strongly affected by underrepresented and visually challenging pathologies. Removing the most problematic minority classes progressively improved the results.

The best five-class configuration achieved:

**Accuracy: 82%**  
**Macro F1-score: 0.80**  
**Weighted F1-score: 0.82**

The configuration included:

- Effusion
- Emphysema
- No Finding
- Pneumonia
- Pneumothorax

A comparable configuration replacing Effusion with Nodule achieved 79% accuracy and a macro F1-score of 0.78. :contentReference[oaicite:3]{index=3}

##  Explainable AI – Grad-CAM

To improve model interpretability, Grad-CAM was used to visualize the image regions that contributed to the model's predictions.

The analysis compared:

**Original X-ray → Clinical Mask → Grad-CAM Activation**

DenseNet201 produced the most precise and consistent activation maps among the evaluated architectures. Higher-resolution images (768×768) also produced more localized activations. :contentReference[oaicite:4]{index=4}

The visual analysis showed that the model tended to focus on clinically relevant lung regions, particularly for conditions such as:

- Pneumonia
- Pleural Effusion
- Pneumothorax
- Emphysema

However, smaller or more diffuse abnormalities such as Nodules and Atelectasis remained more challenging to localize accurately. :contentReference[oaicite:5]{index=5}

##  Visual Results

### Model Performance

![DenseNet201 Training Curves]
<img width="605" height="217" alt="image" src="https://github.com/user-attachments/assets/fd266a26-d097-4313-a197-109df8ad862b" />


### Confusion Matrix

![DenseNet201 Confusion Matrix]
<img width="605" height="243" alt="image" src="https://github.com/user-attachments/assets/66238f59-9297-49bb-a10b-3ba06e1d51de" />


### Architecture Comparison

![Architecture Comparison]
<img width="472" height="208" alt="{34B57E37-16F9-4501-B2CF-7233A78B26E4}" src="https://github.com/user-attachments/assets/29a86a67-46b7-49bd-abd8-adfb586c91dd" />

### Clinical Mask vs Grad-CAM

![Clinical Mask Comparison]
<img width="605" height="459" alt="image" src="https://github.com/user-attachments/assets/e75cb1f6-f463-4806-a4ff-3d7c172f302b" />

##  Interactive Demo

An interactive Gradio interface was developed to demonstrate the model's predictions and visualization capabilities.

![Gradio Interface]

<img width="503" height="283" alt="image" src="https://github.com/user-attachments/assets/4b2644ff-d28b-45c9-a427-466370615a93" />

## 🛠️ Technologies

**Programming:** Python

**Deep Learning:** TensorFlow, Keras, PyTorch

**Computer Vision:** OpenCV

**Data Science:** NumPy, Pandas, Scikit-learn, Matplotlib

**Tools:** Jupyter Notebook, Git, GitHub, Docker, Linux

**Deployment:** Gradio

