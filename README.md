# Brain Tumor Detection using Deep Learning

## 📋 Overview
A deep learning project for automated brain tumor detection from MRI scans using Convolutional Neural Networks (CNN). The model achieves 88.24% validation accuracy in distinguishing between tumorous and non-tumorous brain MRI scans.

## 🎯 Key Features
- Binary classification of brain MRI scans (tumor/no-tumor)
- Custom CNN architecture
- Comparative analysis of augmented vs non-augmented approaches
- Simple and efficient preprocessing pipeline
- Ready-to-use prediction function

## 📊 Dataset
- **Source**: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Size**: 253 MRI scans
  - 155 tumor cases
  - 98 non-tumor cases
- **Format**: JPG images
- **Resolution**: Standardized to 150x150 pixels

## 🛠️ Technical Details
### Model Architecture
```python
model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### Performance Metrics
- Validation Accuracy: 88.24%
- Precision: 0.90
- Recall: 0.90
- F1-Score: 0.90

## 📦 Requirements
- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## 🚀 Installation & Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook or use the prediction function:
```python
from model import predict_tumor

result, probability = predict_tumor('path_to_image.jpg')
print(f"Result: {result}")
print(f"Confidence: {probability:.2f}%")
```



## 📈 Results
- Non-augmented model outperformed augmented version
- Balanced performance across classes
- Stable training process
- Effective for preliminary screening

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


## 📧 Contact
- Mohamad Jarad
- Email: swe.mohamad.jarad@gmail.com
- LinkedIn: [Your LinkedIn Profile]([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/mohamad-jarad-976545226/))
- Portfolio 
```


