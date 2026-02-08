# 🐙 Marine Species Image Classification using Deep CNNs & Transfer Learning

This repository presents a **multi-class image classification project** that identifies different sea animals using **Deep Convolutional Neural Networks (CNNs)** and **Transfer Learning with EfficientNetB7**.
The model is trained on a diverse marine life image dataset and evaluated using standard classification metrics.

## 📌 Project Highlights

* 🔍 **23 marine animal classes**
* 🧠 **Transfer Learning with EfficientNetB7**
* 🎯 High-performance CNN-based classifier
* 📊 Detailed evaluation: Accuracy, Precision, Recall, F1-score
* 🔥 Visual explanations using **Grad-CAM**
* 🧪 Error Level Analysis (ELA) for image inspection
* 📈 Training monitoring with TensorBoard & callbacks

## 🧬 Dataset

* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste)

### 🐠 Classes Included (23)

Clams, Corals, Crabs, Dolphin, Eel, Fish, Jelly Fish, Lobster, Nudibranchs, Octopus, Otter, Penguin, Puffers, Sea Rays, Sea Urchins, Seahorse, Seal, Sharks, Shrimp, Squid, Starfish, Turtle/Tortoise, Whale

Total images: **~13,700+**

## 🧠 Model Architecture

* **Base Model:** EfficientNetB7 (ImageNet weights)
* **Input Shape:** (224 × 224 × 3)
* **Custom Dense Layers:**

  * Dense (128) → Dropout (0.45)
  * Dense (256) → Dropout (0.45)
* **Output Layer:** Softmax (23 classes)
* **Optimizer:** Adam (LR = 1e-5)
* **Loss Function:** Categorical Crossentropy

## 🏗️ Project Workflow

1. Data loading & exploration
2. Label distribution analysis
3. Image visualization
4. Error Level Analysis (ELA)
5. Data preprocessing & augmentation
6. Train / validation / test split
7. Model training with callbacks
8. Model evaluation
9. Prediction visualization
10. Confusion matrix & classification report
11. Grad-CAM interpretability

## ⚙️ Training Configuration

* **Batch Size:** 32
* **Epochs:** 100
* **Callbacks Used:**

  * EarlyStopping
  * ModelCheckpoint
  * TensorBoard

> ⚠️ **Note:** GPU acceleration is strongly recommended.
> Some notebook cells include Linux shell commands.

## 📊 Results

* **Test Accuracy:** **~77.7%**
* Strong performance across most classes
* Detailed class-wise metrics available in the notebook
* Confusion matrix and Grad-CAM visualizations included

## 🔥 Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to highlight image regions that influenced the model’s predictions, improving model interpretability and trust.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn
* OpenCV
* Scikit-learn

## 📂 Repository Structure

```bash
├── marine_species_deep_cnns_transfer_learning.ipynb
├── Output.zip
└── README.md
```

## ▶️ How to Run

1. Clone the repository

```bash
git clone https://github.com/NishatTasnim01/Marine-Species-Image-Classification-using-Deep-CNNs-Transfer-Learning
```

2. Open the notebook in **Jupyter Notebook / Google Colab / Kaggle**
3. Enable **GPU**
4. Run cells **top to bottom**

## 🙌 Maintainer & Contact

* **Nishat Tasnim**
* 📎 GitHub: [https://github.com/NishatTasnim01](https://github.com/NishatTasnim01)
* 🔗 LinkedIn: [https://www.linkedin.com/in/nishat-ts](https://www.linkedin.com/in/nishat-ts)

## ⭐ Support & Feedback

If you find this project helpful:

* ⭐ Star the repository
* 🍴 Fork it
* 📝 Share feedback or suggestions

Your support motivates me to build and share more ML & DL projects. Have a wonderful day — and happy coding! 😊🌊
