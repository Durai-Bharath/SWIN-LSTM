# MRI-Brain Tumor Detection using SWIN-LSTM Model

## 📖 Description
The **MRI-Brain Tumor Detection using SWIN-LSTM model** project aims to accurately detect brain tumors using a hybrid model that combines the strengths of **SWIN Transformer** and **LSTM**. This system effectively classifies binary and multiclass labels, offering a reliable tool for medical imaging analysis. It also features a simple and intuitive user interface for uploading MRI scans and displaying the classification results.

The model was trained effectively using an **integrated GPU** for improved performance and compared against a **CNN-LSTM** model to evaluate and validate its output.

---

## 🚀 Key Features
- **Hybrid Model**: Combines the **SWIN Transformer** for visual feature extraction and **LSTM** for sequential data processing.
- **Binary and Multiclass Classification**: Accurately classifies MRI scans into tumor and non-tumor classes or multiple tumor types.
- **GPU-Accelerated Training**: Utilized an integrated GPU to significantly enhance model training speed and efficiency.
- **Model Comparison**: Compared the performance of SWIN-LSTM with a CNN-LSTM model to validate its effectiveness.
- **User-Friendly Interface**: Intuitive web-based UI built with Flask for uploading MRI scans and viewing results in real time.

---

## 🔧 Tech Stack
- **Programming Language**: Python
- **Libraries Used**:
  - `torch` (PyTorch) for implementing and training the model.
  - `Flask` for building the web interface.

---

## 🧠 Algorithms
1. **SWIN Transformer**:
   - A hierarchical vision transformer that efficiently extracts visual features.
   - Reference: ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/abs/2103.14030)
2. **LSTM (Long Short-Term Memory)**:
   - A type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data.
   - Reference: ["Long Short-Term Memory"](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

## 🔮 Future Implementations
1. **Cutting-Edge Integration**: Incorporate advanced technologies for further enhancing model accuracy and efficiency.
2. **3D MRI Support** : Expand functionality to process 3D MRI scans for volumetric analysis.
3. **Explainable AI**: Integrate explainability features for better interpretability of classification results.
