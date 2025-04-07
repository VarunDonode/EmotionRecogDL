# EmotionRecogDL: Audio Emotion Recognition System

---

## **Overview**

This project focuses on building a deep learning model to **recognize emotions from audio signals**. The model leverages a combination of **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** layers to extract features and capture temporal dependencies in the audio data. The primary objective is to classify audio samples into one of **six emotion categories**.

---

## **Dataset**

The dataset used for this project consists of audio files labeled with corresponding emotions. Multiple public emotional speech datasets such as **CREMA-D, RAVDESS, SAVEE, and TESS** are utilized. Each audio file undergoes preprocessing, including:

- **Feature Extraction:** Using **Mel-frequency cepstral coefficients (MFCCs)** to represent the audio signal.
- **Data Augmentation:** Techniques like **time-stretching** and **pitch-shifting** to increase the dataset size and diversity.

---

## **Preprocessing Steps**

- **Feature Extraction:**  
  Extract MFCC features for each audio sample to capture the essential spectral properties of the audio.

- **Data Augmentation:**  
  Apply transformations such as time-stretching and pitch-shifting to augment the dataset, mitigating issues related to insufficient data.

- **Normalization:**  
  Scale the extracted features using **StandardScaler** to improve model convergence and performance.

- **One-Hot Encoding:**  
  Encode the emotion labels into a one-hot vector format for compatibility with the multiclass classification model.

---

## **Model Architecture**

The model architecture combines the feature extraction capabilities of **CNNs** with the sequence modeling power of **LSTMs**. The architecture includes:

- **Convolutional Layer:**  
  - **Extracts spatial features** from the input data.  
  - **Kernel size:** 5  
  - **Filters:** 256  
  - **Activation:** ReLU

- **LSTM Layers:**  
  - **Captures temporal dependencies** in the sequential data.  
  - **Four LSTM layers** with 128 units each help model the dynamics of the audio signal.

- **Fully Connected Layers:**  
  - Processes the **flattened feature vector**.  
  - Includes **dropout layers for regularization** to prevent overfitting.

- **Output Layer:**  
  - **Dense layer** with **6 units** (corresponding to the six emotion categories) and **softmax activation** for multiclass classification.

---

## **Challenges**

- **Low Accuracy Due to Class Imbalance:**  
  The model may struggle with skewed class distributions, affecting overall performance.

- **Overfitting During Training:**  
  Limited data and complex model architecture can lead to overfitting, where the model performs well on training data but poorly on unseen data.

---

## **Potential Improvements**

- **Data Augmentation:**  
  Increase the diversity of the dataset using additional augmentation techniques to further improve the model's robustness.

- **Hyperparameter Tuning:**  
  Experiment with different learning rates, batch sizes, and optimizer configurations to optimize model performance.

- **Model Architecture:**  
  Consider adding more LSTM layers or experimenting with **bidirectional LSTMs** to better capture temporal dependencies.

- **Regularization:**  
  Enhance regularization by incorporating techniques like **dropout** and **L2 regularization** in deeper layers.

- **Pretrained Models:**  
  Leverage pretrained audio feature extraction models such as **Wav2Vec** or **OpenL3** to improve the initial feature representations.
