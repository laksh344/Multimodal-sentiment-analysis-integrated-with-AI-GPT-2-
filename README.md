# Multimodal Sentiment Analysis Integrated with AI GPT-2

This project implements a multimodal sentiment analysis system that integrates facial expression recognition (FER) and EEG (electroencephalogram) signals with AI language models (GPT-2). The goal is to build a robust emotion detection system that classifies emotions from both visual and EEG data while leveraging GPT-2 for natural language understanding.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Setup](#setup)
5. [Workflows](#workflows)
6. [Usage](#usage)
7. [License](#license)

## Project Overview

This project aims to perform sentiment analysis using both **facial expression recognition (FER)** and **EEG signals**, integrating the power of **AI language models** (like GPT-2) to analyze text. The system combines two different data modalities for more accurate emotion detection:
- **Facial Expressions**: Analyzes facial emotion expressions using computer vision techniques.
- **EEG Data**: Processes EEG signals to detect emotions based on brainwave patterns.
- **GPT-2 Integration**: Uses GPT-2 for sentiment analysis from text data, leveraging the language model to provide a deeper understanding of emotions.

The notebook uses libraries such as **TensorFlow**, **Keras**, **PyTorch**, **OpenCV**, and **transformers** to implement the solution.

## Features

- **Multimodal Emotion Recognition**: Combines facial expressions, EEG signals, and text for comprehensive emotion detection.
- **Face Detection**: Uses Haar Cascades for detecting faces in images.
- **EEG Data Processing**: Analyzes EEG signals from datasets such as Bonn EEG for emotional insights.
- **GPT-2 Integration**: Leverages GPT-2 for performing sentiment analysis on textual data.
- **Kaggle API**: Retrieves datasets from Kaggle, such as FER2013 for facial expressions and Bonn EEG for brainwave data.

## Installation

Ensure you have Python 3.x installed. You can install the necessary libraries using the following commands:

```bash
pip install -q --upgrade transformers datasets accelerate torch
pip install -q numpy pandas opencv-python scikit-learn scipy tensorflow Pillow tqdm kaggle seaborn matplotlib
```

## Setup

1. **Kaggle API Setup**:
   - To access datasets from Kaggle, you need to upload your `kaggle.json` file and set up Kaggle API.
   - Generate the `kaggle.json` API key from [Kaggle's API page](https://www.kaggle.com/docs/api).
   - Upload the `kaggle.json` file to the notebook as instructed in "Step 1.5: Configuring Kaggle API".

2. **Face Detection Setup**:
   - The notebook uses Haar Cascade for face detection. Ensure the appropriate files for face detection are available and referenced correctly.

3. **Data Download**:
   - The notebook supports downloading datasets using Kaggle API (e.g., FER2013 for facial expressions, Bonn EEG for EEG signals).

## Workflows

### Workflow 1: **Data Preprocessing**
1. **EEG Signal Processing**:
   - The EEG signals are loaded and split into segments. Features are extracted from the frequency domain using methods like Welch for improved classification.
   
2. **Facial Expression Data Processing**:
   - Images are preprocessed by resizing, normalizing, and detecting faces using Haar Cascade before feeding them into the model.

3. **Text Data Processing**:
   - Text data is processed using GPT-2 for sentiment analysis, and emotions are mapped to predefined categories.

### Workflow 2: **Model Training**
1. **Feature Extraction**:
   - EEG features are extracted using frequency-domain analysis.
   - Facial features are extracted from images using convolutional neural networks (CNNs).
   - GPT-2 is used to process text data and extract sentiment.

2. **Model Architecture**:
   - A multimodal model is created that integrates EEG, facial expressions, and text data to classify emotions. The model could combine neural networks for each data type and combine features for final classification.

3. **Training the Model**:
   - The model is trained using labeled datasets. Hyperparameter tuning and cross-validation are used for optimization.

### Workflow 3: **Emotion Classification**
1. **Unified Emotion Mapping**:
   - Different emotion labels from the datasets are mapped into a unified emotion set, such as "Happy", "Sad", "Angry", "Anxious", etc.

2. **Model Evaluation**:
   - The trained model is evaluated using metrics like accuracy, confusion matrix, and classification report.

### Workflow 4: **Text Analysis with GPT-2**
1. **Sentiment Analysis**:
   - GPT-2 is used to analyze text data and perform sentiment analysis to determine the emotional tone of the text.
   
2. **Emotion Mapping for Text**:
   - The emotional tone of the text is mapped to a unified emotion category, contributing to the multimodal classification.

## Usage

1. **Clone the Repository**:
   Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/multimodal-sentiment-analysis-integrated-with-ai-gpt-2.git
   cd multimodal-sentiment-analysis-integrated-with-ai-gpt-2
   ```

2. **Run the Notebook**:
   Open the Jupyter notebook (`emotion_detection.ipynb`) and execute the cells to process data, train the model, and perform emotion classification.

3. **Model Training**:
   After preprocessing the data, you can proceed to the model training section. The notebook guides you through training and evaluating the emotion detection model.

4. **Evaluate Model Performance**:
   The notebook includes sections for evaluating the model’s performance using classification metrics.

## License

This project is licensed under the MIT License.

---

This README is now more detailed with clear workflows that describe how the project works. It includes instructions on setup, installation, and usage, along with steps for each phase of the project, from data preprocessing to model evaluation and sentiment analysis with GPT-2.
