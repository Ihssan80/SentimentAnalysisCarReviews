# Sentiment Analysis of Car Reviews

This notebook contains Python code for sentiment analysis on a dataset of car reviews. The code leverages natural language processing (NLP) techniques and machine learning models to classify car reviews into positive or negative sentiment categories. Various preprocessing methods, vectorization techniques, and machine learning algorithms are explored to achieve accurate sentiment classification.

## Table of Contents
1. [Introduction](#introduction)
2. [Preprocessing and Text Cleaning](#preprocessing-and-text-cleaning)
3. [Feature Vectorization](#feature-vectorization)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [LSTM-Based Sentiment Analysis](#lstm-based-sentiment-analysis)
6. [How to Use](#how-to-use)

## 1. Introduction
Sentiment analysis is a vital task in natural language processing that involves determining the sentiment or emotion expressed in a piece of text. In this notebook, we focus on sentiment analysis of car reviews using various techniques to achieve accurate classification.

## 2. Preprocessing and Text Cleaning
- The code imports necessary libraries and downloads essential NLTK resources.
- Functions are defined to remove URLs, email addresses, and emojis from text data.
- The `get_wordnet_pos` function is used to map POS tags to WordNet tags for lemmatization.
- The `preprocess` function performs comprehensive text cleaning, including:
  - Lowercasing
  - Removing punctuation, emojis, URLs, and non-English letters
  - Tokenization
  - Optional stemming or lemmatization

## 3. Feature Vectorization
- The code defines the `get_vectorized_data` function to:
  - Split data into training and testing sets
  - Perform Bag-of-Words (BoW) or TF-IDF vectorization on text data
- The chosen vectorization method can be either BoW or TF-IDF.
- Vectorization is performed using the `CountVectorizer` or `TfidfVectorizer` from Scikit-learn.

## 4. Model Training and Evaluation
- The code trains and evaluates the following classifiers:
  - Multinomial Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression
- Hyperparameter tuning is performed using `GridSearchCV` for each model.
- Classification reports and confusion matrices are printed for model evaluation.
- Confusion matrices are visualized using Matplotlib and Seaborn.

## 5. LSTM-Based Sentiment Analysis
- The code explores deep learning with LSTM-based neural networks for sentiment analysis.
- Text data is preprocessed without stemming or lemmatization.
- Tokenization and sequence creation are performed.
- An LSTM-based model is built using Keras with a TensorFlow backend.
- The model is trained and evaluated on the sentiment classification task.

## 6. How to Use
1. Ensure required libraries are installed: Pandas, NumPy, NLTK, Scikit-learn, TensorFlow, Matplotlib, Seaborn.
2. Load the 'car-reviews.csv' dataset containing 'Review' and 'Sentiment' columns.
3. Run each section of the code sequentially to perform different analyses.
4. In sections with hyperparameter tuning, adjust parameter values as needed.
5. Observe classification reports, confusion matrices, and LSTM model evaluation.
6. This notebook provides a guide to the code and its functionalities.
