#!/usr/bin/env python
# coding: utf-8

# # Part 1: Solving using methods - Naïve Bayes classifier

# ####
#  The first step is importing all the required libraries that will be used to create the classification model.
#  These libraries include NumPy, Pandas, NLTK (Natural Language Toolkit), WordCloud, scikit-learn, 
#  and Matplotlib for data manipulation, text processing, machine learning, and visualization.

# In[1]:


#Importing all required libraries which will be used to create our classification models 

import numpy as np
import random

import pandas as pd

import string
import re

import nltk
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer,SnowballStemmer, PorterStemmer
from nltk import pos_tag

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Embedding,Bidirectional, LSTM, Dense, Dropout

import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading Data from .csv file and Labels Encoding

# In[2]:


# Read the dataset from the 'car-reviews.csv' file into a DataFrame
df1 = pd.read_csv("car-reviews.csv")
df1.head()


# In[3]:


# Convert the 'Sentiment' labels from {Neg, Pos} to {0, 1}
labels = {'Neg': 0, 'Pos': 1}

df1['Sentiment'] = df1['Sentiment'].map(labels)
df=df1.copy()


# In[4]:


df


# ###  Download required nltk packages (One time) to facilitate preprocessing.

# In[5]:


#Download only for (One time) to facilitate preprocessing.
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
stop_words = stopwords.words('english')


# Text Preprocessing Functions
# ##### 1. remove_links_and_emails(text):
# This function takes a text input as a parameter and cleans the text by removing any related URLs and email addresses using regular expressions (Regex). URLs and email addresses are common in text data, especially in reviews, and can be noise during analysis or modeling.
# 
# ##### 2. remove_emojis(text):
# This function is used to remove non-text objects such as emojis, shapes, lines, and symbols from the text. Emojis and other graphical elements might not provide valuable information for natural language processing tasks and can be safely removed.
# 
# ##### 3. get_wordnet_pos(tag):
# This function is used to convert the POS tags obtained from the NLTK pos_tag() function to the appropriate POS tags recognized by WordNetLemmatizer. Lemmatization requires the POS tag to perform accurate word reduction.
# 
# - Above 3 functions are used (run) in the below (Preprocessing function).
# 
# ##### 4. preprocess(text, flag):
# This is the main preprocessing function used for cleaning the text and preparing it for vectorizing and modeling. It applies various text cleaning techniques such as removing stopwords, non-English letters, punctuation, numbers, surrounded letters, and repeated spaces. The flag parameter is used to determine whether to use stemming or lemmatization.
# 
# 
# * Stemming: refers to the process of removing suffixes and reducing a word to some
# base form such that all different variants of that word can be represented by the same
# form (e.g., “car” and “cars” are both reduced to “car”). This is accomplished by applying
# a fixed set of rules (e.g., if the word ends in “-es,” remove “-es”), In this excersise we used snowball stemming (as requested)
# 
# 
# * Lemmatization
# Involves some amount of linguistic analysis of the word and its
# context, it is expected that it will take longer to run than stemming, and it’s also typically
# used only if absolutely necessary. We’ll see how stemming and lemmatization are
# useful in the next chapters. The choice of lemmatizer is optional; we can choose
# NLTK or spaCy given what framework we’re using for other pre-processing steps in
# order to use a single framework in the complete pipeline.
# 
# Reference:
# Title: Practical Natural Language Processing
# Authors: Somya Vajala, Bodhisattwa Majumder, Anuj Gupta, Harshit Surana
# Publisher: O'Reilly
# Pages: 53-54

# In[6]:


# This Function is used to remove URLs (links) and email addresses from the all the text in "Reviews column"
def remove_links_and_emails(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    return text

# Function to remove emojis from the text
def remove_emojis(text):
    emoji_pattern = re.compile(
        pattern="[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251" 
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)



def get_wordnet_pos(tag):
    """Map POS tag to first character used by WordNetLemmatizer."""
    tag = tag[0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)  # Default to Noun if the tag is not found


# Add the get_wordnet_pos function here

def preprocess(text, flag):
    stop_words = set(stopwords.words("english"))
    
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-English letters
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuations
    text = remove_emojis(text)  # Remove emojis
    text = remove_links_and_emails(text)  # Remove URLs and email addresses
    text = re.sub(r"\w*\d\w*", "", text)  # Remove numbers and surrounding letters
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Remove repeated spaces
    word_tokens = word_tokenize(text)  # Tokenize the text into words
    filtered_words = [w for w in word_tokens if len(w) > 1]  # Filter out single-letter words

    if flag == "stemmer":
        stemmer = SnowballStemmer('english')  # Create a Snowball Stemmer instance
        words = [stemmer.stem(word) for word in filtered_words]  # Perform stemming
       
    elif flag == "lemmatizer":
        lemmatizer = WordNetLemmatizer()
        tagged_words = pos_tag(filtered_words)  # Get Part-of-Speech tags for lemmatization
        words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_words]
        
    else:
        raise ValueError("Invalid flag. Please choose 'stemmer' or 'lemmatizer' as the flag.")
    words = [w for w in words if w.lower() not in stop_words]  # Remove stopwords
    
    return " ".join(words)


# ### Implement preprocessing after choosing "Stemming"
# 
# ##### In this step we use apply method to implement Preprocessing on all reviews then store the result in the new column "New_Reviews". 
# as per requested in part1 , we will use stemming and get the result for that we will pass value "Text_Processing[0]" in the flag and New_Reviews will built result based on this input.

# In[7]:


Text_Processing=['stemmer','lemmatizer']
# Apply the preprocessing function to the 'Review' column and store the results in 'New_Reviews' column
df["New_Reviews"] = df["Review"].apply(lambda x: preprocess(x, Text_Processing[0]))


# In[8]:


df


# #### Below statement will show the Reviews before and after preprocessing (stored in "New_Reviews") ,using Stemming.
# * After preprocessing its clear to all that 2nd text is does not have any stopwords,shapes emojis ,emails , URLs .

# In[9]:


print(f'Before preprocessing, Reviews:\n\n\n {df["Review"]} \n\n\n\n\nAfter using Stemming Preprocessing, Reviews became New_Reviews:\n\n\n {df["New_Reviews"]}')


# ### Visualize Possitive sentiments for example using word cloud 

# In[10]:


# Visualization - Word Cloud for Positive Reviews
positive_reviews = df[df["Sentiment"] == 1]["New_Reviews"].str.cat(sep=' ')
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud - Positive Reviews')
plt.show()


# In[ ]:





# ### Make the text vectorised by choosing " BoW or TF-IDF" vectorising methods
# 
# ##### -get_vectorized_data() :
# 
# This function works after choosing one of vectorising methods " BoW or Tf-idf " where both methods used to convert text into numerical feature vectors,where both of them are widely used in Natural Language Processing tasks including text classification and information retrieval.
# 
# below function implement text vectorising after assigning (vectorization_method) value to be "bow" or "Tf-idf".
# as per the part one , we start by "bow" and in part2 we will use tfidf in for improving results.
# 
# * BoW "Bag of Words": 
# (BoW) is a classical text representation technique that has been used
# commonly in NLP, especially in text classification problems . The key
# idea behind it is as follows: represent the text under consideration as a bag (collection)
# of words while ignoring the order and context. The basic intuition behind it is
# that it assumes that the text belonging to a given class in the dataset is characterized
# by a unique set of words. If two text pieces have nearly the same words, then they
# belong to the same bag (class). Thus, by analyzing the words present in a piece of text,
# one can identify the class (bag) it belongs to.
# 
# * Tf-IDF "Term Frequency-inverse document frequency":
# 
# TF (term frequency) measures how often a term or word occurs in a given document.
# Since different documents in the corpus may be of different lengths, a term may
# occur more often in a longer document as compared to a shorter document. To normalize
# these counts, we divide the number of occurrences by the length of the document.
# TF of a term t in a document d is defined as:
# 
# Reference:
# Title: Practical Natural Language Processing
# Authors: Somya Vajala, Bodhisattwa Majumder, Anuj Gupta, Harshit Surana
# Publisher: O'Reilly
# Pages: 87-88

# In[11]:


def get_vectorized_data(X, y, vectorization_method='bow'):
    if vectorization_method == 'bow':
        vectorizer = CountVectorizer()
    elif vectorization_method == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid vectorization method. Please choose 'bow' or 'tfidf'.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and transform the preprocessed text data into a numerical feature matrix
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)

    return X_train_v, X_test_v, y_train, y_test, vectorizer


# We have text data (X) and corresponding sentiment labels (y) obtained from a DataFrame (df).
# * The user can choose between two vectorization methods: Bag-of-Words ('bow') or TF-IDF ('tfidf').
# * The text data is preprocessed using functions to remove URLs, emails, emojis, and non-text elements, and perform normalization.
# * The preprocessed data is split into training and testing sets (X_train_v, X_test_v, y_train, y_test).
# * The chosen vectorization method is applied to convert the text data into numerical format.
# * The feature matrix, representing the word counts after text cleaning and normalization, is printed as a DataFrame.
# *In summary, this code performs text preprocessing, vectorization, and train-test split, and then prints the feature matrix as a DataFrame containing word counts for each review, preparing the data for sentiment analysis.

# In[12]:


# Assuming you already have X (New_Reviews) and y (Sentiment) from your DataFrame
X = df["New_Reviews"]
y = df["Sentiment"]

# Let the user choose the vectorization method ('bow' or 'tfidf')
chosen_vectorization_method = 'bow'  # Change this to 'tfidf' if you want to use TF-IDF

# Get the vectorized data and other train-test splits
X_train_v, X_test_v, y_train, y_test, vectorizer = get_vectorized_data(X, y, chosen_vectorization_method)

# Print the feature matrix as a DataFrame
print(pd.DataFrame(X_train_v.toarray(), columns=vectorizer.get_feature_names_out()))


# ## Classification: by using Naive Byes (MultinomialNB) classifier and Calculating the accuracy of the model:
# 
# In below code ,the Naive Byes(MultinomialNB) classification model is used to fit,predict ,find accuracy and printing classification report on the splitted data (previously).
# 
# * Accuracy is a common evaluation metric used to measure the performance of a classification model. It represents the proportion of correctly classified instances (samples) out of the total number of instances in the dataset.
# * in our result below accuracy is 82% after we used "stemming" and "BoW".

# In[13]:


# Train the classifier (Multinomial Naive Bayes) using the training data
classifier = MultinomialNB()
classifier.fit(X_train_v, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test_v)

# Evaluate the model by calculating accuracy and printing the classification report and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)



# ### Visualise the Accuracy result:
# 
# * confusion matrix is another method to show number of correct predicted samples compared to real samples.
# * It can be used after importing accuracy_score from sklearn.metrics import accuracy_score ,and it shows results visualised.

# In[14]:


conf_matrix = confusion_matrix(y_test, y_pred)

# Visualization - Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# ## Conclusion : 

# # Part 2: Solving using advanced methods:

# #### In this part, we will try to improve the accuracy achieved in Part 1 through the following enhancements:
# * 1- Lemmatization instead of stemming will be used, with printed results to showcase the differences.
#      Lemmatization will be used with BoW then with Tf-idf and we will find the best accuracy between both combinations.
# 
# * 2- We will use Pipeline() and GridSearch() methods from Scikit-learn  To explore additional classifiers
# 
# * 3- LSTM (deep learning) method will be used to enhance the model and get better accuracy.

# ### 1- Lemmatization :
# 
# * We will use a copy from origional dataframe then implement Lemmatizing on it.
# * Text_Processing value will be changed to Text_Processing[1] when applying lambda on the "Review" texts,  to select Lemmatizing and store preprocessed texts in "New_Review".
# 
# * in make X=df_Opt["New_Reviews"] and y=df_Opt["Sentiment"]
# * is highlighted earlier about Lemmatization is a linguistic technique that reduces words to their base or root form, known as the lemma, by taking into account the word's context and grammatical meaning. as shown in below printed result.
# 

# In[15]:


df_Opt=df1.copy()
# Apply the preprocessing function to the 'Review' column and store the results in 'New_Reviews' column

Text_Processing=['stemmer','lemmatizer']
# Apply the preprocessing function to the 'Review' column and store the results in 'New_Reviews' column

df_Opt["New_Reviews"] = df_Opt["Review"].apply(lambda x: preprocess(x, Text_Processing[1]))


X = df_Opt["New_Reviews"]
y = df_Opt["Sentiment"]


print(f'Before preprocessing, Reviews:\n\n\n {df_Opt["Review"]} \n\n\n\n\nAfter Using Lemmatization Preprocessing Reviews became New_Reviews:\n\n\n {df_Opt["New_Reviews"]}')


# ### Lemmatization with BoW:
# 
# 1- Feature names 
# 2- Accuracy and classification report

# In[16]:


# Let the user choose the vectorization method ('bow' or 'tfidf')
chosen_vectorization_method = 'bow'  # Change this to 'tfidf' if you want to use TF-IDF

# Get the vectorized data and other train-test splits
X_train_v, X_test_v, y_train, y_test, vectorizer = get_vectorized_data(X, y, chosen_vectorization_method)

# Print the feature matrix as a DataFrame
print(pd.DataFrame(X_train_v.toarray(), columns=vectorizer.get_feature_names_out()))


# In[17]:


# Train the classifier (Multinomial Naive Bayes) using the training data
classifier = MultinomialNB()
classifier.fit(X_train_v, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test_v)

# Evaluate the model by calculating accuracy and printing the classification report and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

conf_matrix = confusion_matrix(y_test, y_pred)

# Visualization - Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()




# ### Lemmatization with Tf-idf:
# 
# 1- Feature names 
# 2- Accuracy and classification report

# In[18]:


# Let the user choose the vectorization method ('bow' or 'tfidf')
chosen_vectorization_method = 'tfidf'  # Change this to 'tfidf' if you want to use TF-IDF

# Get the vectorized data and other train-test splits
X_train_v, X_test_v, y_train, y_test, vectorizer = get_vectorized_data(X, y, chosen_vectorization_method)

# Print the feature matrix as a DataFrame
print(pd.DataFrame(X_train_v.toarray(), columns=vectorizer.get_feature_names_out()))


# In[19]:


# Train the classifier (Multinomial Naive Bayes) using the training data
classifier = MultinomialNB()
classifier.fit(X_train_v, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test_v)

# Evaluate the model by calculating accuracy and printing the classification report and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

conf_matrix = confusion_matrix(y_test, y_pred)

# Visualization - Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()




# ### Conclusion :
# * Using the combination of Lemmatization and Tf-idf give the highest accuracy as mentioned.

# ### 2- Pipeline and GridSearch():
# 
# * To explore additional classifiers, we will import the Pipeline() and GridSearch() methods from Scikit-learn. These methods offer various advantages and some disadvantages:
# 
# #### Advantages:
# 
# - a. The Pipeline and GridSearch() provide a customized and flexible approach, allowing us to train data on multiple classifiers simultaneously.
# 
# - b. They enable us to fine-tune the hyperparameters of the classifiers, helping us find the best parameter combinations after thorough testing in the background.
# 
# Two new classifiers, Support Vector Machine (SVM) and Logistic Regression, will be employed using the Pipeline and GridSearch() methods.
# 
# #### Disadvantages:
# 
# - a. One potential drawback is that these methods might take longer to execute, possibly a couple of minutes or even more, depending on the size of the data and the complexity of the classifiers.
# 
# - b. In some cases, using these methods may require high-specification machines with ample computational resources.
# 
# By implementing these enhancements, we aim to achieve better accuracy in our classification task while gaining insights into the performance of different classifiers and feature extraction techniques. However, it's important to consider the trade-offs in terms of time and computational resources required when utilizing these methods.
# 
# * These methods will contains 3 Classifiers with their main important hyperparameters:
#   - Naive Byes MultinomialNB()
#   - SVM().
#   - Logistic Regression()
# 
# 

# ### Naive Bayes MultinomialNB()  parameters tuning :
# 
# * Alpha: It represents the additive smoothing parameter. If you choose 0 as its value, then there will be no smoothing.
#   in our examples we suggested list of alpha values to test its impact on the accuracy.
#   if alpha = 0 --> no smoothing.
#   

# ### SVM() parameters tuning:
# 
# * Kernel: SVM algorithms use a set of mathematical functions that are defined as the kernel. The function of kernel is to take     data as input and transform it into the required form. Different SVM algorithms use different types of kernel functions.
#   there are different kernel types each with different parameters (linear, nonlinear, polynomial, radial basis function (RBF),     and sigmoid.
#   
#   in below examples i asked pipe/grid search to choose between RBF (Gaussian radial basis function) which used to transform the data into a higher-dimensional space. This makes it easier to find a separation boundary (1). RBF is used with parameters (gamma and C) , which i will explain below.  
#   
# 
# 
# 
# * Gamma parameter:Defines how far the influence of a single training example reaches (2)  ,it is the determinent of choosing       points of the classes which separated by decision boundary (hyperplane) ([nearest:gamma is high and more curvature] or [nearest with farther points:gamma is low and less  curvature]), gamma can have two options {scale , auto } where scale will use  the formula : 1 / (n_features * X.var()) (var=variance],auto will use formula (1 / n_features).(2)
# 
# * C parameter: trades off correct classification of training examples against maximization of the decision function’s margin.     -For larger values of C, a smaller margin will be accepted if the decision function is better at classifying all training         points correctly. 
#   -A lower C will encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy. In other      words C behaves as a regularization parameter in the SVM.
#   
#   (1),(2) :https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

# ### Logistic Regression () parameters tuning:
# 
# * C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify         stronger regularization. (1)
# 
# * Penalty {L1,L2} : This parameter is used to specify the norm (L1 or L2) used in penalization (regularization).(2)       
#   L1 regularization penalizes the LLF with the scaled sum of the absolute values of the weights: |𝑏₀|+|𝑏₁|+⋯+|𝑏ᵣ|.
#   L2 regularization penalizes the LLF with the scaled sum of the squares of the weights: 𝑏₀²+𝑏₁²+⋯+𝑏ᵣ².(3)
# 
# 
# (1):https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html [Accessed on 8 August] 8/8/2023:09:45
# (2):https://www.tutorialspoint.com/scikit_learn/scikit_learn_logistic_regression.htm [Accessed 8/8/2023:10:05]                      
# (3):https://realpython.com/logistic-regression-python/ [Accessed 8/8/2023:10:40]
# 

# In[20]:


df_Stem=df1.copy()
Text_Processing=['stemmer','lemmatizer']
df_Stem["New_Reviews"] = df_Stem["Review"].apply(lambda x: preprocess(x, Text_Processing[0]))## Text_Processing[0] "use Stemming"
X = df_Stem["New_Reviews"]
y = df_Stem["Sentiment"]


# In[21]:


X_train_v, X_test_v, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# For Multinomial Naive Bayes
nb_clf = MultinomialNB()
nb_params = {
    'tfidf__use_idf': (True, False),
    'clf__alpha': [0.006,0.0085,0.009,0.0095,0.01,0.02,0.03,0.05, 0.1,0.2]
}
# For Support Vector Machine
svm_clf = SVC()
svm_params = {
    'tfidf__use_idf': (True, False),
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma':['scale', 'auto'],
    'clf__C': [1,2,2.2,2.3,2.4,2.5,2.7,2.9,3,3.2]
}
# For Logistic Regression
lr_clf = LogisticRegression()
lr_params = {
    'tfidf__use_idf': (True, False),
    'clf__C': [1,2,2.5,2.85,2.87,2.89,2.9,2.92,2.93,2.95,2.98,3],
    'clf__penalty': ['l1', 'l2']
}


# In[23]:


# Define the pipeline for Multinomial Naive Bayes
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', nb_clf)
])

# Define the pipeline for Support Vector Machine
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', svm_clf)
])

# Define the pipeline for Logistic Regression
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', lr_clf)
])


# In[24]:


# For Multinomial Naive Bayes
nb_grid_search = GridSearchCV(nb_pipeline, nb_params, cv=5, n_jobs=-1)
nb_grid_search.fit(X_train_v, y_train)

# For Support Vector Machine
svm_grid_search = GridSearchCV(svm_pipeline, svm_params, cv=5, n_jobs=-1)
svm_grid_search.fit(X_train_v, y_train)


# For Logistic Regression
lr_grid_search = GridSearchCV(lr_pipeline, lr_params, cv=5, n_jobs=-1)
lr_grid_search.fit(X_train_v, y_train)


# ## Accuracy of 3 classifiers when using (stemming):

# In[25]:


# For Multinomial Naive Bayes
nb_predictions = nb_grid_search.predict(X_test_v)
print("Multinomial Naive Bayes:")
print(classification_report(y_test, nb_predictions))

svm_predictions = svm_grid_search.predict(X_test_v)
print("Support Vector Machine:")
print(classification_report(y_test, svm_predictions))

# For Logistic Regression
lr_predictions = lr_grid_search.predict(X_test_v)
print("Logistic Regression:")
print(classification_report(y_test, lr_predictions))


# ## Parameters which result best accuracy for the  3 classifiers using (stemming) :
# 
# * As we can see below , after hyperparameters tunning and when using Stemming , svc() Regression achieved the best accuracy (83%) using below optimal parameters :
#     
#     -Best Parameters for Support Vector Machine:
#      {'clf__C': 2.3, 'clf__gamma': 'scale', 'clf__kernel': 'rbf', 'tfidf__use_idf': True}

# In[26]:


# For Multinomial Naive Bayes
print("Best Parameters for Multinomial Naive Bayes:")
print(nb_grid_search.best_params_)

# Print the best parameters and classification report for Support Vector Machine
svm_best_params = svm_grid_search.best_params_
print("Best Parameters for Support Vector Machine:")
print(svm_best_params)

# For Logistic Regression
print("Best Parameters for Logistic Regression:")
print(lr_grid_search.best_params_)


# In[ ]:





# In[27]:


df_Lemm=df1.copy()
Text_Processing=['stemmer','lemmatizer']
df_Lemm["New_Reviews"] = df_Lemm["Review"].apply(lambda x: preprocess(x, Text_Processing[1]))## Text_Processing[1] "use Lemmatizing"
X = df_Lemm["New_Reviews"]
y = df_Lemm["Sentiment"]


# In[28]:


X_train_v, X_test_v, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[29]:


# For Multinomial Naive Bayes
nb_clf = MultinomialNB()
nb_params = {
    'tfidf__use_idf': (True, False),
    'clf__alpha': [0.088,0.0895,0.0898,0.09,0.0912,0.0915, 0.1,0.2]
}
# For Support Vector Machine
svm_clf = SVC()
svm_params = {
    'tfidf__use_idf': (True, False),
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma':['scale', 'auto'],
    'clf__C': [0.1,1.5,1.8,1.95,1.98,2,2.03,2.05,2.07,3,3.5,4,5,8]
}
# For Logistic Regression
lr_clf = LogisticRegression()
lr_params = {
    'tfidf__use_idf': (True, False),
    'clf__C': [100,100.2,100.4,100.5,100.8,100.9,101,101.1,101.2,101.5,102,103],
    'clf__penalty': ['l1', 'l2']
}


# In[30]:


# Define the pipeline for Multinomial Naive Bayes
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', nb_clf)
])

# Define the pipeline for Support Vector Machine
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', svm_clf)
])

# Define the pipeline for Logistic Regression
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', lr_clf)
])


# In[31]:


# For Multinomial Naive Bayes
nb_grid_search = GridSearchCV(nb_pipeline, nb_params, cv=5, n_jobs=-1)
nb_grid_search.fit(X_train_v, y_train)

# For Support Vector Machine
svm_grid_search = GridSearchCV(svm_pipeline, svm_params, cv=5, n_jobs=-1)
svm_grid_search.fit(X_train_v, y_train)


# For Logistic Regression
lr_grid_search = GridSearchCV(lr_pipeline, lr_params, cv=5, n_jobs=-1)
lr_grid_search.fit(X_train_v, y_train)


# ## Accuracy of 3 classifiers when using (Lemmatizing):

# In[32]:


# For Multinomial Naive Bayes
nb_predictions = nb_grid_search.predict(X_test_v)
print("Multinomial Naive Bayes:")
print(classification_report(y_test, nb_predictions))

svm_predictions = svm_grid_search.predict(X_test_v)
print("Support Vector Machine:")
print(classification_report(y_test, svm_predictions))

# For Logistic Regression
lr_predictions = lr_grid_search.predict(X_test_v)
print("Logistic Regression:")
print(classification_report(y_test, lr_predictions))


# ## Parameters which result best accuracy for the classifiers using (Lemmatizing) :
# 
# * As we can see below , after hyperparameters tunning and when using Lemmatizing , the 3 classifiers got equal accuracy = (82%).
#           

# In[33]:


# For Multinomial Naive Bayes
print("Best Parameters for Multinomial Naive Bayes:")
print(nb_grid_search.best_params_)

# Print the best parameters and classification report for Support Vector Machine
svm_best_params = svm_grid_search.best_params_
print("Best Parameters for Support Vector Machine:")
print(svm_best_params)

# For Logistic Regression
print("Best Parameters for Logistic Regression:")
print(lr_grid_search.best_params_)


# In[ ]:





# ## Conclusion:
# 
# * After using Stemming and Lemmatizing on 3 classifiers , the best accuracy was for SVC() algorithm using Stemming with (83%) .
# 
# * The selected C value in SVM() model will gain larger margin, therefore a simpler decision function, at the cost of training     accuracy.
# 

# ### 3- Deep Learning (Embedding(glove) and LSTM )
# 
# Below I rewrite the preprocessing functions with excluding stemming and lemmatizing.
# this preprocess will be used on a copy of origional dataset then ,LSTM model will be trained on it .

# In[34]:


# This Function is used to remove URLs (links) and email addresses from the all the text in "Reviews column"
def remove_links_and_emails(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    return text

# Function to remove emojis from the text
def remove_emojis(text):
    emoji_pattern = re.compile(
        pattern="[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251" 
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)



def get_wordnet_pos(tag):
    """Map POS tag to first character used by WordNetLemmatizer."""
    tag = tag[0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)  # Default to Noun if the tag is not found


# Add the get_wordnet_pos function here

def preprocess_without_Stem_Lemm(text):
    stop_words = set(stopwords.words("english"))

    text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-English letters
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuations
    text = remove_emojis(text)  # Remove emojis
    text = remove_links_and_emails(text)  # Remove URLs and email addresses
    text = re.sub(r"\w*\d\w*", "", text)  # Remove numbers and surrounding letters
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Remove repeated spaces
    word_tokens = word_tokenize(text)  # Tokenize the text into words
    filtered_words = [w for w in word_tokens if len(w) > 1]  # Filter out single-letter words
    words = [w for w in filtered_words if w.lower() not in stop_words]  # Remove stopwords
    
    return " ".join(words)


# In[35]:


# make a copy from origional dataset Load the dataset

df_LSTM =df1.copy()


# In[36]:


df_LSTM.head()  # view sample of the new dataset


# In[37]:


# to stablize the accuracy for every running time
np.random.seed(42)
tf.random.set_seed(42)
# Here I load copy of origional dataset and i make text cleaning without stemming and lemmatizing  
df_LSTM["New_Reviews"] = df_LSTM["Review"].apply(lambda x: preprocess_without_Stem_Lemm(x))

X = df_LSTM["New_Reviews"]
y = df_LSTM["Sentiment"]

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text and create sequences
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences to have the same length
max_sequence_length = 200
train_data = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
test_data = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Build the model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=100, input_length=max_sequence_length))

model.add(Bidirectional(LSTM(64)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_split=0.3, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {accuracy * 100:.2f}%")


# ### Conclusion of using Text embedding with LSTM (Deep Learning):
# 
# * LSTM: stands for Long Short-Term Memory Network, which belongs to a larger category of neural networks called Recurrent Neural   Network (RNN). Its main advantage over the vanilla RNN is that it is better capable of handling long term dependencies through   its sophisticated architecture that includes three different gates: input gate, output gate, and the forget gate.(1)
#   LSTM and other deep learning models can perform very well on text classification tasks, there can be situations where           traditional machine learning algorithms outperform them.
# 
# * I implemented LSTM on a copy of origional dataset called it (df_LSTM), then i did preprocessing with avoiding selecting stemming   and lemmatizing .
# 
# * The Accuracy which I got from LSTM =79.06 which is accepted but not the best if we compared it with earlier implemented         algorthms accuracies. 
# 
# * This low accuracy may belong to different reasons from my perspective which are:
# 
#     1- Data set size is small since implementing LSTM require large dataset to do well and to gain better results. 
#     
#     2- Hyperparameter Tuning: LSTM models have several hyperparameters that need to be tuned.
#        i tried tuning these parameters to find best combination, and the result was as shown above. 
#        Finding the optimal combination of hyperparameters (such as the number of layers, units, dropout rates, etc.) can greatly        impact the model's performance but it takes more time to do it.                               
#        
#     3- The embeddings which I did above are learned from scratch during the training process of the model.
#         I suggest to use a pretrained embeddings like (word2vec,glove,etc..) and compare pre/post accuracy for both models.    
#     4- Preprocessing and cleaning the text ( which i already did ) before implementing LSTM can impact the model accuracy also.
#     
#     5- Another impact of small used dataset for LSTM is that this dataset can lead to overfitting.
#         Regularization techniques like dropout and early stopping are important to prevent overfitting in deep learning models.
# 
# 
# (1) https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0 [visited 8/8/2023:12:21]

# In[ ]:




