# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from imblearn.over_sampling import SMOTE
import os

# Load and preprocess data
df = None
try:
    df = pd.read_csv("C:/Users/kumar/Downloads/spam.csv", encoding='ISO-8859-1')
    
    # Keep only relevant columns and rename them
    df = df[['v1', 'v2']]
    df.columns = ['Category', 'Message']
    
    st.write("Columns after renaming:", df.columns.tolist())  # Display renamed columns

except FileNotFoundError:
    st.error("The file 'spam.csv' was not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the file: {e}")
    st.stop()

# Ensure the DataFrame was loaded successfully before proceeding
if df is not None:
    # Convert 'Category' column to binary for spam detection
    df['spam'] = df['Category'].apply(lambda x: 1 if x.strip().lower() == 'spam' else 0)

    # Model selection in Streamlit sidebar
    st.sidebar.title("Choose a Model")
    model_choice = st.sidebar.selectbox("Select a model:", ["MultinomialNB", "GaussianNB", "BernoulliNB", "ComplementNB"])

    # Vectorize text data
    vect = TfidfVectorizer()
    X = vect.fit_transform(df['Message'])
    Y = df['spam']

    # Handle class imbalance with SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, Y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Model training based on user selection
    if model_choice == "MultinomialNB":
        model = MultinomialNB(alpha=1)
    elif model_choice == "GaussianNB":
        model = GaussianNB()
        X_train, X_test = X_train.toarray(), X_test.toarray()
    elif model_choice == "BernoulliNB":
        model = BernoulliNB(alpha=1)
    elif model_choice == "ComplementNB":
        model = ComplementNB(alpha=1)

    # Train and evaluate model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    clf_rep = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(clf_rep).transpose()

    # Display results in Streamlit app
    st.title(f'{model_choice} - Spam Detection')
    st.subheader('Confusion Matrix')
    fig, ax = plt.subplots(figsize=(8, 6)) 
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    st.subheader('Classification Report')
    st.write(df_report)

    # Generate Word Clouds for spam and ham messages
    st.subheader('Word Clouds')
    for category in df['Category'].unique():
        text = ' '.join(df[df['Category'] == category]['Message'])
        wordcloud = WordCloud(width=600, height=400, background_color='white').generate(text)
        st.image(wordcloud.to_array(), caption=f"Word Cloud for {category}", use_column_width=True)

    # Save model functionality
    if st.button('Save Model'):
        os.makedirs('models', exist_ok=True)
        model_path = f'models/spam_model_{model_choice}.pickle'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        st.success(f"Model {model_choice} saved successfully at {model_path}!")

    # User input for single message prediction
    st.subheader('Test a Single Message')
    single_message = st.text_input("Enter a message to classify as Spam or Ham:")

    if st.button("Classify Message"):
        if single_message:
            single_message_transformed = vect.transform([single_message])
            if model_choice == "GaussianNB":
                single_message_transformed = single_message_transformed.toarray()

            single_prediction = model.predict(single_message_transformed)
            prediction_label = "Spam" if single_prediction[0] == 1 else "Ham"
            st.write(f"Prediction: {prediction_label}")
        else:
            st.error("Please enter a message to classify.")








