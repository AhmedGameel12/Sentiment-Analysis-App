import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
# Load the pre-trained model and vectorizer

print(nltk.data.path)
model = load_model('sentiment_model.h5')
vectorizer = joblib.load('vectorizer.pkl')

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove numbers and punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit App UI
st.title("Sentiment Analysis App")
st.write("Enter text to analyze its sentiment!")

# Input text from the user
user_input = st.text_area("Enter text:")

if st.button("Analyze"):
    if user_input.strip() != "":
        # Preprocess the input text
        cleaned_text = clean_text(user_input)
        transformed_text = vectorizer.transform([cleaned_text])
        
        # Make predictions
        prediction = model.predict(transformed_text)
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        
        st.write(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"Confidence Score: {prediction[0][0]:.2f}")
    else:
        st.write("Please enter valid text to analyze!")