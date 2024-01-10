import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')



# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

# Get the input SMS from the user
input_sms = st.text_area("Enter the message", value="")

# Function to preprocess text 
def transform_text(text):
    text = text.lower() 
    tokens = nltk.word_tokenize(text) 
    filtered_tokens = [token for token in tokens if token.isalnum()] 
    filtered_tokens = [token for token in filtered_tokens if token not in stopwords.words('english') and token not in string.punctuation]
    porter = PorterStemmer() 
    stemmed_tokens = [porter.stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter an SMS or email to predict.")
    else:
        # Preprocess the input
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])

        # Predict using the model
        result = model.predict(vector_input)[0]

        # Display prediction result
        if result == 1:  # Spam
            st.header("Spam")
        else:  # Not Spam
            st.header("Not Spam")

# Display info
st.markdown("<p style='color: green'>Imbalanced Dataset, Accuracy 99.03 % & Precision 99.61 % with the used dataset. Implemented Multinomial Naive Bayes (MNB) model using TF-IDF vectorizer with a max feature limit of 3000.</p>", unsafe_allow_html=True)
