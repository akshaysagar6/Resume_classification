import pandas
import numpy 
import pickle
import docx 
import PyPDF2
import textract
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import string
import re
import contractions
from spacy import displacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def extract_text(file):
    if file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page_obj = pdf_reader.pages[page_num]
            text += page_obj.extract_text()
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    elif file.name.endswith('.doc'):
        text = textract.process(file).decode('utf-8')
    else:
        text = file.getvalue().decode("utf-8")
    return text




def clean_text(text):
    text_lower = text.lower()
    email_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text_lower)
    url_text = re.sub(r'https?://\S+|www\.\S+', ' ', email_text)
    num_text = re.sub(r'\d+', ' ', url_text)
    stop_text = num_text.translate(str.maketrans('', '', string.punctuation))
    un_text = re.sub("[^A-Za-z]+", " ", stop_text)
    tokens = word_tokenize(un_text)
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in tokens if word not in stop_words]
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(' '.join(filtered_text))
    lemmatized_text = ' '.join(token.lemma_ for token in doc)
    without_single_char = re.sub(pattern=r'\s+[a-zA-Z]\s+', repl=" ", string=lemmatized_text)
    return without_single_char

model = pickle.load(open(r'final_model', 'rb'))
vectorizer = pickle.load(open(r'vectorizer.pkl', 'rb'))


page = st.sidebar.radio("Navigation", ["Home", "About"])


if page == "Home":
    title_html = """
        <div style="background-color:#FFB600;padding:5px;border-radius:5px;font-family:  'Times New Roman', serif;box-shadow: 0px 3px 4px white;">
            <h1 style="color:#ffffff;text-align:center;text-shadow: 6px 6px 6px #000000">Resume Classification</h1>
            <hr style="border: .5px solid white; margin: 10px 0;"color:#212529">
            <p style="color:#212529; text-align:center; font-size: 18px; margin-top: 10px;font-weight: bold;">classify resumes with ease</p>
            
        </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)


    

    uploaded_files = st.file_uploader("Upload resume files", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="file_uploader", help="Please upload PDF, DOCX, or TXT files.")


    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            text2 = extract_text(uploaded_file)
            text3 = clean_text(text2)
            text4 = vectorizer.transform([text3])
            predt = model.predict(text4)
            
            
            subheader_text = f"<span style='font-size: 15px;color:  blue;text-shadow: 1px 1px 2px white;'>Predicted Class for {uploaded_file.name}:</span>"
             
            st.markdown(subheader_text, unsafe_allow_html=True)
            if predt == 0:
                st.write('Internship')
            elif predt == 1:
                st.write('PeopleSoft')
            elif predt == 2:
                st.write('React')
            elif predt==3:
                st.write('SQL')
            else:
                st.write('Workday')
            
if page == "About":
    title_html = """
        <div style="background-color:#1A2238;padding:5px;border-radius:5px;font-family:  'Times New Roman', serif;box-shadow: 0px 3px 4px white;">
            <h2 style="color:#ffffff;text-align:center;text-shadow: 5px 0px 4px #000000">Resume Classification Web Application</h2>
            <hr style="border: .5px solid white; margin: 10px 0;"color:#212529">
            <p style="color:#9DAAF2; text-align:center; font-size: 16px; margin-top: 10px;font-weight: bold;">classify resumes with ease</p>    
        
    """
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown(
        "<h2 style='color: #9DAAF2; font-size: 24px; font-weight: bold; text-transform: uppercase; letter-spacing: 2px;'>Algorithms Used</h2>",
        unsafe_allow_html=True
    )
    st.write("Voting Classifier (Random Forest, XGBoost, MultinomialNB)")
    st.markdown(
        "<h2 style='color: #9DAAF2; font-size: 24px; font-weight: bold; text-transform: uppercase; letter-spacing: 2px;'>Vectorization Method Used</h2>",
        unsafe_allow_html=True
    )
    st.write("TFIDF with Unigram and Bigram Combination")
  
