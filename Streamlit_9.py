import pandas as pd
import numpy as np
import streamlit as st
from docx import Document
import pdfplumber
import csv
import io
import os
import json
import re
import warnings
import subprocess
import sys
from textwrap import dedent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import spacy
from openai import OpenAI

# Ignore warnings
warnings.filterwarnings("ignore")

# --- Resource Caching ---
@st.cache_resource
def load_resources():
    """Download NLTK data and load Spacy model once."""
    try:
        nltk.download('stopwords', quiet=True)
        # Ensure spacy model is loaded. 
        # Note: In Streamlit Cloud, ensure 'en_core_web_lg' is in requirements.txt
        nlp = spacy.load("en_core_web_lg")
        return nlp
    except Exception as e:
        st.error(f"Error loading NLP resources: {e}")
        return None

nlp = load_resources()
sw = stopwords.words("english")
sw.extend(['[Redacted]'])

# --- Helper Functions ---

def read_docx(file):
    doc = Document(file)
    data = [para.text for para in doc.paragraphs if para.text.strip() != ""]
    return pd.DataFrame(data, columns=['Redacted Text'])

def read_txt(file):
    data = file.read().decode("utf-8").splitlines()
    return pd.DataFrame([line for line in data if line.strip() != ""], columns=['Redacted Text'])

def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        data = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                data.extend([line for line in text.splitlines() if line.strip() != ""])
        return pd.DataFrame(data, columns=['Redacted Text'])

def calc_sim(user_resume_df, database_df):
    """Calculate cosine similarity between user resume and database."""
    # Convert dataframe to a single string of text
    new_resume_text = " ".join(user_resume_df['Redacted Text'].astype(str).tolist())
    
    db_texts = database_df['Redacted Text'].astype(str).tolist()
    all_texts = db_texts + [new_resume_text]
    
    vec = TfidfVectorizer(stop_words=sw)
    tfidf_matrix = vec.fit_transform(all_texts)
    
    # Compare the last item (user) against all previous items (database)
    similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    similar_indices = similarity_matrix.argsort()[0][-5:][::-1]
    
    return similar_indices

def ner_redaction(text):
    """Redact PII using Spacy NER."""
    if not nlp:
        return text
    doc = nlp(text)
    redacted_text = text

    # Sort entities by length descending to avoid partial redaction issues
    entities = sorted(doc.ents, key=lambda x: len(x.text), reverse=True)
    
    # Specific categories to redact
    target_labels = ['PERSON', 'ORG', 'GPE', 'FAC', 'LOC']
    
    for ent in entities:
        if ent.label_ in target_labels:
            redacted_text = redacted_text.replace(ent.text, "[Redacted]")

    # Regex for Contact Info
    redacted_text = re.sub(r'\S+@\S+', '[Redacted Email]', redacted_text)
    redacted_text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[Redacted Phone]', redacted_text)
    redacted_text = re.sub(r'\b(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9._-]+\b', '[Redacted Website]', redacted_text)
    
    return redacted_text

# --- UI Setup ---

st.set_page_config(page_title="Saxa-4 Recommendation System", layout="wide")

# Use local relative path for images
if os.path.exists('georgetown_image.jpeg'):
    st.image('georgetown_image.jpeg', use_container_width=True)

st.markdown("""
    <style>
    .main-caption { font-size: 24px; font-weight: bold; text-align: center; color: #333; }
    .name-list { font-size: 18px; text-align: center; color: #555; }
    </style>
    <p class="main-caption">Saxa - 4</p>
    <p class="name-list">Nicholas Reese | Ashlyn Bellardine | Osama Bin Habib | Dezmond Richardson | Genesis Roberto</p>
    """, unsafe_allow_html=True)

st.markdown('---')

# --- Data Loading ---
# Using relative paths for deployment compatibility
csv_file_path = 'spacy_redacted_documents_with_id_and_category.csv'
json_db_path = 'redacted_resumes_output.json'

if os.path.exists(csv_file_path):
    resumes_db = pd.read_csv(csv_file_path)
else:
    st.error(f"Database file {csv_file_path} not found.")
    resumes_db = pd.DataFrame(columns=['Redacted Text'])

# --- OpenAI Client Setup ---
# Priority: Streamlit Secrets (for cloud) -> Environment Variable
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# --- Main App Logic ---

st.markdown('# Recommendation System')
st.write('## Section 1 - Loading Resume')

uploaded_file = st.file_uploader(
    'Drag & Drop your resume (PDF, DOCX, TXT, CSV)',
    type=['docx', 'txt', 'pdf', 'csv']
)

resume_df = None
resume_text_raw = ""

if uploaded_file:
    if uploaded_file.name.endswith('.docx'):
        resume_df = read_docx(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        resume_df = read_txt(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):
        resume_df = read_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        resume_df = pd.read_csv(uploaded_file)
    
    if resume_df is not None:
        st.markdown('### Non-Processed Resume')
        st.dataframe(resume_df.head())
        resume_text_raw = " ".join(resume_df.iloc[:, 0].astype(str).tolist())
    else:
        st.error("Could not process file.")

st.markdown('---')

if resume_df is not None and not resumes_db.empty:
    st.markdown('## Section 2 - LLM Recommendations')
    
    default_prompt = dedent("""
        You are a helpful recommender tool. You will be provided a resume and a list of similar job examples. 
        Your goal is to provide the five most similar job roles based on the skills in the resume. 
        For each recommendation, provide a brief explanation.
    """)

    user_prompt = st.text_area("Customize your prompt:", value=default_prompt, height=150)

    if st.button('Get Recommendations'):
        if not client:
            st.error("OpenAI API Key missing. Please check your secrets.")
        else:
            with st.spinner('Analyzing similarities...'):
                indices = calc_sim(resume_df, resumes_db)
                similar_docs = "\n---\n".join(resumes_db.iloc[indices]['Redacted Text'].tolist())
                
                full_query = f"{user_prompt}\n\nUser Resume Content:\n{resume_text_raw}\n\nSimilar Examples to Reference:\n{similar_docs}"
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": full_query}],
                        temperature=0.3
                    )
                    st.markdown("### Recommendation Output")
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"LLM Error: {e}")

    st.markdown('---')
    st.markdown('## Section 3 - PII Redaction')
    
    if st.checkbox("Show Redacted Version"):
        redacted_text = ner_redaction(resume_text_raw)
        st.markdown("### Redacted Resume Content")
        st.text_area("Redacted Output", redacted_text, height=300)
        
        redacted_df = pd.DataFrame({'Redacted Text': [redacted_text]})

        st.markdown('### Section 4 - Recommendation with Prompt Enhancement')
        st.info("This version informs the LLM specifically that information has been redacted.")
        
        if st.button('Get Recommendations (Redacted)'):
            with st.spinner('Processing...'):
                enh_prompt = f"The following resume has PII redacted as [Redacted]. Please recommend 5 roles based on the visible skills:\n\n{redacted_text}"
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": enh_prompt}],
                        temperature=0.5
                    )
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error: {e}")

        # --- Data Privacy Section ---
        st.markdown('---')
        st.subheader('To better our recommendations, could we add your redacted resume to our database?')
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Yes, save my redacted data'):
                # Append to JSON
                new_entry = redacted_df.to_dict(orient='records')
                try:
                    existing_entries = []
                    if os.path.exists(json_db_path):
                        with open(json_db_path, 'r') as f:
                            for line in f:
                                existing_entries.append(json.loads(line))
                    
                    # Check for duplicates
                    if any(e['Redacted Text'] == redacted_text for e in existing_entries):
                        st.warning("This content is already in our database.")
                    else:
                        with open(json_db_path, 'a') as f:
                            f.write(json.dumps(new_entry[0]) + "\n")
                        st.success("Thank you! Your data was added anonymously.")
                except Exception as e:
                    st.error(f"Saving error: {e}")
        
        with col2:
            if st.button('No, keep it private'):
                st.info("We respect your privacy. No data was saved.")

else:
    st.info("Upload a resume to begin the analysis.")
