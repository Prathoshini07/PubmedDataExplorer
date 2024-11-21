import streamlit as st
import psycopg2
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# Function to connect to PostgreSQL
def connect_db():
    try:
        connection = psycopg2.connect(
            user="postgres",  # Replace with your PostgreSQL username
            password="Sanjeeth1.,",  # Replace with your PostgreSQL password
            host="localhost",  # Replace with your PostgreSQL host if necessary
            port="5433",  # Replace with your PostgreSQL port if necessary
            database="postgres"  # Replace with your PostgreSQL database name
        )
        return connection
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Function to get all article titles and PMIDs for clustering
def get_all_articles(connection):
    cursor = connection.cursor()
    query = "SELECT pmid, title FROM Datas"
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return pd.DataFrame(results, columns=['pmid', 'title'])

# Function to get PubMed data by PMID
def get_pubmed_data_by_pmid(pmid, connection):
    cursor = connection.cursor()
    query = "SELECT * FROM Datas WHERE pmid = %s"
    cursor.execute(query, (pmid,))
    result = cursor.fetchone()
    cursor.close()
    return result

# Function to ask a question using Hugging Face's GPT-2 (or a better model like GPT-Neo or GPT-J)
def ask_llm(question, context):
    try:
        generator = pipeline('text-generation', model='gpt2')  # Replace 'gpt2' with a larger model if available
        prompt = f"Context: {context}\nQuestion: {question}"
        response = generator(prompt, max_length=1000, max_new_tokens=400, num_return_sequences=1)
        return response[0]['generated_text'].strip()
    except Exception as e:
        st.error(f"Error querying LLM: {e}")
        return None

# Streamlit UI
st.title("PubMed Data Explorer with LLM and KMeans Clustering")

# Input field for PMID
pmid = st.text_input("Enter a PubMed ID:", value="")

if pmid:
    connection = connect_db()
    if connection:
        pubmed_data = get_pubmed_data_by_pmid(pmid, connection)
        if pubmed_data:
            st.subheader("PubMed Data")
            st.write(f"*PMID*: {pubmed_data[0]}")
            st.write(f"*Title*: {pubmed_data[5]}")
            st.write(f"*Abstract*: {pubmed_data[6]}")
            st.write(f"*Authors*: {pubmed_data[7]}") 
            st.write(f"*Journal*: {pubmed_data[8]}")
            st.write(f"*Publication Date*: {pubmed_data[9]}")

            # Input field for asking LLM a question
            question = st.text_input("Ask a question about this article:")
            if question:
                context = f"Article Title: {pubmed_data[5]}"
                answer = ask_llm(question, context)
                st.subheader("LLM Answer")
                st.write(f"*\nAnswer*: {answer}")
            
            # Clustering similar articles
            st.subheader("Similar Articles (Based on KMeans Clustering)")
            all_articles = get_all_articles(connection)
            
            # Vectorize the titles
            vectorizer = TfidfVectorizer(stop_words='english')
            title_vectors = vectorizer.fit_transform(all_articles['title'])
            
            # Apply KMeans clustering
            num_clusters = 10  # Adjust as needed
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            all_articles['cluster'] = kmeans.fit_predict(title_vectors)
            
            # Get the cluster of the current article
            article_cluster = all_articles[all_articles['pmid'] == int(pmid)]['cluster'].values[0]
            
            # Find articles in the same cluster
            similar_articles = all_articles[(all_articles['cluster'] == article_cluster) & (all_articles['pmid'] != int(pmid))]
            
            # Display only the top 5 similar articles
            top_similar_articles = similar_articles.head(5)
            if not top_similar_articles.empty:
                for _, article in top_similar_articles.iterrows():
                    st.write(f"[{article['title']} (PMID: {article['pmid']})](https://pubmed.ncbi.nlm.nih.gov/{article['pmid']})")
            else:
                st.write("No similar articles found in this cluster.")
        else:
            st.error(f"No data found for PMID {pmid}")

        connection.close()
