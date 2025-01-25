import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
from pymongo import MongoClient

# Function to connect to MongoDB
def connect_db():
    try:
        # MongoDB URI
        client = MongoClient("mongodb+srv://postgres:12345@cluster0.qnioi.mongodb.net/")  
        db = client['pubmed']  # Database name
        collection = db['Datas']  # Collection name
        print("Connected to MongoDB")
        return collection
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

# Function to get PubMed data by PMID
def get_pubmed_data_by_pmid(pmid, collection):
    # Query MongoDB to fetch data by PMID
    result = collection.find_one({"pmid": pmid})
    return result

# Function to get the range of PubMed IDs
def get_pmid_range(collection, limit=100):
    # Query MongoDB to get a list of PMIDs
    pmid_list = [doc['pmid'] for doc in collection.find({}, {'pmid': 1}).limit(limit)]
    return pmid_list

# Function to ask a question using GPT-2
def ask_llm(question, context):
    try:
        generator = pipeline('text-generation', model='gpt2') 
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        response = generator(prompt, max_length=500, max_new_tokens=300, num_return_sequences=1)
        return response[0]['generated_text'].strip()
    except Exception as e:
        st.error(f"Error querying LLM: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="PubMed Data Explorer", page_icon="🔬", layout="wide")

# Title and custom CSS for better styling
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            color: #0d6efd;
            font-weight: bold;
        }
        .sub-header {
            font-size: 20px;
            color: #0d6efd;
            font-weight: bold;
        }
        .content {
            font-size: 16px;
            color: #555;
        }
        .button-style {
            background-color: #0d6efd;
            color: white;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">PubMed Data Explorer</div>', unsafe_allow_html=True)
st.markdown("### Explore articles, ask questions, and find similar papers")

# Connect to MongoDB
collection = connect_db()

# Check if the collection was connected successfully
if collection is not None:
    pmid_range = get_pmid_range(collection, limit=210)
    
    # Dropdown for PubMed IDs
    selected_pmid = st.selectbox("Select a PubMed ID:", pmid_range, index=0, help="Choose an article by PubMed ID")

    # Display article details only if a valid PubMed ID is selected
    if selected_pmid and selected_pmid != "Select a PubMed ID":
        pubmed_data = get_pubmed_data_by_pmid(selected_pmid, collection)
        
        if pubmed_data:
            # Display PubMed article data
            with st.expander("PubMed Data", expanded=True):
                st.write(f"**PMID**: {pubmed_data['pmid']}")
                st.write(f"**Title**: {pubmed_data['title']}")
                st.write(f"**Abstract**: {pubmed_data['abstract']}")
                st.write(f"**Authors**: {pubmed_data['authors']}")
                st.write(f"**Journal**: {pubmed_data['journal']}")
                st.write(f"**Publication Date**: {pubmed_data['publication_date']}")

            # Input field for asking LLM a question
            question = st.text_input("Ask a question about this article:")
            if question:
                context = f"Article Title: {pubmed_data['title']}\nAbstract: {pubmed_data['abstract']}"
                answer = ask_llm(question, context)
                st.subheader("LLM Answer")
                st.write(f"**Answer**:\n{answer}")

            # Clustering similar articles
            st.subheader("Similar Articles (Based on KMeans Clustering)")

            # Fetching all articles from MongoDB
            all_articles = pd.DataFrame(list(collection.find({}, {'pmid': 1, 'title': 1})))

            # Vectorize the titles
            vectorizer = TfidfVectorizer(stop_words='english')
            title_vectors = vectorizer.fit_transform(all_articles['title'])

            # Apply KMeans clustering
            num_clusters = 10  # Set the number of clusters
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            all_articles['cluster'] = kmeans.fit_predict(title_vectors)

            # Get the cluster of the current article
            article_cluster = all_articles[all_articles['pmid'] == selected_pmid]['cluster'].values[0]

            # Find similar articles in the same cluster
            similar_articles = all_articles[(all_articles['cluster'] == article_cluster) & (all_articles['pmid'] != selected_pmid)]

            # Display only the top 5 similar articles
            top_similar_articles = similar_articles.head(5)
            if not top_similar_articles.empty:
                for _, article in top_similar_articles.iterrows():
                    st.write(f"[{article['title']} (PMID: {article['pmid']})](https://pubmed.ncbi.nlm.nih.gov/{article['pmid']})")
            else:
                st.write("No similar articles found in this cluster.")
        else:
            st.error(f"No data found for PMID {selected_pmid}")
else:
    st.error("Failed to connect to the database.")
