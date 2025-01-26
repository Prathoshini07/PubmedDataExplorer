import requests
from bs4 import BeautifulSoup
import pandas as pd
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
import time

# Function to fetch data from PubMed
def get_pubmed_data(row):
    pmid = row['pmid']
    lev1_cluster_id = row['lev1_cluster_id']
    lev2_cluster_id = row['lev2_cluster_id']
    lev3_cluster_id = row['lev3_cluster_id']
    lev4_cluster_id = row['lev4_cluster_id']

    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    response = requests.get(url)

    if response.status_code != 200:
        return (pmid, lev1_cluster_id, lev2_cluster_id, lev3_cluster_id, lev4_cluster_id, None, None, None, None, None, None, None)

    soup = BeautifulSoup(response.content, 'html.parser')

    title_section = soup.find('h1', class_='heading-title')
    title = title_section.get_text(strip=True) if title_section else "Title not found"

    abstract_section = soup.find('div', class_='abstract-content selected')
    abstract = abstract_section.get_text(strip=True) if abstract_section else "Abstract not found"

    authors_section = soup.find('span', class_='authors-list-item')
    authors = ", ".join([author.get_text(strip=True) for author in authors_section.find_all('a')]) if authors_section else "Authors not found"

    journal_section = soup.find('span', class_='journal-title')
    journal = journal_section.get_text(strip=True) if journal_section else "Journal not found"

    date_section = soup.find('span', class_='cit')
    pub_date = date_section.get_text(strip=True) if date_section else "Publication date not found"

    doi_section = soup.find('a', class_='id-link')
    doi = doi_section.get_text(strip=True) if doi_section else "DOI not found"

    # Extract the keywords section
    keywords_section = soup.find('strong', class_='sub-title', string="Keywords:")
    keywords = keywords_section.find_next_sibling(text=True).strip() if keywords_section else "Keywords not found"

    return (pmid, lev1_cluster_id, lev2_cluster_id, lev3_cluster_id, lev4_cluster_id, title, abstract, authors, journal, pub_date, doi, keywords)

# Function to insert data into MongoDB
def insert_data_to_mongo(data):
    try:
        # MongoDB connection URI
        mongo_uri = "mongodb+srv://postgres:12345@cluster0.qnioi.mongodb.net/pubmed?retryWrites=true&w=majority"
        client = MongoClient(mongo_uri)  # Connect to MongoDB
        db = client.get_database() 
        collection = db.get_collection('Datas') 
        collection.insert_many(data)
        print(f"Inserted {len(data)} records into MongoDB.")
    except Exception as e:
        print(f"Failed to insert data into MongoDB: {e}")
    finally:
        client.close()

# Main function to handle the workflow
def main():
    try:
        # Read the first 500 rows from the CSV
        df = pd.read_csv(r"P:\Ml_package\ML\PMID_cluster_relation_202401.csv", nrows=500)

        start_time = time.time()

        # Use ThreadPoolExecutor for concurrent scraping
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(get_pubmed_data, df.to_dict('records')))

        # Filter out entries missing title or abstract
        data = [{
            "pmid": pmid,
            "lev1_cluster_id": lev1_cluster_id,
            "lev2_cluster_id": lev2_cluster_id,
            "lev3_cluster_id": lev3_cluster_id,
            "lev4_cluster_id": lev4_cluster_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "publication_date": pub_date,
            "doi": doi,
            "keywords": keywords
        } for pmid, lev1_cluster_id, lev2_cluster_id, lev3_cluster_id, lev4_cluster_id, title, abstract, authors, journal, pub_date, doi, keywords in results if title and abstract]

        if data:
            insert_start_time = time.time()
            insert_data_to_mongo(data)  # Insert into MongoDB
            print(f"Time taken to insert data: {time.time() - insert_start_time:.2f} seconds")

        print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
