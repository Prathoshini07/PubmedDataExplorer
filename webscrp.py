import requests
from bs4 import BeautifulSoup
import pandas as pd
import psycopg2
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

# Function to insert data into the PostgreSQL database
def insert_data_to_postgres(data, connection):
    try:
        cursor = connection.cursor()
        # SQL query to insert data into the Datas table
        insert_query = """
        INSERT INTO Datas (pmid, lev1_cluster_id, lev2_cluster_id, lev3_cluster_id, lev4_cluster_id, title, abstract, authors, journal, publication_date, doi, keywords) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(insert_query, data)
        connection.commit()
    except Exception as e:
        print(f"Failed to insert data: {e}")
    finally:
        cursor.close()

# Function to create the Datas table if it doesn't exist
def create_table_if_not_exists(connection):
    try:
        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS Datas (
            pmid BIGINT PRIMARY KEY,
            lev1_cluster_id INT,
            lev2_cluster_id INT,
            lev3_cluster_id INT,
            lev4_cluster_id INT,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            journal TEXT,
            publication_date TEXT,
            doi TEXT,
            keywords TEXT
        )
        """
        cursor.execute(create_table_query)
        connection.commit()
    except Exception as e:
        print(f"Failed to create table: {e}")
    finally:
        cursor.close()

# Main function to handle the workflow
def main():
    try:
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(
            user="postgres",
            password="Sanjeeth1.,",
            host="localhost",
            port="5433",
            database="postgres"
        )
        print("Connected to the database")

        # Create the Datas table if it doesn't exist
        create_table_if_not_exists(connection)

        # Read the first 500 rows from the CSV
        df = pd.read_csv("C:/Users/sanje/Desktop/ML/PMID_cluster_relation_202401.csv", nrows=500)

        start_time = time.time()

        # Use ThreadPoolExecutor for concurrent scraping
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(get_pubmed_data, df.to_dict('records')))

        # Filter out entries missing title or abstract
        data = [(pmid, lev1_cluster_id, lev2_cluster_id, lev3_cluster_id, lev4_cluster_id, title, abstract, authors, journal, pub_date, doi, keywords) 
                for pmid, lev1_cluster_id, lev2_cluster_id, lev3_cluster_id, lev4_cluster_id, title, abstract, authors, journal, pub_date, doi, keywords in results 
                if title and abstract]

        if data:
            insert_start_time = time.time()
            insert_data_to_postgres(data, connection)
            print(f"Time taken to insert data: {time.time() - insert_start_time:.2f} seconds")

        print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if connection:
            connection.close()
            print("PostgreSQL connection is closed")

if __name__ == "__main__":
    main()
