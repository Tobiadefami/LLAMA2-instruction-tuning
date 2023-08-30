import requests
import os
import json
import typer
import tqdm
from bs4 import BeautifulSoup
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import time

# ... (fetch_bibtex, process_bibtex_futures functions remain unchanged)

def fetch_paper_metadata(identifier, headers):
    url = f"http://export.arxiv.org/oai2?verb=GetRecord&metadataPrefix=arXiv&identifier={identifier}"
    response = requests.get(url, headers=headers)
    return response

# ... (other functions and imports remain unchanged)

def scrape_arxiv_for_machine_learning_papers(num_papers: int, start_index: int, max_workers: int = 5):
    base_url = "http://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXiv"  # Use OAI-PMH ListRecords endpoint
    resumption_token = None
    fetched_papers = 0  # Counter for fetched papers

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    result: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=max_workers)

    while fetched_papers < num_papers:
        url = base_url

        if resumption_token:
            url += f"&resumptionToken={resumption_token}"

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "xml")

            records = soup.find_all("record")
            for record in records:
                identifier = record.find("identifier").text.strip()
                paper_metadata_future = executor.submit(fetch_paper_metadata, identifier, headers)

                result.append(
                    {
                        # Extract required metadata here (similar to your previous code)
                        "paper_metadata_future": paper_metadata_future,
                    }
                )

                fetched_papers += 1
                if fetched_papers >= num_papers:
                    break

            resumption_token_tag = soup.find("resumptionToken")
            if resumption_token_tag:
                resumption_token = resumption_token_tag.text.strip()
            else:
                break  # No more records to fetch

        else:
            print(f"Failed to fetch data from arXiv. Status Code: {response.status_code}")
            break

    return process_paper_metadata_futures(result)

def save_checkpoint(data, checkpoint_file):
    with open(checkpoint_file,'w') as f:
        json.dump(data, f, indent=4)


def load_checkpoint(checkpoint_file):

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as json_file:
            return json.load(json_file)
    return {}



def main(num_papers: int = 3, data_dir='test'):
    # ... (rest of your code remains unchanged)

    if data_dir is not None:
        data_dir = os.path.join(os.path.dirname(__file__), data_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        output_file = f"{data_dir}/machine_learning_papers.json"
    else:
        output_file = "machine_learning_papers.json"
    
    checkpoint_file: str = os.path.join(data_dir, 'checkpoint.json')

    checkpoint_data = load_checkpoint(checkpoint_file=checkpoint_file)

    for start_index in tqdm.trange(checkpoint_data.get('start_index', 0), num_papers, 100): # scrape in batches of 100
        papers = scrape_arxiv_for_machine_learning_papers(num_papers=num_papers, start_index=start_index)
        if papers:
            with open(output_file, "w") as json_file:
                json.dump(papers, json_file, indent=4)
            print(f"{len(papers)} papers scraped and saved into {output_file}")
            save_checkpoint({'start_index': start_index + 100}, checkpoint_file)

if __name__ == "__main__":
    typer.run(main)
