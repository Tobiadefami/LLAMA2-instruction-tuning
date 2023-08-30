import requests
import os
import json
import typer
import tqdm
from bs4 import BeautifulSoup
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import time



def fetch_bibtex(bibtex_url:str, headers:dict[str, str]):
    response = requests.get(bibtex_url, headers=headers)
    return response

def process_bibtex_futures(papers:list[dict[str, Any]]) -> list[dict[str, Any]]:
    for paper in papers:
        bibtex_response = paper.pop('bibtex_future').result()
        bibTex_citation = (
            bibtex_response.content.decode("utf-8")
            if bibtex_response.status_code == 200
            else f"failed to retrieve BibTeX data for {paper['url']}"
        )
        paper['bibTex_citation'] = bibTex_citation
    return papers


def scrape_arxiv_for_machine_learning_papers(num_papers: int, start_index: int, max_workers: int = 5):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=cat:cs.LG&sortBy=submittedDate&sortOrder=descending&start={start_index}&max_results={num_papers}"

    url = base_url + search_query

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    
    # response = requests.get(url, headers=headers)
    executor = ThreadPoolExecutor(max_workers = max_workers)
    result: list[dict[str, Any]] = []
    arxiv_id = ""

    response = None
    retries = 4
    delay = 5  # Initial delay in seconds

    while retries > 0:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                break  # Success, exit the loop
            else:
                print(f"Received status code: {response.status_code}, Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                retries -= 1
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}, Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
            retries -= 1


    if response and response.status_code == 200:
        soup = BeautifulSoup(response.content, "xml")

        entries = soup.find_all("entry")
        for entry in entries:
            title = entry.find("title").text.strip()
            authors = [author.text for author in entry.find_all("name")]
            pub_date = entry.find("published").text.strip()
            source_tag = entry.find("category", scheme="http://arxiv.org/schemas/atom")
            source = str(source_tag).split('term="')[1].split('"')[0]
            arxiv_id += entry.find("id").text.split("/")[-1]
            data_url = entry.find("id").text.strip()
            
            # Fetch BibTeX data for the paper
            
            bibtex_url = f"https://arxiv.org/bibtex/{arxiv_id}"
            bibtex_future = executor.submit(fetch_bibtex, bibtex_url, headers=headers)
            

            result.append(
                {
                    "prompt": f"Obtain the title, author, publication year of the paper {title}, and generate the citation",
                    "title": title,
                    "authors": authors,
                    "publication_year": pub_date,
                    "source": source,
                    "url": data_url,
                    "bibtex_future": bibtex_future,
                }
            )


        return process_bibtex_futures(result)
    else:
        print(f"Failed to fetch data from arXiv. Status Code: {response.status_code}")
        return None


def save_checkpoint(data, checkpoint_file):
    with open(checkpoint_file,'w') as f:
        json.dump(data, f, indent=4)


def load_checkpoint(checkpoint_file: str | os.PathLike)->str|dict[str, Any]:

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as json_file:
            return json.load(json_file)
    return {}

def main(num_papers: int = 3, data_dir=None):

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
            print(f"{num_papers} papers scraped and saved into {output_file}")
            save_checkpoint({'start_index': start_index+100}, checkpoint_file)

if __name__ == "__main__":
    typer.run(main)
