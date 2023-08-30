import os
import json
import aiohttp
import asyncio
import typer
from bs4 import BeautifulSoup
from typing import Any
import lxml

async def fetch_arxiv_paper(session, arxiv_id, headers):
    bibtex_url = f"https://arxiv.org/bibtex/{arxiv_id}"
    async with session.get(bibtex_url, headers=headers) as response:
        if response.status == 200:
            bibTex_citation = await response.text()
            return bibTex_citation
        else:
            print(f"Failed to retrieve BibTeX data for {arxiv_id}")
            return None

async def scrape_arxiv_for_machine_learning_papers(num_papers: int):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=cat:cs.LG&sortBy=submittedDate&sortOrder=descending&start=0&max_results={num_papers}"
    url = base_url + search_query

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                result:list[dict[str, Any]] = []
                soup = BeautifulSoup(await response.text(), "xml")

                entries = soup.find_all("entry")
                tasks = []

                for entry in entries:
                    title = entry.find("title").text.strip()
                    authors = [author.text for author in entry.find_all("name")]
                    pub_date = entry.find("published").text.strip()
                    source_tag = entry.find("category", scheme="http://arxiv.org/schemas/atom")
                    source = str(source_tag).split('term="')[1].split('"')[0]
                    arxiv_id = entry.find("id").text.split("/")[-1]
                    data_url = entry.find("id").text.strip()

                    task = asyncio.create_task(fetch_arxiv_paper(session, arxiv_id, headers))
                    tasks.append(task)

                    result.append(
                        {
                            "prompt": f"Obtain the title, author, publication year of the paper {title}, and generate the citation",
                            "title": title,
                            "authors": authors,
                            "publication_year": pub_date,
                            "source": source,
                            "url": data_url,
                            "bibTex_citation": None,
                        }
                    )

                bibTex_citations = await asyncio.gather(*tasks)

                for i, bibTex_citation in enumerate(bibTex_citations):
                    result[i]["bibTex_citation"] = bibTex_citation

                return result
            else:
                print(f"Failed to fetch data from arXiv. Status Code: {response.status}")
                return None
            
def main_sync(num_papers: int = 10, data_dir=None):
    asyncio.run(main_async(num_papers, data_dir))

async def main_async(num_papers: int = 10, data_dir=None):
    if data_dir is not None:
        data_dir = os.path.join(os.path.dirname(__file__), data_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        output_file = f"{data_dir}/machine_learning_papers.json"
    else:
        output_file = "machine_learning_papers.json"

    papers = await scrape_arxiv_for_machine_learning_papers(num_papers)
    if papers:
        with open(output_file, "w") as json_file:
            json.dump(papers, json_file, indent=4)
        print(f"{num_papers} papers scraped and saved into {output_file}")

app = typer.Typer()
app.command()(main_sync)

if __name__ == "__main__":
    app()
