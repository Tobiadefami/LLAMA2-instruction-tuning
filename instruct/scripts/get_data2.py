import os
import json
import typer
import aiohttp
import asyncio
import tqdm
import re


def generate_apa_citation(author, publication_year, title, source=None, url=None):
    # Format the author(s) name for APA citation
    formatted_author = " and ".join(author.split(", "))

    if source is not None:
        if url is not None:
            return f"{formatted_author} ({publication_year}). {title}. {source}. {url}"
        else:
            return f"{formatted_author} ({publication_year}). {title}. {source}."
    else:
        if url is not None:
            return f"{formatted_author} ({publication_year}). {title}. Retrieved from {url}"
        else:
            return f"{formatted_author} ({publication_year}). {title}."


async def fetch_paper_data(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.text()
        else:
            print(f"Failed to fetch data from {url}. Status Code: {response.status}")
            return None

async def scrape_arxiv_for_machine_learning_papers_batch(session, num_papers, start_index):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=cat:cs.LG&sortBy=submittedDate&sortOrder=descending&start={start_index}&max_results={num_papers}"
    url = base_url + search_query

    response = await fetch_paper_data(session, url)
    if response:
        papers = response.split("<entry>")
        papers = papers[1:]  # The first element is not a paper

        tasks = []

        for paper in papers:
            title_match = re.search("<title>(.*?)</title>", paper, re.DOTALL)
            if title_match:
                title = title_match.group(1)
            else:
                title = "Title Not Found"

            author_matches = re.findall("<name>(.*?)</name>", paper, re.DOTALL)
            authors = [author.strip() for author in author_matches]

            pub_date_match = re.search("<published>(.*?)</published>", paper, re.DOTALL)
            if pub_date_match:
                published_date = pub_date_match.group(1)
                publication_year = published_date.split("-")[0]
            else:
                publication_year = "Year Not Found"

            source_match = re.search("<category term=\"(.*?)\" scheme=\"http://arxiv.org/schemas/atom\"/>", paper, re.DOTALL)
            if source_match:
                source = source_match.group(1)
            else:
                source = "Source Not Found"
            
            url_match = re.search("<id>(.*?)</id>", paper, re.DOTALL)
            if url_match:
                url = url_match.group(1)
            else:
                url = "URL Not Found"

            bibTex_citation = f"@article{{AuthorYear, \n\ttitle={{{title}}}, \n\tauthor={{{', '.join(authors)}}}, \n\tjournal=arXiv:XXXX.XXXX, \n\tyear={{{publication_year}}} \n}}"
            apa_citation = generate_apa_citation(', '.join(authors), publication_year, title, source=source, url=url)
            
            tasks.append({
                "prompt": f'Obtain the title, author, publication year of the paper {title}, and generate the citation',
                "title": title,
                "authors": authors,
                "publication_year":publication_year,
                "source": source,
                "url": url,
                "bibTex_citation": bibTex_citation,
                "apa_citation": apa_citation
            })

        return tasks
    else:
        return None

async def scrape_arxiv_for_machine_learning_papers(num_papers:int):
    tasks = []
    async with aiohttp.ClientSession() as session:
        for start_index in range(0, num_papers, 200):  # Fetch in batches of 200 papers
            batch_tasks = await scrape_arxiv_for_machine_learning_papers_batch(session, 200, start_index)
            if batch_tasks:
                tasks.extend(batch_tasks)

    return tasks

async def scrape_and_save_papers(num_papers, output_file):

    papers = await scrape_arxiv_for_machine_learning_papers(num_papers)
    if papers:
        with open(output_file, "w") as json_file:
            json.dump(papers, json_file, indent=4)
        print(f'{num_papers} papers scraped and saved into {output_file}')



def main(num_papers:int = 100000, data_dir=None):
    if data_dir is not None:
        data_dir = os.path.join(os.path.dirname(__file__), data_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        output_file = f"{data_dir}/machine_learning_papers.json"
    else:
        output_file = 'machine_learning_papers.json'

    asyncio.run(scrape_and_save_papers(num_papers, output_file))

# Example usage:
if __name__ == "__main__":
    typer.run(main)
