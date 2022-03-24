import requests
import bs4
import json
from tqdm.auto import tqdm


def append_one(papers, page, before=False):
    page_content = requests.get("https://dl.acm.org" + page).content.decode()
    soup = bs4.BeautifulSoup(page_content)
    try:
        content = {
            "link": "https://dl.acm.org" + page,
            "title": soup.find(attrs={"class": "citation__title"}).text,
            "authors": [author.text for author in soup.find_all(attrs={"class": "loa__author-name"})],
            "abstract": soup.find(attrs={"class": "abstractSection"}).text.strip(),
            "pages": soup.find(attrs={"class": "pageRange"}).text.split()[-1],
            "previous": soup.find(attrs={"class": "content-navigation__btn--pre"})["href"],
            "next": soup.find(attrs={"class": "content-navigation__btn--next"})["href"]
        }
    except:
        content = {
            "link": "https://dl.acm.org" + page,
            "title": soup.find(attrs={"class": "citation__title"}).text,
            "previous": soup.find(attrs={"class": "content-navigation__btn--pre"})["href"],
            "next": soup.find(attrs={"class": "content-navigation__btn--next"})["href"]
        }
    finally:
        if before:
            papers.insert(0, content)
        else:
            papers.append(content)


def iterate(papers, next_type, before, current_page):
    try:
        with tqdm() as pbar:
            while True:
                append_one(papers, current_page, before=before)
                last_idx = 0 if before else -1
                current_page = papers[last_idx][next_type]
                try:
                    print(f"{papers[last_idx]['pages']}: {papers[last_idx]['title']}")
                except:
                    print(f"Error: {papers[last_idx]['title']}")
                pbar.update(1)
    except:
        print("Finished!")


def clean_paper_sessions(papers):
    clean_papers = []
    last_session = "other"
    for paper in papers:
        if "pages" not in paper:
            last_session = paper["title"].split("Session details: ")[-1]
        else:
            paper["session"] = last_session
            clean_papers.append(paper)

    titles = [paper["title"] for paper in clean_papers]

    fixed_sessions_starts = {
        "Managing publications and bookmarks with BibSonomy": "Demonstration session",
        "A 3D hypermedia with biomedical stereoscopic images: from creation to exploration in virtual reality": \
            "Poster session",
        "Web 3.0: merging semantic web with social web": "Workshop session"
    }

    for title, session in fixed_sessions_starts.items():
        idx = titles.index(title)
        for i in range(idx, len(clean_papers)):
            clean_papers[i]["session"] = session
    
    return clean_papers


if __name__ == '__main__':
    papers = []
    start_page = "/doi/10.1145/1557914.1557966"

    iterate(papers, "previous", True, start_page)
    iterate(papers, "next", False, papers[-1]["next"])
    papers = clean_papers(papers)

    with open("data/papers.json", "w") as f:
        json.dump(clean_papers, f)