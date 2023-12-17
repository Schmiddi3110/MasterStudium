import requests, re

from bs4 import BeautifulSoup

def crawl(url):
    html = requests.get(url)
    soup = BeautifulSoup(html.text, "html.parser")

    text = soup.get_text()
    links = []
    for link in soup.find_all('a'):
        if(link.get('href')) is not None:
            links.append(link.get('href'))
    return text, links

def crawl_with_links(url, pattern, n):
    main_text, links = crawl(url)
    list_texts = []

    for index in range(0, n, 1):
        link = links[index]
        if(pattern.match(link)):
            html = requests.get(link)
            soup = BeautifulSoup(html.text, "html.parser")
            text = soup.get_text()
            list_texts.append(text)
    return main_text, list_texts

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Hannibal"
    pattern = re.compile('^https://de.wikipedia.org/wiki')

    main_text, link_content = crawl_with_links(url, pattern, 1301)
    print(main_text)
    print(link_content)
    