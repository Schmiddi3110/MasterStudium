import re 
import json 
import requests 

from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

def crawl(url):
    return requests.get(url).text

def crawl_with_links(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    url_list = []
    for link in soup.find_all('link', href=True):
        url_list.append(link["href"])

    return url_list


if __name__ == "__main__":
    html = crawl('https://en.wikipedia.org/wiki/Hannibal')
    links = crawl_with_links('https://en.wikipedia.org/wiki/Hannibal')
    for link in links:
        if link.find("https://") != -1:
            additionalcrawl = crawl(link)

    print(additionalcrawl)