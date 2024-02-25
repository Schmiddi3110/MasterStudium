import requests, time, random
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

class TableCrawler:
    """
    Class for crawling the matchdate data.
    """
    def __init__(self):
        return
        
    def get_season_md_dates(self, base_link):
        """
        Get a date-matchday mapping for each playdate in the season.
        Parameters:
            base_link: Base link for the table website
        Returns:
            date_mappings: mapping where each playdate gets assigned the matchday in this format: 2019-2020_1
        """
        #These parts of the url vary from season to season
        link_specs = {
            '2013-2014': ('se11976/', 'ro38889/'),
            '2014-2015': ('se15388/', 'ro47269/'),
            '2015-2016': ('se18336/', 'ro57041/'),
            '2016-2017': ('se20812/', 'ro63882/'),
            '2017-2018': ('se23906/', 'ro73072/'),
            '2018-2019': ('se28567/', 'ro92360/'),
            '2019-2020': ('se31723/', 'ro100673/'),
            '2020-2021': ('se35753/', 'ro109214/'),
            '2021-2022': ('se39227/', 'ro117247/'),
            '2022-2023': ('se45495/', 'ro132754/'),
            '2023-2024': ('se51884/', 'ro148505/'),
        }
        date_mappings = {}
        pre_year = 2013
        post_year = 2014
        german_date_format = "%d.%m.%Y"
        english_date_format = "%Y-%m-%d"

        while pre_year < 2024:
            #When the current year is reached, return
            if pre_year == 2024:
                return date_mappings

            #Build url for each matchday and print the currently crawled website
            for match_day in range(1, 35):
                url = base_link + link_specs[f'{pre_year}-{post_year}'][0] + f'{pre_year}-{post_year}/' + link_specs[f'{pre_year}-{post_year}'][1] + f'spieltag/md{match_day}/spiele-und-ergebnisse/'
                print(f'Crawling: {url}')
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, "html.parser")
                    match_date_containers = soup.find_all('div', class_='match-date')
                    season_md = f'{pre_year}-{post_year}_{match_day}'
                    for el in match_date_containers:
                        date = el.get_text()
                        german_datetime = datetime.strptime(date, german_date_format)
                        english_date = german_datetime.strftime(english_date_format)
                        date_mappings[english_date] = season_md
                except Exception as e:
                    print(f'Error: {e} while crawling {url}.')

                #Random pause between requests
                time.sleep(random.randint(2, 10))
            pre_year += 1
            post_year += 1
        return date_mappings
        