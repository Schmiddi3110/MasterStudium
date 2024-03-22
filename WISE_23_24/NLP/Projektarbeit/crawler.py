import requests, re, time, validators, os, fitz, utils
from lingua import Language, LanguageDetectorBuilder #To detect language of a website
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class Crawler:
    """
    Crawler class to crawl a website and extract all links and text content.
    Parameters:
        start_url: The URL to start crawling from
        url_pattern: A regex pattern to filter out all unwanted links 
        exclude_objects: Html objects/classes that should be filtered out during the crawling process
        exclude_divs_w_class: Divs with special class that should be filtered out 
        languages: Languages that should be taken into account when detecting the language of a website
        link_thresh: The maximum depth to crawl (amount of valid documents)
        delay_seconds: The delay between each request
        main_tag: Main tag to use when text is extracted. If a main_tag is passed, only text inside a div containing this tag will be extracted.
    """
    def __init__(self, start_url, url_pattern, exclude_objects, exclude_obj_w_class, languages, link_thresh=5, delay_seconds=0.5, main_tag=None):
        self.start_url = start_url
        self.url_pattern = url_pattern
        self.exclude_objects = exclude_objects
        self.exclude_obj_w_class = exclude_obj_w_class
        self.link_thresh = link_thresh
        self.languages = languages
        self.delay_seconds = delay_seconds
        self.main_tag = main_tag
        
        self.depth = 0
        self.visited_links = []
        self.link_tree = {}
        self.link_contents = []
        self.urls_to_crawl = [start_url]

    def __add_hostname(self, link, prefix):
        """
        Checks if a link is a valid url, and
        adds a prefix to this link, if it is missing
        Parameters:
            link: The link to check and add the hostname to
            prefix: The prefix to add to the link if it is invalid
        """
        prefix_pattern = re.compile(prefix)
        if prefix_pattern.search(link) is None:
            #Only add the hostname if the link is invalid 
            if validators.url(link):
                return link
            else:
                return prefix + link
        else:
            return link

    def __parse_pdf(self, folder, url, content):
        """
        Downloads a PDF file, saves it into a folder in the current working directory,
        then extracts the textual content.
        Patameters: 
            folder: The location the pdf should be saved in (if it doesn't exist, folder will be created)
            url: The url the pdf was requested from
            content: The content of the pdf request
        Returns:
            text: The textual content of the PDF file
            path: The path of the PDF file
        """
        #Create a folder for the saved pdfs if not already present
        if not os.path.exists(folder):
            os.makedirs(folder)

        #Extract the file name of the pdf from the url
        file_name = url.rsplit('/',1)[1]
        path = os.path.join(folder, file_name)

        #Write the content from the request to a pdf file and save it locally
        with open(path, 'wb') as pdf_file:
            pdf_file.write(content)

        #Read pdf text and return it
        pdf_text = ""
        with fitz.open(path) as doc:
            for page in doc:
                pdf_text += page.get_text().strip()

        return pdf_text, path

    def __rem_tags_and_objects(self, soup):
        """
        Removes classes and objects (text) specified in self.exclude objects from a given soup object.
        Paramters:
            soup: The soup object containing website html and text
        Returns:
            soup: The cleaned soup object 
        """         
        #Remove general objects like buttons
        for ex_obj in self.exclude_objects:        
            objects = soup.find_all(ex_obj)
            for obj in objects:
                obj.decompose()

        #Remove obj-class mappings 
        for obj_w_class in self.exclude_obj_w_class:
            objects = soup.find_all(obj_w_class[0], class_=obj_w_class[1])
            for obj in objects:
                obj.decompose()

        return soup

    def __is_doc_lang(self, text, lang, confidence=0.95):
        """
        Checks if the corresponding document (text) is written in the desired language.
        Paramters:
            text: The text to be checked
            lang: The desired language
            conf: The minimum confidence for passing the language check
        Returns:
            boolean: True if the check is passed, false otherwise
        """
        #Init language detector
        detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        
        conf_value_lang = 0
        detected_lang = detector.detect_language_of(text)
        confidence_values = detector.compute_language_confidence_values(text)

        #Get the confidence Value for English
        for conf in confidence_values:
            if conf.language == lang:
                conf_value_lang = conf.value

        if detected_lang == lang and conf_value_lang >= confidence:
            return True
        else:
            return False

    def __extract_urls(self, soup, url):
        """
        Extracts the urls from a soup object and adds them to the crawlers visit queue.
        Parameters:
            soup: Bs4 object of the website content
            url: website url
        """
        #Extract links and link information
        links = [link.get('href') for link in soup.find_all('a') if link.get('href') is not None]
        scheme = urlparse(url).scheme
        hostname = urlparse(url).hostname
        prefix = scheme + "://" + hostname
                
        #Add hostname to the link if needed (this will only happen for internal links, not external links!) and filter irrelevant links
        hostname_urls = list(set([self.__add_hostname(link, prefix) for link in links]))
        checked_urls = [link for link in hostname_urls if self.url_pattern.search(link) is not None and link is not url]
        not_visited = [url for url in checked_urls if url not in self.visited_links]
        self.urls_to_crawl = self.urls_to_crawl + not_visited

        #remove possible duplicates
        self.urls_to_crawl = list(set(self.urls_to_crawl))
        self.link_tree[self.depth] = {'url': url, 'valid_urls': not_visited}
    
    def scrape_url(self, url):
        """
        Scrapes content from a single URL. If the url links to a pdf, the content of the pdf is extracted 
        as text and returned, just like the html content for a normal website. 
        For other types of sites (like pictures etc.), no Text is returned.
        Parameters:
            url: The URL to crawl
        Returns:
            text: The text/pdf content of the URL
        """
        #Send a request to the provided url and extract the content type
        message = None
        response = requests.get(url)
        content_type = response.headers.get('content-type')
        response.raise_for_status()
        
        #If the url links to a pdf, download it, read and then return its content and source url
        if 'application/pdf' in content_type:   
            pdf_text, pdf_path = self.__parse_pdf("crawled_pdfs", url, response.content)

            #Check if language is english   
            if self.__is_doc_lang(pdf_text, Language.ENGLISH):
                return pdf_text
                
            #Remove the pdf file, as its content is not being extracted anyways
            else:
                os.remove(pdf_path)
                message = ("[INFO]: Language of pdf content is not english, skipping this file.")
                print(message)
                utils.write_message(message, "crawling_log.txt")
                
                return None
                
        #If the url links to an ordinary website, extract and return its text
        elif 'text/html' in content_type:
            soup = BeautifulSoup(response.text, "html.parser")  #Format text for readability
            self.__extract_urls(soup, url)
            soup = self.__rem_tags_and_objects(soup)

            #Find section with main content to exclude header and footer directly if main_tag is specified
            if self.main_tag is not None:
                page_content_div = soup.find('div', id=self.main_tag)
                if page_content_div:
                    #Directly remove unnecessary linebreaks and whitespaces. Separate nested text with whitespace
                    main_content = page_content_div.get_text(separator=' ', strip=True)
                            
                    if self.__is_doc_lang(main_content, Language.ENGLISH):
                        return main_content
                    else:
                        message = ("[INFO]: Language of website content is not english, skipping this site.")
                        print(message)
                        utils.write_message(message, "crawling_log.txt")
                        
                        return None
                else:
                    message = ("[INFO]: No main content available, skipping this site.")
                    print(message)
                    utils.write_message(message, "crawling_log.txt")
                    
                    return None
            #If no main_tag is specified, just return the raw, unfiltered text and the corresponding links
            else:
                return soup.get_text()
            
        #If no text can be found, just return links 
        else:
            soup = BeautifulSoup(response.text)  #Format text for readability
            self.__extract_urls(soup, url)
            message = ("[INFO]: Website content is no text, skipping this site")
            print(message)
            utils.write_message(message, "crawling_log.txt")
            
            return None
    
    def crawl_urls(self):
        """
        Starts the crawling process on the start_url that is passed to the crawler object.
        """
        while self.urls_to_crawl and self.depth < self.link_thresh:
            url = self.urls_to_crawl.pop(0)
            iteration_log = f"[{self.depth}] Currently crawling: {url}"
            print(f"[{self.depth}] Currently crawling: {url} with {len(self.urls_to_crawl)} urls left")
            utils.write_message(iteration_log, "crawling_log.txt")
            time.sleep(self.delay_seconds)
            self.visited_links.append(url)
            
            try:    
                text = self.scrape_url(url)
    
                #If a result is returned, update the corresponding content
                if text is not None:
                    self.link_contents.append({url: text})
                    self.depth = self.depth + 1
    
                    #Save the crawled data 
                    utils.save_as_json(self.link_contents, "data_and_preprocessing/crawler_result.json")

            #If crawling fails, save corresponding error and the url it failed on 
            except Exception as e:
                iteration_error_log = f"[INFO]: Something went wrong while crawling {url}: {e}"
                print(iteration_error_log)
                utils.write_message(iteration_error_log + '\n', "crawling_log.txt")

        utils.save_as_json(self.link_tree, "data_and_preprocessing/link_tree.json")
        