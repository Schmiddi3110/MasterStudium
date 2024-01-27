import nltk, re, utils
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm

nltk.download('punkt', download_dir='NLTK_Data/Punkt')
nltk.data.path.append('NLTK_Data/Punkt')
nltk.download('wordnet', download_dir='NLTK_Data/Wordnet')
nltk.data.path.append('NLTK_Data/Wordnet')
nltk.download('stopwords', download_dir='NLTK_Data/Stopwords')
nltk.data.path.append('NLTK_Data/Stopwords')

class Preprocessor:
    """
    Preprocessor class to preprocess data that has been obtained with the crawler class.
    Parameters:
        reg_mappings: Regular expressions with corresponding replacement strings. Any matches in the text will be replaced with that string
        sent_patt: Regex for characters that should be surrounded with whitespaces
    """
    def __init__(self, reg_mappings, sent_patt):
        self.reg_mappings = reg_mappings
        self.sent_patt = sent_patt

    def regex_preprocessing(self, filepath):
        """
        Iterates over given json data and removes the content that matches the regex list that was passed.
        Parameters:
            inputfile: Path to json data
        Returns:
            data: The data after removing content that matches the regex expressions
        """
        #Load crawling data
        data = utils.load_json(filepath)

        #Create a progress bar
        total_docs = len(data)
        progress_bar = tqdm(total=total_docs, desc="Regex cleaning documents: ", unit="doc", position=0, leave=True)

        for item in data:
            for key, value in item.items():
                curr_text = item[key]
                curr_text = curr_text.strip().lower()

                #Remove content that matches user defined regexp (email, time, links etc.)
                for mapping in self.reg_mappings:
                    curr_text = re.sub(mapping[1], mapping[0], curr_text)

                #Replace characters
                curr_text = curr_text.replace("\n", " ").replace("\"", "").replace("/", "").replace("“", "").replace("”", "").replace("*", "").replace("|", " ").replace("`", " ").replace("'", " ").replace(":_", " ").replace("_", " ").replace(" mrs. ", " mrs ").replace(" ms. ", " ms ").replace(" mr. ", " mr ").replace(" dr. ", " dr ").replace(" prof. ", " prof "). replace(" dr.-ing. ", " dr.-ing ").replace("dipl.-ing.","dipl.-ing").replace(" !", " .").replace(" ?", " .").replace("%", "").replace(" i.e .", " i.e ").replace(" e.g. "," e.g ").replace(" etc. "," etc ").replace(" a.m. "," am ").replace(" p.m. "," pm ").replace(" ggf. "," ggf ").replace(" ca. "," ca ").replace(" bzw. "," bzw ").replace("&", "").replace("$", "").replace("(", "").replace(")", "").replace("{", "").replace("}", "").replace(" m.sc.", " m.sc ").replace(" m.a. "," m.a ").replace(" b.sc. ", " b.sc ").replace(" b.a. ", " b.a ").replace("[", "").replace("]", "").replace("§", "").replace("=", "").replace(",", "").replace("\u00AE", "").replace("\u00a9", "").replace("\u2122", "").replace("\uFFFD", "").replace("\u0001", "").replace("\u0002", "").replace("\u0003", "").replace("\u0004", "").replace("\u0005", "").replace("\u0006", "").replace("\u0007", "").replace("\u0008", "").replace("\uF8FB", "").replace("\uF8F8", "").replace("\uF8EB", "").replace("\uF8EC", "").replace("\uF8F6", "").replace("\uF8F7", "").replace("\uF8ED", "").replace("\uF8FA", "").replace("\uF8F9", "").replace("\uF8F0", "").replace("\uF8EF", "").replace("\uF8EE", "").replace("α", "").replace("ε", "").replace("µ", "").replace("π", "").replace("ψ", "").replace("τ", "").replace("∑", "").replace("δ", "").replace("ω", "").replace("ρ", "").replace("υ", "").replace("η", "").replace("γ", "").replace("ν", "").replace("ξ", "").replace("λ", "").replace("∞", "").replace("∆", "").replace("=", "").replace("<", "").replace(">", "").replace("≈", "").replace("⋅", "").replace("√", "").replace("•", "").replace("⎟", "").replace("⎝", "").replace("⎠", "").replace("⎛", "").replace("⎞", "").replace("⎜",  "").replace("✩", "").replace("↓", "").replace("^", "").replace("...", "").replace("≠", "")
        
                all_pts = self.sent_patt.finditer(curr_text)

                #Set whitspace before and after every , . ! ? : ; in case sentences aren't interpolated correctly
                for m in all_pts:
                    curr_text = curr_text.replace(m.group(), m.group()[0]+' '+m.group()[1]+' ', 1)
                    curr_text = curr_text.replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ')
                    
                item[key] = curr_text
            progress_bar.update()
            
        #Save the cleaned data to a file
        utils.save_as_json(data, "data_and_preprocessing/crawler_result_regex_cleaned.json")
            
    def rem_non_english_words(self, filepath):
        #Load the data
        data = utils.load_json(filepath)
        
        #Load the vocabulary from a textfile
        vocab_file = open("british-english", "r")
        vocab = set(re.sub("[^\w]", " ",  vocab_file.read()).split())
        vocab_file.close()

        #Create a progress bar
        total_docs = len(data)
        progress_bar = tqdm(total=total_docs, desc="Removing non english words from docs: ", unit="doc", position=0, leave=True)

        for item in data:
            for key, doc in item.items():
                sentences = sent_tokenize(doc)
                filter_words = []
                for sent in sentences:  
                    word_tokens = word_tokenize(sent)
                    for word in word_tokens:
                        if word not in vocab and word.lower() not in vocab and word.isalpha():
                            filter_words.append(word)
                            
                pattern = r'\b(?:' + '|'.join(re.escape(word) for word in filter_words) + r')\b'
                text = re.sub(pattern, '', doc)
                item[key] = text
                progress_bar.update()
                
        #Save the cleaned data to a file
        utils.save_as_json(data, "data_and_preprocessing/crawler_result_lang_cleaned.json")
        
    def lemmatize(self, filepath):
        """
        Lemmatizes over the given json data and performs sentence tokenization
        Parameters:
            inputfile: Path to json data
        Returns:
            data: the data lemmatized and sentence tokenized
        """
        #Open file
        data = utils.load_json(filepath)
        lemmatizer = WordNetLemmatizer()
        
        #Create a progress bar
        total_docs = len(data)
        progress_bar = tqdm(total=total_docs, desc="Lemmatizing documents: ", unit="doc", position=0, leave=True)
        
        for item in data:
            for key, value in item.items():
                #Generate sentence tokens -> for each sentence generate word tokens
                sentences = sent_tokenize(value, 'english')
                sentence_words = []
                for sentence in sentences:
                    sentence_words.append(word_tokenize(sentence))
                
                #lemmatize all non stopword words and add them to dictionary 
                item[key] = []
                for sentence in sentence_words:
                    words = []
                    for word in sentence:
                        if word in stopwords.words('english'):
                            continue
                            
                        words.append(lemmatizer.lemmatize(word))
                    string = ' '.join(words)
                    if len(string) <= 5:
                        continue
                        
                    regex = re.compile(r"([\s!:–?=@-]{2,}\s)")
                    string = re.sub(regex, " ", string)

                    item[key].append(string)
            progress_bar.update()

        #Save lemmatized data to a file
        utils.save_as_json(data, "data_and_preprocessing/crawler_result_lemmatized.json")
        