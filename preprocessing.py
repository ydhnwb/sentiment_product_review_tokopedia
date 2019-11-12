import re
import pandas as pd
from string import punctuation
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Preprocessing :
    def __init__(self):
        print("Initializing preprocessing...")
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()
        pass

    def processtext(self, text):
        text = text.lower()
        text = re.sub(r'\&\w*;', '', text)
        text = re.sub('@[^\s]+','',text)
        text = re.sub(r'\$\w*', '', text)
        text = text.lower()
        text = re.sub(r'https?:\/\/.*\/\w*', '', text)
        text = re.sub(r'#\w*', '', text)
        text = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', text)
        text = re.sub(r'\b\w{1,2}\b', '', text)
        text = re.sub(r'\s\s+', ' ', text)
        text = text.lstrip(' ')
        text = ''.join(c for c in text if c <= '\uFFFF')
        return text

    def stem(self, text):
        text = self.stemmer.stem(text)
        return text

    def remove_stopwords(self, param):
        f = "id_stopwords.txt"
        with open(f, 'r') as my_stopwords:
            stopwords_list = my_stopwords.read()
            list = param.split()
            index = []
            i = 0
            d = ""
            while i < len(list):
                if list[i] not in stopwords_list:
                    index.append(i)
                i += 1
            for k in index:
                d += list[k]+" "
            #s = ' '.join(list)
            return d.strip()
