import re
import pandas as pd
from string import punctuation

class Preprocessing :
    def __init__(self):
        print("Initializing preprocessing...")
        pass

    def processtext(self, text):
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

