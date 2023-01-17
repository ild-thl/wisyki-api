from keybert import KeyBERT
from nltk.corpus import stopwords

class keyword_extractor():
    def __init__(self):
        self.model = KeyBERT()
  
    def extract_keywords(self, title, description):
        text = title + " \n\n " + description
        top_keywords = self.model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=stopwords.words('german'), top_n=3)
        top_keyphrases = self.model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words=stopwords.words('german'), top_n=2)
        return top_keyphrases + top_keywords