from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')


class keyword_extractor():
    def __init__(self):
        self.model = KeyBERT(model="bert-base-german-cased")

    def extract_keywords(self, title, text):
        seed_keywords = word_tokenize(title, language="german")
        text = title + ' \n\n ' + text
        top_keyphrases = self.model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words=stopwords.words('german'), top_n=2, seed_keywords=seed_keywords)

        top_keywords = self.model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=stopwords.words('german'), top_n=2, use_maxsum=True, seed_keywords=seed_keywords)

        keywords = set()

        for phrase in top_keyphrases:
            keywords.update(phrase[0].split(" "))

        for word in top_keywords:
            keywords.add(word[0])

        return list(keywords)
