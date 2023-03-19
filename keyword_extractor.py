from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')
stop_words = ['telelearning', 'optional', 'zielgruppe', 'lerninhalte', 'persönliches', 'individuell', 'praktikum', 'inhalte', 'unterschiedliche', 'kurse', 'einführung',
              'zunächst', 'zeigt', 'bescheinigung', 'teilnahmebescheinigung', 'lehrveranstaltung', 'kursinhalte', 'gibt', 'hwk', 'abschluss', 'teil', 'training', 'prüfungsvorbereitung']
stop_words.extend(stopwords.words('german'))


class keyword_extractor():
    def __init__(self):
        # self.model = KeyBERT(model='bert-base-german-cased')
        self.model = KeyBERT()

    def extract_keywords(self, text, seed=None):
        seed_keywords = None
        if seed:
            seed_keywords = word_tokenize(seed, language='german')
            text = seed + ' \n\n ' + text

        top_keyphrases = self.model.extract_keywords(text, keyphrase_ngram_range=(
            1, 3), stop_words=stop_words, top_n=3, seed_keywords=seed_keywords)
        top_keywords = self.model.extract_keywords(text, keyphrase_ngram_range=(
            1, 1), stop_words=stop_words, top_n=3, use_maxsum=True, seed_keywords=seed_keywords)

        keywords = set()

        for phrase in top_keyphrases:
            if phrase[1] >= .2:
                keywords.update(phrase[0].split(" "))

        for word in top_keywords:
            if word[1] >= .2:
                keywords.add(word[0])

        return list(keywords)
