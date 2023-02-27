from keybert import KeyBERT
from nltk.corpus import stopwords
from flair.embeddings import TransformerDocumentEmbeddings


class keyword_extractor():
    def __init__(self):
        germancased = TransformerDocumentEmbeddings('bert-base-german-cased')
        self.model = KeyBERT(model=germancased)

    def extract_keywords(self, text):
        top_keyphrases = self.model.extract_keywords(text, keyphrase_ngram_range=(
            1, 4), stop_words=stopwords.words('german'), top_n=3)
        return top_keyphrases
