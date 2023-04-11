from keyword_extractor import keyword_extractor
import requests
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
stop_words = ['telelearning', 'optional', 'zielgruppe', 'lerninhalte', 'persönliches', 'individuell', 'praktikum', 'inhalte', 'unterschiedliche', 'kurse', 'kurs', 'einführung',
              'zunächst', 'zeigt', 'bescheinigung', 'teilnahmebescheinigung', 'lehrveranstaltung', 'veranstaltung', 'kursinhalte', 'gibt', 'hwk', 'abschluss', 'teil', 'training',
              'prüfungsvorbereitung', 'ausbildung', 'umschulung', 'bildung', 'schulung', 'konstenlos', 'ideal', '##en', '##ung', 'en', 'ung', 'seminar', 'online', 'zertifikat', 'tätigkeit',
              'grundlagen', 'basis']
stop_words.extend(stopwords.words('german'))


class esco_predictor():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-german-cased')
        pass

    def predict(self, searchterms, extract_keywords, schemes, filterconcepts,
                min_relevancy, exclude_irrelevant, doc=None):
        results = []
        # If a doc exists and extract_keywords is True, extract the top keywords from the doc using nlp.
        if doc and extract_keywords:
            bertmodel = keyword_extractor()

            if 'keywords' in searchterms:
                # If there are keywords use those as a seed for the ai keyword extraction.
                doc_keywords = bertmodel.extract_keywords(
                    doc, ", ".join(searchterms['keywords']))
                searchterms['keywords'] += doc_keywords
            else:
                doc_keywords = bertmodel.extract_keywords(doc)
                searchterms['keywords'] = doc_keywords

        # Search for ESCO Terms based on the given keywords.
        if 'keywords' in searchterms:
            for searchterm in searchterms['keywords']:
                results += self.search_esco(searchterm,
                                            schemes, filterconcepts)

        # Get the optional and mandatory skills of the given esco occupations.
        if 'occupations' in searchterms:
            for occupationuri in searchterms['occupations']:
                results += self.get_occupation_skills(occupationuri)

        # Get the related skills of the given esco skills.
        if 'skills' in searchterms:
            for skilluri in searchterms['skills']:
                related_skills = self.get_related_skills(skilluri)
                related_skills = [
                    related for related in related_skills if related['uri'] != skilluri]
                results += related_skills

        results = [dict(t) for t in set([tuple(d.items()) for d in results])]

        if doc:
            results = self.sort_by_relevancy(
                results, min_relevancy, exclude_irrelevant, doc)

        return {'searchterms': searchterms, 'results': results}

    # Search esco for skills, skill-concepts and occupation based on a given searchterm.
    def search_esco(self, searchterm, schemes, filterconcepts):
        skills = []
        params = {
            'text': searchterm,
            'limit': 5,
            'language': 'de',
            'full': False,
            'isInScheme': schemes
        }

        url = "https://ec.europa.eu/esco/api/search?" + \
            requests.compat.urlencode(params)
        response = requests.get(url)

        if response.ok:
            results = json.loads(response.text)
            for result in results['_embedded']['results']:
                skill = {
                    'uri': result['uri'], 'title': result['title'], 'className': result['className']}

                # If filterconcepts are set, exlcude all terms that are not a child of either of the cocepts.
                if len(filterconcepts):
                    if result['className'] == 'Concept' or 'broaderHierarchyConcept' not in result:
                        continue

                    is_part_of_concept = False
                    for filterconcept in filterconcepts:
                        if filterconcept in result['broaderHierarchyConcept']:
                            is_part_of_concept = True

                    if not is_part_of_concept:
                        continue

                if result['className'] == 'Concept':
                    if not re.match(r'.*esco/(skill/S\d\.\d\.\d)|(isced-f/\d{4})', result['uri']):
                        continue

                skills.append(skill)

        return skills

    # Get the related skills of the given esco skills. (broaderSkills, broaderHierarchy, narrowerSkills, siblingSkills if get_siblings is True)

    def get_related_skills(self, uri, get_siblings=True):
        skills = []

        # Get skill details from esco api.
        params = {
            'uri': uri,
            'language': 'de'
        }
        url = "https://ec.europa.eu/esco/api/resource/skill?" + \
            requests.compat.urlencode(params)
        response = requests.get(url)

        # If request is successful, get the related skills described in the response body.
        if response.ok:
            results = json.loads(response.text)
            broaderSkills = []
            if 'broaderHierarchyConcept' in results['_links']:
                broaderSkills = results['_links']['broaderHierarchyConcept']
                for result in broaderSkills:
                    skills.append(
                        {'uri': result['uri'], 'title': result['title'], 'className': 'Skill'})

            if 'broaderSkill' in results['_links']:
                broaderSkills = results['_links']['broaderSkill']
                for result in broaderSkills:
                    skills.append(
                        {'uri': result['uri'], 'title': result['title'], 'className': 'Skill'})

            if 'narrowerSkill' in results['_links']:
                for result in results['_links']['narrowerSkill']:
                    skills.append(
                        {'uri': result['uri'], 'title': result['title'], 'className': 'Skill'})

            # Get sibling skills, by calling the same method on the next broader skill or concept.
            # Sibling skills are the narrower Skills of the next broader skill/concept.
            if get_siblings:
                for skill in broaderSkills:
                    skills += self.get_related_skills(skill['uri'], False)

        return skills

    # Get the optional and mandatory skills of a given esco occupation.
    def get_occupation_skills(self, uri):
        skills = []
        params = {
            'uri': uri,
            'language': 'de'
        }

        url = "https://ec.europa.eu/esco/api/resource/occupation?" + \
            requests.compat.urlencode(params)
        response = requests.get(url)

        if response.ok:
            results = json.loads(response.text)
            for result in results['_links']['hasEssentialSkill']:
                skills.append(
                    {'uri': result['uri'], 'title': result['title'], 'className': 'Skill'})

            for result in results['_links']['hasOptionalSkill']:
                skills.append(
                    {'uri': result['uri'], 'title': result['title'], 'className': 'Skill'})

        return skills

    # Define a custom tokenizer that uses lemmatization to count synonyms and different versions of the same words as one and the same token and splits composits into subwords.
    def lemma_tokenizer(self, text):
        # Replace hyphens with spaces.
        text = re.sub('[^a-zA-Z0-9\n\.]', ' ', text)
        # Get full word tokens.
        tokens = word_tokenize(text, 'german')
        # Filter out german stopwords and tokens with less than 2 alphabetic characters.
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha()
                           and token.lower() not in stop_words]
        # Define a custom tokenizer that uses lemmatization to count synonyms and different versions of the same words as one and the same token and splits composits into subwords.
        return filtered_tokens

    def bert_tokenizer(self, text):
        # Get subword tokens.
        subword_tokens = self.tokenizer.tokenize(text)
        filtered_tokens = [token.lower() for token in subword_tokens if token.lower(
        ) not in stop_words and bool(re.search(r'[a-zA-Z].*[a-zA-Z]', token))]
        return filtered_tokens

    def combined_tokenizer(self, text):
        return self.lemma_tokenizer(text) + self.bert_tokenizer(text)

    # Sum up the scores of every token in a keyphrase.
    def sum_scores(self, keyphrase, token_scores):
        lemma_tokens = self.lemma_tokenizer(keyphrase)
        bert_tokens = self.bert_tokenizer(keyphrase)
        score = 0.0
        for token_score in token_scores:
            if token_score in lemma_tokens or token_score in bert_tokens:
                score += token_scores[token_score].sum()

        if score > 0 and len(lemma_tokens):
            score = score / len(lemma_tokens)

        return score

    # Sort a set of skills based on their titles semantic similarity to a document.
    def sort_by_relevancy(self, skills, min_relevancy, exclude_irrelevant, doc):
        # Create relevancy scores for every token in the document.
        vectorizer = TfidfVectorizer(tokenizer=self.combined_tokenizer)
        vectors = vectorizer.fit_transform([doc])
        feature_names = vectorizer.get_feature_names_out()
        dense = vectors.todense()
        denselist = dense.tolist()

        df = pd.DataFrame(denselist, columns=feature_names)
        lowest_token_score = df.sum().min()

        results = []

        # Sum up the scores for the tokens of each skill title and sort the results by score.
        for skill in skills:
            score = self.sum_scores(skill['title'], df)
            # If exclude_irrelevant is True do not append skills with a low relevancy score.
            if exclude_irrelevant:
                # Exclude skills with a relevancy lower than min_relevancy or if min_relevancy is not set,
                # exclude all skills by default which scores are lower that half of the lowest known token score.
                if (min_relevancy and score > min_relevancy) or (not min_relevancy and score >= lowest_token_score / 2.0):
                    skill['score'] = score
                    results.append(skill)
            else:
                skill['score'] = score
                results.append(skill)

        # Sort the results by score.
        return sorted(results, key=lambda x: x['score'], reverse=True)
