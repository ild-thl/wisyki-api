import requests
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keyword_extractor import keyword_extractor
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

class esco_predictor():
    def __init__(self):
        pass

    def predict(self, searchterms, extract_keywords, filterconcepts, min_relevancy, exclude_irrelevant, doc=None):
        if doc and extract_keywords:
            bertmodel = keyword_extractor()

            if 'keywords' in searchterms:
                doc_keywords = bertmodel.extract_keywords(
                    doc, ", ".join(searchterms['keywords']))
                searchterms['keywords'] += doc_keywords
            else:
                doc_keywords = bertmodel.extract_keywords(doc)
                searchterms['keywords'] = doc_keywords

        results = []
        if 'keywords' in searchterms:
            for searchterm in searchterms['keywords']:
                results += self.search_esco(searchterm, filterconcepts)

        if 'occupations' in searchterms:
            for occupationuri in searchterms['occupations']:
                results += self.get_occupation_skills(occupationuri)

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

    def search_esco(self, searchterm, filterconcepts):
        skills = []
        params = {
            'text': searchterm,
            'limit': 5,
            'language': 'de',
            'full': False,
            'isInScheme': 'http://data.europa.eu/esco/concept-scheme/member-skills, http://data.europa.eu/esco/concept-scheme/member-occupations'
        }

        if len(filterconcepts) == 0:
            params['isInScheme'] += ', http://data.europa.eu/esco/concept-scheme/skills-hierarchy'

        url = "https://ec.europa.eu/esco/api/search?" + \
            requests.compat.urlencode(params)
        response = requests.get(url)

        if response.ok:
            results = json.loads(response.text)
            for result in results['_embedded']['results']:
                skill = {
                    'uri': result['uri'], 'title': result['title'], 'className': result['className']}

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

    def get_related_skills(self, uri, get_siblings=True):
        skills = []
        params = {
            'uri': uri,
            'language': 'de'
        }

        url = "https://ec.europa.eu/esco/api/resource/skill?" + \
            requests.compat.urlencode(params)
        response = requests.get(url)

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

            if get_siblings:
                for skill in broaderSkills:
                    skills += self.get_related_skills(skill['uri'], False)

        return skills

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

    # Define a custom tokenizer that uses lemmatization to count synonyms and different versions of the same words as one and the same token

    def my_tokenizer(self, text):
        tokens = word_tokenize(text, 'german')
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token not in stopwords.words('german')]

    def sum_scores(self, keyphrase, token_scores):
        tokens = self.my_tokenizer(keyphrase)
        score = 0.0
        for token_score in token_scores:
            if token_score in tokens:
                score += token_scores[token_score].sum()

        if score > 0:
            score = score / len(tokens)

        return score

    def sort_by_relevancy(self, skills, min_relevancy, exclude_irrelevant, doc):
        vectorizer = TfidfVectorizer(tokenizer=self.my_tokenizer)
        vectors = vectorizer.fit_transform([doc])
        feature_names = vectorizer.get_feature_names_out()
        dense = vectors.todense()
        denselist = dense.tolist()

        df = pd.DataFrame(denselist, columns=feature_names)
        lowest_token_score = df.sum().min()

        results = []

        # Sum up the scores for the tokens of each keyphrase and sort the results by score.
        for skill in skills:
            score = self.sum_scores(skill['title'], df)
            if exclude_irrelevant:
                if (min_relevancy and score > min_relevancy) or (not min_relevancy and score > lowest_token_score/2):
                    skill['score'] = score
                    results.append(skill)
            else:
                skill['score'] = score
                results.append(skill)

        return sorted(results, key=lambda x: x['score'], reverse=True)
