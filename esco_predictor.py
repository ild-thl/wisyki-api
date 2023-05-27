from keyword_extractor import keyword_extractor
import requests
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from InstructorEmbedding import INSTRUCTOR


class esco_predictor():
    def __init__(self, model):
        self.model = model

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
                    if not re.match(r'.*esco/(skill/S\d+\.\d+\.\d+)|(isced-f/\d{4})', result['uri']):
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

    # Sort a set of skills based on their titles semantic similarity to a document.
    def sort_by_relevancy(self, skills, min_relevancy, exclude_irrelevant, doc):
        results = []

        embeddings_a = self.model.embed_documents([doc])
        embeddings_b = self.model.embed_documents([re.sub(r'\([^)]*\)', '', skill['title']) for skill in skills])
        similarities = cosine_similarity(embeddings_a,embeddings_b)

        # Add the similarities to the key "score" of every skill.
        for i in range(len(skills)):
            if exclude_irrelevant and min_relevancy:
                if similarities[0][i].item() < min_relevancy:
                    continue

            skill = skills[i]
            skill['score'] = similarities[0][i].item()
            results.append(skill)

        # # Sort the list of strings based on the similarity score.
        return sorted(results, key=lambda x: x['score'], reverse=True)
