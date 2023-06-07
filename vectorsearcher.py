from langchain.vectorstores import Chroma
import os
import json


class vectorsearcher():
    def __init__(self, embedding):
        dir = os.path.dirname(__file__)
        persist_directory = dir + '/data/esco_vectorstore'
        self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    def predict(self, doc, top_k, filterconcepts):
        results = []

        if len(filterconcepts):
            skills = self.vectordb.similarity_search_with_score(doc, top_k*5)
        else:
            skills = self.vectordb.similarity_search_with_score(doc, top_k)

        for skill in skills:
            # If filterconcepts are set, exlcude all terms that are not a child of either of the cocepts.
            if len(filterconcepts) and skill[0].metadata["broaderHierarchyConcepts"]:
                broaderconcepts = json.loads(skill[0].metadata["broaderHierarchyConcepts"])
                is_part_of_concept = False
                for broaderconcept in broaderconcepts:
                    if broaderconcept["uri"] in filterconcepts:
                        is_part_of_concept = True
                        break

                if not is_part_of_concept:
                    continue

            results.append({
                'uri': skill[0].metadata["conceptUri"],
                'title': skill[0].metadata["preferredLabel"],
                'className': 'Skill', 'score': str(skill[1]),
                # 'broaderConcepts': skill[0].metadata["broaderHierarchyConcepts"]
            })

        return {'searchterms': [], 'results': results[:top_k]}