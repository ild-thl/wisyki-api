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
            docs = self.vectordb.similarity_search_with_score(doc, top_k*5)
        else:
            docs = self.vectordb.similarity_search_with_score(doc, top_k)

        for doc in docs:
            # If filterconcepts are set, exlcude all terms that are not a child of either of the cocepts.
            if len(filterconcepts) and doc[0].metadata["broaderHierarchyConcepts"]:
                broaderconcepts = json.loads(doc[0].metadata["broaderHierarchyConcepts"])
                is_part_of_concept = False
                for filterconcept in filterconcepts:
                    for broaderconcept in broaderconcepts:
                        if filterconcept in broaderconcept["uri"]:
                            is_part_of_concept = True
                            break
                
                if not is_part_of_concept:
                    continue
                
            results.append({
                'uri': doc[0].metadata["conceptUri"],
                'title': doc[0].metadata["preferredLabel"],
                'className': 'Skill', 'score': str(doc[1]),
                # 'p': is_part_of_concept,
                # 'broaderConcepts': doc[0].metadata["broaderHierarchyConcepts"]
            })
        
        return {'searchterms': [], 'results': results[:top_k]}