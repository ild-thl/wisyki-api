from langchain.vectorstores import Chroma


class vectorsearcher():
    def __init__(self, embedding):
        persist_directory = 'data/esco_vectorstore'
        self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    def predict(self, doc, top_k):
        results = []
        
        docs = self.vectordb.similarity_search_with_score(doc, top_k)

        for doc in docs:
            results.append({'uri': doc[0].metadata["conceptUri"], 'title': doc[0].metadata["preferredLabel"], 'className': 'Skill', 'score': str(doc[1])})
        
        return {'searchterms': [], 'results': results}