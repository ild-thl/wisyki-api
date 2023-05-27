from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings


class vectorsearcher():
    def __init__(self):
        persist_directory = 'data/esco_vectorstore'
        embedding = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            embed_instruction="Represent the document for retrieval: ",
            query_instruction="Represent the query for retrieval: "
        )
        self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    def predict(self, doc, top_k):
        results = []
        
        docs = self.vectordb.similarity_search_with_score(doc, top_k)

        for doc in docs:
            results.append({'uri': doc[0].metadata["conceptUri"], 'title': doc[0].metadata["preferredLabel"], 'className': 'Skill', 'score': str(doc[1])})
        
        return {'searchterms': [], 'results': results}