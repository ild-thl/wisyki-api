import json
from langchain.chat_models  import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from sklearn.metrics.pairwise import cosine_similarity


class chatsearcher():
    def __init__(self, vectordb, embedding):
        self.vectordb = vectordb
        self.embedding = embedding
        self.systemmessage = "Du bist ein KI-Assistent der Kursbeschreibungen analysiert, um die wichtigsten Lernziele zu identifizieren. Du antwortest ausschlißlich mit einer Liste von Lernzielen."
    

    def predict(self, doc, top_k, strict, trusted_score, temperature, openai_api_key, known_skills, filterconcepts):
        chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature, openai_api_key=openai_api_key)
        
        messages = [
            SystemMessage(content=self.systemmessage),
            HumanMessage(content=doc)
        ]

        response = chat(messages).content

        predictions = []

        embedded_doc = self.embedding.embed_documents([response])

        known_skill_uris = [skill["uri"] for skill in known_skills]
        known_skill_labels = [skill["title"] for skill in known_skills]
        doc = " ".join(known_skill_labels) + " " + response

        if len(filterconcepts):
            relevant_skills = self.vectordb.similarity_search_with_score(doc, top_k*5)
        else:
            relevant_skills = self.vectordb.similarity_search_with_score(doc, top_k*2)

        for relevant_skill in relevant_skills:
            if relevant_skill[0].metadata["conceptUri"] in known_skill_uris:
                continue
            # If filterconcepts are set, exlcude all terms that are not a child of either of the cocepts.
            if len(filterconcepts) and relevant_skill[0].metadata["broaderHierarchyConcepts"]:
                broaderconcepts = json.loads(relevant_skill[0].metadata["broaderHierarchyConcepts"])
                is_part_of_concept = False
                for broaderconcept in broaderconcepts:
                    if broaderconcept["uri"] in filterconcepts:
                        is_part_of_concept = True
                        break

                if not is_part_of_concept:
                    continue

            predictions.append({
                'uri': relevant_skill[0].metadata["conceptUri"],
                'title': relevant_skill[0].metadata["preferredLabel"],
                'className': 'Skill',
                'score': relevant_skill[1],
                # 'broaderConcepts': relevant_skill[0].metadata["broaderHierarchyConcepts"]
            })

        # Define artificial threshholds for relevancy by identifying where the similarity rating decreases the fastest.
        if strict > 0 and len(predictions):
            # Identify the biggest and second biggest gap between the skills with scores higher than 0.2.
            gaps = []
            for i in range(len(predictions) - 1):
                gaps.append(predictions[i + 1]["score"] - predictions[i]["score"])

            # Get the idecies of the two largest gaps.
            max_gap_skill_index = gaps.index(max(gaps)) + 1
            if strict == 3:
                predictions = predictions[:max_gap_skill_index]
            elif strict <= 2:
                max_gap = 0
                max_gap_skill_index_2 = 0
                for i in range(max_gap_skill_index+1, len(predictions) - 1):
                    if predictions[i]["score"] < trusted_score:
                        continue
                    gap = predictions[i + 1]["score"] - predictions[i]["score"]
                    if gap > max_gap:
                        max_gap = gap
                        max_gap_skill_index_2 = i

                if strict == 1:
                    max_gap = 0
                    max_gap_skill_index_3 = 0
                    for i in range(max_gap_skill_index_2+1, len(predictions) - 1):
                        if predictions[i]["score"] < trusted_score:
                            continue
                        gap = predictions[i + 1]["score"] - predictions[i]["score"]
                        if gap > max_gap:
                            max_gap = gap
                            max_gap_skill_index_3 = i

                    predictions = predictions[:max_gap_skill_index_3+1]
                else:
                    predictions = predictions[:max_gap_skill_index_2+1]

        # Predictions base on the known skills.
        for known_skill in known_skills:
            predictions += self.predict_for_skill(known_skill, embedded_doc)

        # Remove knwon skills and duplicates from predictions.
        seen = []
        todelete = []
        for i in range(len(predictions)):
            if predictions[i]['uri'] in seen or predictions[i]['uri'] in known_skill_uris:
                todelete.append(i)
            else:
                seen.append(predictions[i]['uri'])
        
        results = []
        for i in range(len(predictions)):
            if i not in todelete:
                results.append(predictions[i])

                
        results = sorted(results, key=lambda x: x['score'], reverse=False)

        return {'searchterms': [], 'results': results[:top_k]}
    
    def predict_for_skill(self, skill, embedded_doc):
        predictions = []
        relevant_skills = self.vectordb.similarity_search_with_score(skill["title"], 3)
        for relevant_skill in relevant_skills:
            if relevant_skill[0].metadata["conceptUri"] == skill["uri"]:
                continue

            embedded_skill = self.embedding.embed_documents([relevant_skill[0].metadata["preferredLabel"]])
            similarities = cosine_similarity(embedded_doc, embedded_skill)

            predictions.append({
                'uri': relevant_skill[0].metadata["conceptUri"],
                'title': relevant_skill[0].metadata["preferredLabel"],
                'className': 'Skill',
                'score': 1-similarities[0][0].item(),
                # 'broaderConcepts': relevant_skill[0].metadata["broaderHierarchyConcepts"]
            })

        return predictions

