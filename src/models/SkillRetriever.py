from .SkillPrediction import SkillPrediction as Prediction
from .get_chat_llm import get_llm
import json
import re
import math
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage
import warnings
from langchain_core import vectorstores

# Ignore UserWarning from langchain_core.vectorstores
warnings.filterwarnings("ignore", category=UserWarning, module=vectorstores.__name__)


class SkillRetriever:
    def __init__(self, embedding, reranker, skilldb, domains, request):
        """
        Initialize the SkillRetriever object.

        Parameters:
        - embedding: The embedding model.
        - reranker: The reranker model.
        - skilldb: A vector database containing the skills.
        - domains: A set of domains.
        - request: The request object containing the request parameters.
        """
        self.embedding = embedding
        self.reranker = reranker
        self.taxonomies = request.taxonomies
        self.skilldb = skilldb
        self.domains = domains
        self.doc = request.doc
        self.los = request.los
        self.prerequisites = request.prerequisites
        self.validated_skills = request.skills
        self.validated_skill_uris = [skill.uri for skill in self.validated_skills]
        self.valid_skill_labels = [
            skill.title for skill in self.validated_skills if skill.valid
        ]
        self.filterconcepts = request.filterconcepts
        self.top_k = request.top_k
        self.strict = request.strict
        self.trusted_score = request.trusted_score
        self.temperature = request.temperature
        self.use_llm = request.use_llm
        self.llm_validation = request.llm_validation
        self.do_rerank = request.rerank
        self.openai_api_key = request.openai_api_key
        self.mistral_api_key = request.mistral_api_key
        self.score_cutoff = request.score_cutoff
        self.domain_specific_score_cutoff = request.domain_specific_score_cutoff
        self.target = ""

        # Initialize the used_models list
        self.used_models = []

    async def predict(self, target="learning_outcomes", get_sources=False) -> tuple:
        """
        Predicts the top-k skills based on the learning outcomes.

        Returns:
            tuple: A tuple containing the learning outcomes and the predicted skills.
        """
        self.target = target

        if target == "learning_outcomes":
            learningoutcomes = await self.get_learning_outcomes()
        else:  # target == "prerequisites"
            learningoutcomes = await self.get_prerequisites()

        if len(learningoutcomes) == 0:
            return self.los, []

        # Embed the learning outcomes.
        embedded_doc = self.embedding.embed_documents([learningoutcomes])

        # Do similarity search for skills.
        predictions = self.get_top_similar_skills(self.los)

        # Define artificial threshholds for relevancy by identifying where the similarity rating decreases the fastest.
        if not self.llm_validation:
            predictions = self.applyDynamicThreshold(predictions)

        if target == "learning_outcomes":
            # Finetune predictions based on the known skills.
            predictions = self.finetune_on_validated_skills(predictions)

        # Filter out domain specific skills if the domain is not mentioned in the learning outcomes.
        predictions = self.filter_domain(predictions, learningoutcomes)

        # Reduce amount of predictions before performance hungry validation.
        predictions = predictions[: int(self.top_k * 1.5)]

        # Validate predictions.
        if self.llm_validation:
            predictions = await self.validate_with_llm(predictions)

        # Sort predictions by score.
        predictions = sorted(predictions, key=lambda x: x.score, reverse=True)

        # Some scores might have become negative due to penalties. Normalize scores.
        if len(predictions) > 0:
        #     min_score = predictions[-1].score
        #     if min_score < 0:
        #         for prediction in predictions:
        #             prediction.score -= min_score

            max_score = predictions[0].score
            if max_score > 1:
                for prediction in predictions:
                    prediction.score /= max_score

        # Remove predictions with a score higher than the score_cutoff.
        if self.score_cutoff > 0 and self.score_cutoff < 1:
            predictions = [
                prediction
                for prediction in predictions
                if prediction.score > self.score_cutoff
            ]

        if get_sources:
            source_doc = learningoutcomes
            if self.doc and len(self.doc) > 0:
                source_doc = self.doc

            # Get source ngrams for predictions.
            predictions = self.get_source_ngrams(predictions, source_doc)

        return self.los, predictions[: self.top_k]

    def get_source_ngrams(self, predictions: list, source_doc: str) -> list:
        # Calculate 5-grams once and store them in the dictionary.
        self.ngrams = self.get_ngrams(source_doc, 8)

        for prediction in predictions:
            prediction_embedding = self.embedding.embed_documents([prediction.title])[0]
            best_ngram = self.get_best_ngram(prediction_embedding)
            if best_ngram:
                prediction.source = best_ngram

        return predictions

    def get_best_ngram(self, doc_embedding: list) -> str:
        max_similarity = -1
        best_ngram = None
        current_ngrams = self.ngrams

        while True:
            more_similar_found = False
            for ngram in current_ngrams:
                # Get the embedding of the current ngram.
                ngram_embedding = ngram["embedding"]
                similarity = cosine_similarity([doc_embedding], [ngram_embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_ngram = ngram
                    more_similar_found = True

            # If no more similar ngram was found, or the best ngram is a single word, or there are no subngrams that are more similar, break the loop.
            if (
                not more_similar_found
                or len(best_ngram["ngram"].split()) == 1
                or len(best_ngram["subngrams"]) == 0
            ):
                break

            # If the best ngram is a single word, or there are no subngrams that are more similar, break the loop.
            if (
                len(best_ngram["ngram"].split()) == 1
                or len(best_ngram["subngrams"]) == 0
            ):
                break

            # Otherwise, update the current ngrams to the subngrams of the best ngram and continue the loop.
            current_ngrams = best_ngram["subngrams"]
            if len(current_ngrams) == 0:
                # Cache generated subngrams in self.ngram
                best_ngram["subngrams"] = self.get_ngrams(
                    best_ngram["ngram"], len(best_ngram["ngram"].split()) - 1
                )
                current_ngrams = best_ngram["subngrams"]

        return best_ngram["ngram"]

    def get_ngrams(self, document: str, n: int) -> list:
        ngrams = []
        lines = document.split("\n")
        for line in lines:
            words = line.split()
            for i in range(len(words)):
                for j in range(i + 1, min(i + n + 1, len(words) + 1)):
                    ngram = " ".join(words[i:j])
                    ngrams.append(ngram)

        # Create a dictionary with ngrams as keys and their embeddings as values and an empty list of subngrams.
        ngram_embeddings = self.embedding.embed_documents(ngrams)
        ngrams = [
            {"ngram": ngram, "embedding": embedding, "subngrams": []}
            for ngram, embedding in zip(ngrams, ngram_embeddings)
        ]
        return ngrams

    def filter_domain(self, predictions: list, learning_outcomes: str) -> list:
        """
        Filters out predictions based on language set, programming languages, other domains and learning outcomes.
        This method assumes that domain specific skills can only be valid if the domain is metioned by word in the original learning outcomes.

        Args:
        predictions (list): List of prediction objects.
        learning_outcomes (str): String of learning outcomes.

        Returns:
        list: Filtered list of predictions.
        """
        filtered = []

        learning_outcomes_lower = learning_outcomes.lower()

        # Get domain sepecific skills.
        for prediction in predictions:
            is_domain_specific = False
            relevant_domain = ""
            # if domain in prediction title.
            for domain in self.domains:
                prediction_title_lower = prediction.title.lower()
                prediction_title_words = (
                    prediction_title_lower.split()
                    if " " in prediction_title_lower
                    else [prediction_title_lower]
                )

                if len(domain) < 4 and " " not in domain:
                    # Short domains have to be present as whole word.
                    if domain in prediction_title_words:
                        is_domain_specific = True
                        relevant_domain = domain
                        break
                else:
                    # Longer domains are allowed to be part of a word.
                    if domain in prediction_title_lower:
                        is_domain_specific = True
                        relevant_domain = domain
                        break

            if is_domain_specific:
                # Filter out skills, if the relevant domain is not mentioned in the learning outcomes.
                if not relevant_domain in learning_outcomes_lower:
                    continue
                # Or filter out skills, if the score is below the domain specific score cutoff.
                if prediction.score < self.domain_specific_score_cutoff:
                    continue

            filtered.append(prediction)

        return filtered

    async def get_prerequisites(self) -> str:
        """
        Prepares the prerequisites for further processing.

        Returns:
            tuple: A tuple containing the prepared prerequisites and the embedded document.
        """
        prerequisites = ""
        self.los = self.prerequisites
        if len(self.prerequisites) > 0:
            prerequisites = "\n".join(self.prerequisites)
        elif self.use_llm and self.doc and len(self.doc) > 0:
            prerequisites = await self.extract_prerequisites(self.doc)
            self.los = prerequisites.split("\n")

        # Remove empty lines from self.los.
        self.los = [line for line in self.los if line]

        return prerequisites

    async def get_learning_outcomes(self) -> str:
        """
        Prepares the learning outcomes for further processing.

        Returns:
            tuple: A tuple containing the prepared learning outcomes and the embedded document.
        """
        if len(self.los) > 0:
            learningoutcomes = "\n".join(self.los)
        elif self.use_llm:
            learningoutcomes = await self.extract_learning_outcomes(self.doc)
            self.los = learningoutcomes.split("\n")
        else:
            learningoutcomes = self.doc
            self.los.append(learningoutcomes)

        # Add valid skills to learning outcomes to improve the quality of the embeddings.
        learningoutcomes = "\n".join(self.valid_skill_labels) + "\n" + learningoutcomes
        self.los.extend(self.valid_skill_labels)

        # Remove empty lines from self.los.
        self.los = [line for line in self.los if line]

        return learningoutcomes

    async def extract_prerequisites(self, doc: str) -> str:
        """
        Extracts the prerequisites from a given document.

        Args:
            doc (str): The document from which to extract the prerequisites.

        Returns:
            str: The extracted prerequisites.
        """

        # Create messages for chat.
        messages = [
            SystemMessage(
                content=(
                    "Als Redakteur identifizierst du explizit genannte Voraussetzungen oder benötigte Vorerfahrungen in Bildungsdokumenten. Das heißt du benennst Fähigkeiten und Wissen, die Bildungsinteressierte bereits haben sollten bevor sie das Bildungsangebot wahrnehmen."
                )
            ),
            HumanMessagePromptTemplate.from_template(
                "Folgendes Bildungsdokument liegt vor:"
                "{course}"
                ""
                'Liste die im vorrangegangenen Text explizit genannten Voraussetzungen oder benötigten Vorerfahrungen auf, jeweils in einer neuen Zeile beginnend mit einem "-".'
                "Nutze kurze, einfache Sprache und BLOOM-Verben für Fähigkeiten, Nomen für Wissen."
                "Bennene nicht was nicht benötigt wird, sondern nur was benötigt wird."
                "Wenn keine Vorraussetzungen benannt werden, antworte mit 'Keine Voraussetzungen benannt'."
                ""
                "Folgende Voraussetzungen beschreiben das Dokument am besten:"
            ),
        ]

        prerequisites, used_model = await self.get_chatresponse(
            messages, {"course": doc[:3500]}, use_most_competent_llm=False
        )
        self.add_model_stats(used_model, "Extract prerequisites from document.")

        # Check for 'Keine Voraussetzungen benannt' and return empty string if found.
        if "Keine Voraussetzungen benannt" in prerequisites:
            return ""

        # Try to get only lines starting with "-". If not possible, return the whole text.
        prereqs = "\n".join(re.findall(r"^- .*", prerequisites, flags=re.MULTILINE))
        if not prereqs:
            prereqs = prerequisites
        # Remove list decorations using regular expressions
        prereqs = re.sub(
            r"^ *[\d.-]+ *|^ *\* *|^ *- *", "", prereqs, flags=re.MULTILINE
        )

        return prerequisites

    async def extract_learning_outcomes(self, doc: str) -> str:
        """
        Extracts the learning outcomes from a given document.

        Args:
            doc (str): The document from which to extract the learning outcomes.

        Returns:
            str: The extracted learning outcomes.
        """

        # Create messages for chat.
        messages = [
            SystemMessage(
                content=(
                    "Als Redakteur identifizierst du explizit genannte Lernziele in Bildungsangeboten. Das heißt du benennst die Fähigkeiten und das Wissen, das Teilnehmer vermutlich erlangen werden oder erstreben."
                )
            ),
            HumanMessagePromptTemplate.from_template(
                "Folgendes Bildungsdokument liegt vor:"
                "{course}"
                ""
                'Liste die im vorhergegangenen Text explizit genannten Lernziele auf, jeweils in einer neuen Zeile beginnend mit einem "-".'
                "Nutze kurze, einfache Sprache und BLOOM-Verben für Fähigkeiten, Nomen für Wissen."
                "Wenn keine Lernziele benannt werden, antworte mit 'Keine Lernziele benannt'."
                "Der relevante Kontext der Kompetenz sollte in der Bennneung der einzelnen Lernziele deutlich werden."
                ""
                "Folgende Kompetenzen und Lernziele beschreiben das Dokument am Besten:"
            ),
        ]

        learningoutcomes, used_model = await self.get_chatresponse(
            messages, {"course": doc[:3500]}, use_most_competent_llm=False
        )
        self.add_model_stats(used_model, "Extract learning outcomes from document.")

        # Check for 'Keine Lernziele benannt' and return empty string if found.
        if "Keine Lernziele benannt" in learningoutcomes:
            return ""

        # Try to get only lines starting with "-". If not possible, return the whole text.
        los = "\n".join(re.findall(r"^- .*", learningoutcomes, flags=re.MULTILINE))
        if not los:
            los = learningoutcomes

        # Remove list decorations using regular expressions
        los = re.sub(r"^ *[\d.-]+ *|^ *\* *|^ *- *", "", los, flags=re.MULTILINE)

        return los

    def cluster_documents(self, documents: List[str]) -> List[List[str]]:
        """
        Clusters a list of documents based on their similarity.

        Args:
            documents (List[str]): A list of documents to be clustered.

        Returns:
            List[List[str]]: A list of clusters, where each cluster is a list of documents.

        """
        # If there are less than two documents, skip the clustering.
        if len(documents) < 2:
            return [documents]

        # Compute the word embeddings for each learning outcome.
        lo_embeddings = self.embedding.embed_documents(documents)

        # If there are exactly 2 documents, use cosine similarity to determine the similarity between them.
        if len(documents) == 2:
            similarity = cosine_similarity(lo_embeddings)
            if similarity[0][1] > 0.5:
                return [documents]

            return [[documents[0]], [documents[1]]]

        # Use a hierarchical clustering algorithm to group the documents.
        clustering = AgglomerativeClustering().fit(lo_embeddings)

        # Determine the optimal number of clusters.
        silhouette_scores = []
        for n_clusters in range(2, len(documents)):
            clustering.n_clusters = n_clusters
            labels = clustering.labels_
            score = silhouette_score(lo_embeddings, labels)
            silhouette_scores.append(score)
        optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

        # Assign each document to a cluster.
        clustering.n_clusters = optimal_n_clusters
        labels = clustering.labels_

        # Create subsets of documents based on their cluster assignments.
        clusters = []
        for cluster_id in range(optimal_n_clusters):
            cluster = [
                doc for doc, label in zip(documents, labels) if label == cluster_id
            ]
            clusters.append(cluster)

        return clusters

    def get_title_predictions(self, learning_outcomes: list) -> list:
        """
        Retrieves title predictions based on learning outcomes.

        Args:
            learning_outcomes (list): A list of learning outcomes.

        Returns:
            list: A list of title predictions.

        """
        title = ""
        # Check for patterns like "Titel: Bergführer\n", "Kurstitel: Bergführer\n", "Course title: Bergführer\n"
        match = re.match(
            r"^(Titel|Kurstitel|Course title|Course Title|Course Name|Kursname|Kursname|Kursname):*\s*(.*\b) *(\n|Kursbeschreibung|Beschreibung|Description)",
            self.doc,
        )
        if match:
            title = match.group(2).strip()
            # Split the title at the first special vahracter like - or ( and other special characters and remove trailing whitespaces.
            title = re.split(r"[^a-zA-Z0-9äöüÄÖÜß\s]+", title)[0].strip()
        # If no match, check if the document starts with a single word followed by line breaks
        else:
            match = re.match(r"^(\b)\n+", self.doc)

            if match:
                title = match.group(1).strip()

        if not title:
            return []

        document = "\n".join(learning_outcomes)

        # Do similarity search for the title.
        if self.taxonomies and len(self.taxonomies) > 0:
            filter = {"taxonomy": {"$in": self.taxonomies}}
        else:
            filter = None
        top_docs = self.skilldb.similarity_search_with_relevance_scores(
            document,
            min(self.top_k, 20),
            filter=filter,
            where_document={"$contains": title},
        )

        candidates = [self.create_prediction(skill) for skill in top_docs]

        # Filter out predictions that are already known or duplicates or not part of the filterconcepts.
        candidates = self.filter_predictions(candidates, sort=True)

        if self.do_rerank:
            candidates = self.rerank(candidates, document)

        return candidates

    def get_top_similar_skills(self, learning_outcomes: list) -> list:
        """
        Retrieves the top similar skills based on the given learning outcomes.

        Args:
            learning_outcomes (list): A list of learning outcomes.

        Returns:
            list: A list of top similar skills.
        """

        similar_skills = []

        if self.doc and len(self.doc) > 0:
            # Get title predictions based on the document.
            title_predictions = self.get_title_predictions(learning_outcomes)
            similar_skills.extend(title_predictions)

        # Escape early if there are no learning outcomes.
        if len(learning_outcomes) > 0:

            clusters = self.cluster_documents(learning_outcomes)

            # Do similarity search for each cluster.
            self.add_model_stats(
                "pascalhuerten/multilingual-e5-base-course-skill-tuned",
                "Embed learning outcomes for similarity search against skill database.",
            )
            if self.taxonomies and len(self.taxonomies) > 0:
                filter = {"taxonomy": {"$in": self.taxonomies}}
            else:
                filter = None
            for cluster in clusters:
                cluster_doc = "\n".join(cluster)
                similar_cluster_skills = (
                    self.skilldb.similarity_search_with_relevance_scores(
                        cluster_doc,  # Query
                        min(self.top_k, 20)
                        + len(self.validated_skills),  # Number of results
                        filter=filter,  # Filter metadata
                    )
                )
                # Convert skill documents to predictions.
                predictions = [
                    self.create_prediction(skill) for skill in similar_cluster_skills
                ]

                # Filter out predictions that are already known or duplicates or not part of the filterconcepts.
                predictions = self.filter_predictions(predictions, sort=True)

                if self.do_rerank:
                    predictions = self.rerank(predictions, cluster_doc)

                similar_skills.extend(predictions)

        # Filter out predictions that are already known or duplicates or not part of the filterconcepts.
        similar_skills = self.filter_predictions(similar_skills, sort=True)

        # Rerank the predictions based on all learning outcomes.
        if self.do_rerank:
            # similar_skills = self.rerank(similar_skills, "\n".join(learning_outcomes))
            self.add_model_stats(
                "pascalhuerten/bge-reranker-base-course-skill-tuned",
                "Reranking similarity search results based on learning outcomes.",
            )

        return similar_skills

    async def get_chatresponse(
        self, messages: list, context: dict, use_most_competent_llm=False
    ) -> Tuple[str, str]:
        """
        Retrieves a chat response based on the given messages and context.

        Args:
            messages (list): A list of messages exchanged in the chat.
            context (dict): The context of the chat.
            use_most_competent_llm (bool): Flag indicating whether to use the most competent language model.

        Returns:
            str: The chat response generated by the model.
        """
        prompt = ChatPromptTemplate.from_messages(messages)

        model, model_name = get_llm(
            self.openai_api_key,
            self.mistral_api_key,
            self.temperature,
            use_most_competent_llm,
        )

        chain = prompt | model | StrOutputParser()

        chatresponse = chain.invoke(context)

        chatresponse = chatresponse.replace("ASSISTANT: ", "").strip()

        return chatresponse, model_name

    async def validate_with_llm(self, predictions: list) -> list:
        """
        Validates the predictions using a language model.

        Args:
            predictions (list): A list of prediction dictionaries.

        Returns:
            list: A list of validated prediction dictionaries, or the original predictions if strict mode is not enabled.
        """
        # Get skill labels.
        skilllabels = [prediction.title for prediction in predictions]
        # Get course description as context for chat.
        context = ""
        if self.doc and len(self.doc) > 0:
            context = self.doc
        else:
            context = "\n".join(self.los)

        # Create messages for chat.
        if self.target == "learning_outcomes":
            messages = [
                SystemMessage(
                    content=(
                        "Du bist ein Redakteur einer Weiterbildungsplatform. Deine Aufgabe ist es zu prüfen, welche der vorgeschlagenen Kompetenzen zu dem angegebenen Bildungsdokument passen."
                        "Berücksichtige dabei folgende Fragestellungen."
                        "Passen die vorgeschlagenen Kompetenzen zum Thema oder zu einzelnen benannten Lernzielen?"
                        "Sind die Kompetenzen zu allgemein oder zu spezifisch, um auf das Dokument angewandt zu werden?"
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    "Folgendes Dokument ist gegeben:"
                    "{course}"
                    ""
                    "Potenzielle Kompetenzen:"
                    "{skills}"
                    ""
                    "Erzeuge eine optimierte Liste auschließlich derer Kompetenzen, die gut das Dokument beschreiben oder darin benannt werden."
                    "Behalte den genauen Wortlaut der vorgeschlagenen Kompetenzen bei."
                    "Nenne eine Kompetenz pro Zeile. Die Antwort sollte nur die Kompetenzen selbst enthalten, ohne Einleitungen oder zusätzliche Worte."
                    ""
                    "Optimierte Kompetenzen:"
                ),
            ]
        else:  # target == "prerequisites"
            messages = [
                SystemMessage(
                    content=(
                        "Du bist ein Redakteur einer Weiterbildungsplatform. Deine Aufgabe ist es zu prüfen, welche der vorgeschlagenen Kompetenzen von dem Bildungsangebot benötigt oder empfohlen sind."
                        "Berücksichtige dabei folgende Fragestellungen."
                        "Passen die vorgeschlagenen Vorerfahrungen thematisch zu den Voraussetzungen des Kurses oder werden explizit benannt?"
                        "Differenziere zwischen Voraussetzungen, die explizit benannt werden und solchen, die implizit benötigt werden."
                        "Sind die Voraussetzungen zu allgemein oder zu spezifisch, um auf das Dokument angewandt zu werden?"
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    "Folgendes Dokument ist gegeben:"
                    "{course}"
                    ""
                    "Potenziell passende Vorrausetzungen:"
                    "{skills}"
                    ""
                    "Erzeuge eine optimierte Liste auschließlich derer Voraussetzungen, die gut das Dokument beschreiben oder darin explizit benannt werden."
                    "Behalte dabei den genauen Wortlaut der potenziellen Voraussetzungen bei."
                    "Nenne eine Voraussetzung pro Zeile. Die Antwort sollte nur die Voraussetzung selbst enthalten, ohne Einleitungen oder zusätzliche Worte."
                    ""
                    "Optimierte Voraussetzungen:"
                ),
            ]

        # Get chat response from language model.
        chatresponse, used_model = await self.get_chatresponse(
            messages,
            {"course": context[:3500], "skills": "\n".join(skilllabels)},
            use_most_competent_llm=True,
        )
        self.add_model_stats(used_model, "Validate predictions with language model.")

        # Split chatresponse into lines, every line is a valid skill.
        lines = chatresponse.split("\n")
        lines = [line.strip() for line in lines]
        # strip 1. 2. etc or - or * from start of line
        lines = [
            re.sub(r"^ *[\d.-]+ *|^ *\* *|^ *- *", "", line, flags=re.MULTILINE)
            for line in lines
        ]
        # remove empty lines
        validskills = [line for line in lines if line]

        # Validate predictions.
        validated = []
        for i in range(len(predictions)):
            fit = predictions[i].title in validskills
            predictions[i].fit = fit

            # If strict mode is enabled, only keep predictions that are validated.
            if not fit and self.strict > 0:
                continue

            validated.append(predictions[i])

        return validated

    def add_model_stats(self, model_name: str, reason: str):
        self.used_models.append({"model": model_name, "reason": reason})

    def rerank(self, predictions: list, leraningoutcomes: str) -> list:
        """
        Reranks the predictions based on the scores computed using the reranker model.

        Args:
            predictions (list): List of prediction dictionaries.
            leraningoutcomes (str): The document to be used for reranking.

        Returns:
            list: Reranked predictions with updated scores.
        """
        if len(predictions) == 0:
            return predictions

        # Compute scores using the reranker model.
        pairs = [(leraningoutcomes, prediction.title) for prediction in predictions]
        scores = self.reranker.compute_score(pairs)
        # Convert scores to list if necessary.
        if not isinstance(scores, list):
            scores = [scores]

        # Reranked predictions with positive scores.
        validated = []
        for prediction, score in zip(predictions, scores):
            # If the score is positive, the prediction is probably relevant/valid.
            fit = score > 0
            # Normalize score to be between 0 and 1.
            max_score = 13.8
            score = max(min(score, max_score), -max_score)
            score = (score + max_score) / (max_score * 2)
            prediction.score = score
            
            prediction.fit = fit

            # If strict mode is enabled, only keep predictions that are validated.
            # if self.strict > 0 and not fit:
            #     continue

            validated.append(prediction)

        return validated

    def filter_predictions(self, predictions: list, sort=False) -> list:
        """
        Filters the given predictions based on certain criteria.

        Args:
            predictions (list): The list of predictions to filter.
            sort (bool, optional): Whether to sort the predictions by score. Defaults to False.

        Returns:
            list: The filtered predictions.
        """
        if sort:
            # Sort predictions by score. This will assure that for duplicate predictions the one with the better score will be kept.
            predictions = sorted(predictions, key=lambda x: x.score, reverse=True)

        # Filter out duplicate predictions and predictions that are already known and not part of the filterconcepts.
        seen = set()
        filtered = []
        for prediction in predictions:
            if (
                prediction.uri not in seen
                and not self.is_known_skill(prediction)
                and self.is_part_of_concept(prediction)
            ):
                seen.add(prediction.uri)
                filtered.append(prediction)
        return filtered

    def is_known_skill(self, skill: dict) -> bool:
        """
        Checks if a skill is known or validated.

        Args:
            skill (dict): The skill to be checked.

        Returns:
            bool: True if the skill is known, False otherwise.
        """
        return skill.uri in self.validated_skill_uris

    def is_part_of_concept(self, skill: dict) -> bool:
        """
        Checks if a skill is part of a concept.

        Args:
            skill (dict): The skill to check.

        Returns:
            bool: True if the skill is part of a concept, False otherwise.
        """
        # Do not filter for broader concepts if the target is prerequisites.
        if self.target == "prerequisites":
            return True
        # Broader concepts are only available for ESCO skills. If the skill is not an ESCO skill, return True.
        if len(self.filterconcepts) == 0 or "ESCO" not in skill.taxonomy:
            return True

        # Check if the skill is part of a concept that is part of the filterconcepts.
        if "broaderConcepts" in skill.metadata:
            for broaderconcept in skill.metadata["broaderConcepts"]:
                if broaderconcept in self.filterconcepts:
                    return True
        return False

    def create_prediction(self, skill) -> Prediction:
        """
        Creates a prediction object based on the given skill and score.

        Args:
            skill (str): The skill for which the prediction is being created.
            score (float): The score associated with the prediction.

        Returns:
            Prediction: The created prediction object.
        """

        # Create prediction object based on the skill taxonomy.
        if skill[0].metadata["taxonomy"] == "ESCO":
            return Prediction.from_esco(skill)
        elif skill[0].metadata["taxonomy"] == "GRETA":
            return Prediction.from_greta(skill)
        else:
            return Prediction.from_other(skill)

    def applyDynamicThreshold(self, predictions: list) -> list:
        """
        Refactored method to apply dynamic thresholding to the predictions based on the specified strictness level,
        ensuring that the second largest gap is always after the first largest gap, and so on.

        Args:
            predictions (list): A list of prediction dictionaries, each containing a "score" key.

        Returns:
            list: The filtered predictions based on the dynamic thresholding.
        """
        # Return all predictions if strictness is 0 or there are 2 or fewer predictions
        if self.strict == 0 or len(predictions) <= 2:
            return predictions

        def get_predictions_up_to_nth_largest_gap(predictions, n, last_gap_index=0):
            if n <= 0 or not predictions:
                return predictions

            # Calculate gaps and their indices after the last identified gap
            gaps_with_indices = [(predictions[i].score - predictions[i + 1].score, i) 
                                for i in range(last_gap_index, len(predictions) - 1)]

            # Find the largest gap after the last identified gap
            largest_gap = max(gaps_with_indices, key=lambda x: x[0], default=(None, None))

            # Check if largest_gap[1] is None, which means no more gaps were found
            if largest_gap[1] is None:
                # Return all predictions if no more gaps are found
                return predictions
            
            if n == 1:
                # If this is the first largest gap or no more gaps, return predictions up to this gap
                print(len(predictions[:largest_gap[1] + 1]))
                return predictions[:largest_gap[1] + 1]

            # For finding subsequent gaps, update the last_gap_index to the index of the current largest gap
            # and recursively call the function to find the next largest gap
            return get_predictions_up_to_nth_largest_gap(predictions, n-1, largest_gap[1] + 1)

        # Apply thresholding based on strictness level
        if self.strict == 3:
            predictions = get_predictions_up_to_nth_largest_gap(predictions, 1)
        elif self.strict == 2:
            predictions = get_predictions_up_to_nth_largest_gap(predictions, 2)
        elif self.strict == 1:
            predictions = get_predictions_up_to_nth_largest_gap(predictions, 3)


        return predictions

    def finetune_on_validated_skills(self, predictions: list) -> list:
        """
        Finetunes the predictions based on validated skills.

        Args:
            predictions (list): List of predictions to be finetuned.

        Returns:
            list: Finetuned predictions.
        """
        if len(self.validated_skills) == 0:
            return predictions
        # Predictions based on validated skills.
        validSkillUris = [skill.uri for skill in self.validated_skills if skill.valid]
        validSkillLabels = "\n".join(
            [skill.title for skill in self.validated_skills if skill.valid]
        )

        # Do Vector Search to find most similar skills.
        if self.taxonomies and len(self.taxonomies) > 0:
            filter = {"taxonomy": {"$in": self.taxonomies}}
        else:
            filter = None
        valid_docs = self.skilldb.similarity_search_with_relevance_scores(
            validSkillLabels,
            10,
            filter=filter,
        )
        # Create predictions for similar skills and filter out the current skill.
        similarToValidSkills = [
            self.create_prediction(valid_doc)
            for valid_doc in valid_docs
            if valid_doc[0] not in validSkillUris
        ]
        similarToValidSkills = self.filter_predictions(similarToValidSkills)

        # Add skills that are similar to valid skills and reward them with a higher score.
        for similarValidSkill in similarToValidSkills:
            found = False
            for prediction in predictions:
                if prediction.uri == similarValidSkill.uri:
                    penalty = ((similarValidSkill.score) ** 4) * 0.3
                    prediction.penalty += penalty
                    prediction.score += penalty
                    found = True
                    break
            if not found:
                predictions.append(similarValidSkill)

        invalidSkillUris = [
            skill.uri for skill in self.validated_skills if not skill.valid
        ]
        invalidSkillLabels = "\n".join(
            [skill.title for skill in self.validated_skills if not skill.valid]
        )
        # Do Vector Search to find most similar skills.
        if self.taxonomies and len(self.taxonomies) > 0:
            filter = {"taxonomy": {"$in": self.taxonomies}}
        else:
            filter = None
        invalid_docs = self.skilldb.similarity_search_with_relevance_scores(
            invalidSkillLabels, 10, filter=filter
        )
        # Create predictions for similar skills and filter out the current skill.
        similarToInvalidSkills = [
            self.create_prediction(invalid_doc)
            for invalid_doc in invalid_docs
            if invalid_doc[0] not in invalidSkillUris
        ]
        similarToInvalidSkills = self.filter_predictions(similarToInvalidSkills)
        # Penalty for predictions that are similar to invalid skills.
        for similarInvalidSkill in similarToInvalidSkills:
            for prediction in predictions:
                if prediction.uri == similarInvalidSkill.uri:
                    # The lower the score, the higher the penalty.
                    penalty = -((similarInvalidSkill.score) ** 4) * 0.5
                    prediction.penalty += penalty
                    prediction.score += penalty
                    break

        return self.filter_predictions(predictions)
