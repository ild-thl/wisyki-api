import json
import re
from .SkillPrediction import SkillPrediction as Prediction
from sklearn.metrics.pairwise import cosine_similarity
from langchain.llms import HuggingFaceTextGenInference
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage


class SkillRetriever:
    def __init__(self, embedding, reranker, skilldbs, request):
        """
        Initialize the SkillRetriever object.

        Parameters:
        - embedding: The embedding model.
        - reranker: The reranker model.
        - skilldbs: A dictionary of skill databases.
        - request: The request object containing the request parameters.
        """
        self.embedding = embedding
        self.reranker = reranker
        self.skill_taxonomy = request.skill_taxonomy
        self.skilldb = skilldbs[self.skill_taxonomy]
        self.doc = request.doc
        self.los = request.los
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

    async def predict(self) -> tuple:
        """
        Predicts the top-k skills based on the learning outcomes.

        Returns:
            tuple: A tuple containing the learning outcomes and the predicted skills.
        """
        learningoutcomes = await self.get_learning_outcomes()

        # Embed the learning outcomes.
        embedded_doc = self.embedding.embed_documents([learningoutcomes])

        # Do similarity search for skills.
        predictions = self.get_top_similar_skills(self.los)

        # Define artificial threshholds for relevancy by identifying where the similarity rating decreases the fastest.
        if not self.llm_validation and not self.do_rerank:
            predictions = self.applyDynamicThreshold(predictions)

        # Finetune predictions based on the known skills.
        predictions = self.finetune_on_validated_skills(predictions)

        # Reduce amount of predictions before performance hungry validation.
        predictions = predictions[: int(self.top_k * 1.5)]

        # Validate predictions.
        if self.llm_validation:
            predictions = await self.validate_with_llm(predictions)

        # Sort predictions by score.
        predictions = sorted(predictions, key=lambda x: x.score, reverse=True)

        # Some scores might have become negative due to penalties. Normalize scores.
        if len(predictions) > 0:
            min_score = predictions[-1].score
            if min_score < 0:
                for prediction in predictions:
                    prediction.score -= min_score
            
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

        # Return predictions.
        return self.los, predictions[: self.top_k]

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

        return learningoutcomes

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
                    "Als Redakteur identifizierst du explizit genannte Lernziele in Kursbeschreibungen. Ignoriere Vorraussetzungen und Zielgruppen."
                )
            ),
            HumanMessagePromptTemplate.from_template(
                "Kursbeschreibung: {course}"
                "Liste die explizit genannten Lernziele auf, jeweils in einer neuen Zeile."
                "Nutze kurze, einfache Sprache und BLOOM-Verben für Fähigkeiten, Nomen für Wissen."
            ),
        ]

        learningoutcomes = await self.get_chatresponse(
            messages, {"course": doc[:3500]}, use_most_competent_llm=False
        )

        # Remove list decorations using regular expressions
        learningoutcomes = re.sub(
            r"^ *[\d.-]+ *|^ *\* *|^ *- *", "", learningoutcomes, flags=re.MULTILINE
        )

        return learningoutcomes

    def get_top_similar_skills(self, learning_outcomes: list) -> list:
        """
        Retrieves the top similar skills based on the given learning outcomes.

        Args:
            learning_outcomes (list): A list of learning outcomes.

        Returns:
            list: A list of top similar skills.
        """
        similar_skills = []
        # Determine the size of the subsets and the amount of overlap based on the length of learning_outcomes.
        subset_size = max(5, len(learning_outcomes) // 10)
        overlap = max(0, subset_size // 2)

        # Create overlapping subsets of learning_outcomes and do similarity search for each subset.
        subsets = []
        if len(learning_outcomes) <= subset_size:
            subsets.append("\n".join(learning_outcomes))
        else:
            for i in range(0, len(learning_outcomes) - overlap, subset_size - overlap):
                subsets.append("\n".join(learning_outcomes[i : i + subset_size]))

        # Do similarity search for each subset.
        for subset in subsets:
            similar_subset_skills = (
                self.skilldb.similarity_search_with_relevance_scores(
                    subset, min(self.top_k, 20) + len(self.validated_skills)
                )
            )

            # Convert skill documents to predictions.
            predictions = [
                self.create_prediction(skill) for skill in similar_subset_skills
            ]

            # Filter out predictions that are already known or duplicates or not part of the filterconcepts.
            predictions = self.filter_predictions(predictions, sort=True)

            # if self.do_rerank:
            #     predictions = self.rerank(predictions, "\n".join(subset))

            similar_skills.extend(predictions)

        # Filter out predictions that are already known or duplicates or not part of the filterconcepts.
        similar_skills = self.filter_predictions(similar_skills, sort=True)

        # Rerank the predictions based on all learning outcomes.
        if self.do_rerank:
            similar_skills = self.rerank(similar_skills, "\n".join(learning_outcomes))

        return similar_skills

    def get_thl_model(self, use_most_competent_llm=False):
        thlmodel = "em-german-70b" if use_most_competent_llm else "zephyr-7b-alpha"
        return HuggingFaceTextGenInference(
            inference_server_url=f"https://{thlmodel}.llm.mylab.th-luebeck.dev",
            temperature=self.temperature,
            seed=42,  # we use a seed of 42 to get reproducible results
            max_new_tokens=512,  # embedding models are trained on 512 sequence length, so we use this as a max output length for chat responses
        )

    def get_mistral_model(self, use_most_competent_llm=False):
        return ChatMistralAI(
            model="mistral-medium" if use_most_competent_llm else "mistral-small",
            mistral_api_key=self.mistral_api_key,
            temperature=self.temperature,
            seed=42,
            max_tokens=512,
        )

    def get_openai_model(self, use_most_competent_llm=False):
        return ChatOpenAI(
            model="gpt-4-0125-preview"
            if use_most_competent_llm
            else "gpt-3.5-turbo-1106",
            openai_api_key=self.openai_api_key,
            temperature=self.temperature,
            seed=42,
            max_tokens=512,
        )

    def get_model(self, use_most_competent_llm=False):
        """
        Retrieves the appropriate language model for chat responses.

        Args:
            use_most_competent_llm (bool): Whether to use the most competent language model.

        Returns:
            object: The language model.
        """
        if self.mistral_api_key:
            return self.get_mistral_model(use_most_competent_llm).with_fallbacks(
                [self.get_thl_model(use_most_competent_llm)]
            )
        if self.openai_api_key:
            return self.get_openai_model(use_most_competent_llm).with_fallbacks(
                [self.get_thl_model(use_most_competent_llm)]
            )

        return self.get_thl_model(use_most_competent_llm).with_fallbacks(
            [self.get_mistral_model(use_most_competent_llm)]
        )

    async def get_chatresponse(
        self, messages: list, context: dict, use_most_competent_llm=False
    ) -> str:
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

        chain = prompt | self.get_model(use_most_competent_llm) | StrOutputParser()

        chatresponse = chain.invoke(context)

        chatresponse = chatresponse.replace("ASSISTANT: ", "").strip()

        return chatresponse

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
        messages = [
            SystemMessage(
                content=(
                    "Du bist ein Redakteur einer Weiterbildungsplatform. Deine Aufgabe ist es zu prüfen, welche der vorgeschlagenen Kompetenzen zum Kursangebot inhaltlich passen."
                    "Berücksichtige dabei folgende Fragestellungen."
                    "Passen die Kompetenzen thematisch zu den Lernzielen des Kurses?"
                    "Sind die Kompetenzen zu allgemein oder zu spezifisch?"
                    "Sind die Kompetenzen zu umfangreich oder zu einfach?"
                )
            ),
            HumanMessagePromptTemplate.from_template(
                "Kursbeschreibung: {course}"
                "Kompetenzen: {skills}"
                "Erzeuge eine Liste auschließlich derer Kompetenzen, die sehr gut zu den Lernzielen des Kurses passen. Behalte dabei den Wortlaut der Kompetenzen bei."
                "Nenne eine Kompetenz pro Zeile. Die Antwort sollte nur die Kompetenzen selbst enthalten, ohne Einleitungen oder zusätzliche Worte."
            ),
        ]

        # Get chat response from language model.
        chatresponse = await self.get_chatresponse(
            messages,
            {"course": context[:3500], "skills": "\n".join(skilllabels)},
            use_most_competent_llm=True,
        )

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
            max_score = 12
            score = max(min(score, max_score), -max_score)
            score = (score + max_score) / (max_score * 2)
            prediction.score = score

            prediction.fit = fit

            # If strict mode is enabled, only keep predictions that are validated.
            if self.strict > 0 and not fit:
                continue

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
        # Broader concepts are only available for ESCO skills. If the skill is not an ESCO skill, return True.
        if len(self.filterconcepts) == 0 or "ESCO" not in self.skill_taxonomy:
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
        if self.skill_taxonomy == "ESCO":
            return Prediction.from_esco(skill)
        elif self.skill_taxonomy == "GRETA":
            return Prediction.from_greta(skill)
        else:
            return Prediction.from_other(skill)

    def applyDynamicThreshold(self, predictions: list) -> list:
        """
        Applies dynamic thresholding to the predictions based on the specified strictness level.

        Args:
            predictions (list): A list of prediction dictionaries, each containing a "score" key.

        Returns:
            list: The filtered predictions based on the dynamic thresholding.

        """
        if self.strict == 0 or len(predictions) <= 2:
            return predictions

        # Identify the biggest and second biggest gap between the skills with scores higher than 0.2.
        gaps = []
        for i in range(len(predictions) - 1):
            gaps.append(predictions[i].score - predictions[i + 1].score)

        # Get the indices of the two largest gaps.
        max_gap_skill_index = gaps.index(max(gaps)) + 1
        if self.strict == 3:
            # Return predictions up to the first largest gap.
            return predictions[:max_gap_skill_index]
        elif self.strict <= 2:
            # Return predictions up to the second largest gap.
            max_gap = 0
            max_gap_skill_index_2 = 0
            for i in range(max_gap_skill_index + 1, len(predictions) - 1):
                if predictions[i].score > self.trusted_score:
                    continue
                gap = predictions[i].score - predictions[i + 1].score
                if gap > max_gap:
                    max_gap = gap
                    max_gap_skill_index_2 = i

            if self.strict == 2:
                return predictions[: max_gap_skill_index_2 + 1]
            
            if self.strict == 1:
                # Return predictions up to the third largest gap.
                max_gap = 0
                max_gap_skill_index_3 = 0
                for i in range(max_gap_skill_index_2 + 1, len(predictions) - 1):
                    if predictions[i].score > self.trusted_score:
                        continue
                    gap = predictions[i].score - predictions[i + 1].score
                    if gap > max_gap:
                        max_gap = gap
                        max_gap_skill_index_3 = i

                predictions = predictions[: max_gap_skill_index_3 + 1]

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
        valid_docs = self.skilldb.similarity_search_with_relevance_scores(
            validSkillLabels, 10
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
        invalid_docs = self.skilldb.similarity_search_with_relevance_scores(
            invalidSkillLabels, 10
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
                    penalty = -((similarValidSkill.score) ** 4) * 0.5
                    prediction.penalty += penalty
                    prediction.score += penalty
                    break

        return self.filter_predictions(predictions)
