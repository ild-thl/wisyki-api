from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class SkillPrediction:
    """
    Represents a skill prediction.

    Attributes:
        uri (str): The URI of the skill.
        title (str): The title of the skill.
        score (float): The score of the skill prediction.
        penalty (int, optional): The penalty for the skill prediction. Defaults to 0.
        fit (bool, optional): Indicates if the skill prediction is a good fit. Defaults to True.
        metadata (dict, optional): Additional metadata for the skill prediction. Defaults to None.
    """
    uri: str
    title: str
    score: float
    penalty: int = 0
    fit: bool = True
    metadata: Optional[dict] = None

    @classmethod
    def from_esco(cls, skill):
        """
        Creates a SkillPrediction instance from an ESCO skill.

        Args:
            skill (tuple): The ESCO skill data.

        Returns:
            SkillPrediction: The SkillPrediction instance.
        """
        return cls(
            uri=skill[0].metadata["conceptUri"],
            title=skill[0].metadata["preferredLabel"],
            score=skill[1],
            metadata={
                "broaderConcepts": [
                    concept["uri"]
                    for concept in json.loads(skill[0].metadata["broaderHierarchyConcepts"])
                ]
                if skill[0].metadata["broaderHierarchyConcepts"]
                else [],
                "description": skill[0].page_content,
            },
        )

    @classmethod
    def from_greta(cls, skill):
        """
        Creates a SkillPrediction instance from a GRETA skill.

        Args:
            skill (tuple): The GRETA skill data.

        Returns:
            SkillPrediction: The SkillPrediction instance.
        """
        return cls(
            title=skill[0].metadata["Kompetenzfacette"],
            uri=skill[0].metadata["URI"],
            score=skill[1],
            metadata={
                "ID": skill[0].metadata["ID"],
                "Kompetenzaspekt": skill[0].metadata["Kompetenzaspekt"],
                "Kompetenzbereich": skill[0].metadata["Kompetenzbereich"],
                "Kompetenzanforderungen": skill[0].metadata["Kompetenzanforderungen"],
                "Kompetenzbeschreibung": skill[0].metadata["Kompetenzbeschreibung"],
            },
        )

    @classmethod
    def from_other(cls, skill):
        """
        Creates a SkillPrediction instance from another source.

        Args:
            skill (tuple): The skill data.

        Returns:
            SkillPrediction: The SkillPrediction instance.
        """
        return cls(
            uri=skill[0].metadata["id"],
            title=skill[0].page_content,
            score=skill[1],
        )
