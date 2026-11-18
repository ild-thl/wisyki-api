# Learning metadata workflow

The ELM/QualityLink metadata recommender supports systems that curate learning opportunities and credentials. It supplies recommendations for human review; it does not publish a complete application profile, create RDF, or assert that a specific learner achieved an outcome.

The generated OpenAPI documentation describes the endpoint contract: [hosted API docs](https://wisyki-api.dev.dlc.sh/competence-analyser/docs).

## Recommendation stages

1. The caller identifies the source language and provides course context, authored learning outcomes, or both.
2. If outcomes are not supplied, the extraction stage identifies intended learning results and supporting evidence in the course information.
3. Each outcome is embedded and searched independently against the requested framework indexes. This preserves the relationship between an outcome and its candidate skills instead of flattening skills into a course-wide list.
4. Candidate outcome-skill relationships are validated with the configured validation method. Unsupported or uncertain candidates are not represented as validated relationships.
5. A missing summary is generated from the source information and the resulting outcomes. Requested translations are produced for outcome titles and summary text; framework skill labels remain in the language stored in the index.

The pipeline preserves authored outcomes and summaries. It retains outcomes with no supported skill matches so downstream reviewers can distinguish an outcome from its framework links. Evidence quotes and warning references help clients present recommendations for review.

## ELM/QualityLink responsibilities

The endpoint's outcome objects and skill relationships are recommendations, not a serialized QualityLink record. A downstream publisher is responsible for creating identifiers, language-tagged RDF literals, concept references, and other required application-profile properties. In particular:

- Outcome identifiers are intentionally not part of the public recommendation contract. Clients can assign identifiers appropriate to their own records.
- `relatedESCOSkill` recommendations correspond to ELM's ESCO-skill relation; other framework recommendations use the general related-skill relation.
- A learning outcome summary is supplied as text for downstream mapping to the appropriate ELM summary representation.
- Similarity scores, validation reasons, evidence, warnings, and processing statistics are recommendation metadata, not ontology properties.

Reviewers should verify the outcome wording, evidence, skill relevance, framework identifier authority, and any profile requirements before publication.
