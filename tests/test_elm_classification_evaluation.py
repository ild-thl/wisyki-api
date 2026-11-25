import json
from pathlib import Path

import pytest

from src.models.elm_loq_metadata import ElmLoqMetadataRequest, ElmLoqMetadataResponse
from src.scripts.evaluate_elm_classifications import load_dataset, score_results


def test_loads_provisional_public_offer_dataset():
    dataset = load_dataset()

    assert len(dataset["cases"]) == 9
    assert dataset["label_policy"].startswith("provisional")
    assert all(
        case["source_url"].startswith("https://dlc.sh/") for case in dataset["cases"]
    )
    assert all(case["source_language"] in {"de", "en"} for case in dataset["cases"])
    assert all(case["expected_eqf_levels"] for case in dataset["cases"])


def test_dataset_rejects_nonterminal_or_malformed_codes(tmp_path):
    dataset = {
        "version": 2,
        "cases": [
            {
                "id": "invalid",
                "source_url": "https://example.test/",
                "source_language": "de",
                "title": "Title",
                "description": "Description",
                "learning_outcome": "Outcome",
                "expected_isced_f_codes": ["06"],
                "expected_eqf_levels": [4],
            }
        ],
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid expected ISCED-F"):
        load_dataset(path)


def test_dataset_rejects_invalid_eqf_levels(tmp_path):
    dataset = {
        "version": 2,
        "cases": [
            {
                "id": "invalid",
                "source_url": "https://example.test/",
                "source_language": "de",
                "title": "Title",
                "description": "Description",
                "learning_outcome": "Outcome",
                "expected_isced_f_codes": ["0613"],
                "expected_eqf_levels": [9],
            }
        ],
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid expected EQF"):
        load_dataset(path)


def test_scores_acceptable_codes_coverage_and_expected_abstention():
    cases = [
        {
            "id": "match",
            "expected_isced_f_codes": ["0613", "0612"],
            "expected_eqf_levels": [4],
        },
        {"id": "miss", "expected_isced_f_codes": ["0211"], "expected_eqf_levels": [3]},
        {"id": "abstain", "expected_isced_f_codes": [], "expected_eqf_levels": [2]},
    ]
    results = [
        {"id": "match", "isced_f_code": "0613", "eqf_level": 4, "duration_ms": 12},
        {"id": "miss", "isced_f_code": None, "eqf_level": 5, "duration_ms": 15},
        {"id": "abstain", "isced_f_code": None, "eqf_level": 2, "duration_ms": 9},
    ]

    metrics = score_results(cases, results)

    assert metrics["accepted_code_accuracy"] == 0.5
    assert metrics["labelled_coverage"] == 0.5
    assert metrics["expected_abstention_accuracy"] == 1
    assert metrics["eqf_exact_accuracy"] == 2 / 3
    assert metrics["eqf_adjacent_accuracy"] == 2 / 3
    assert metrics["classification_latency_ms"] == {"median": 12, "max": 15}


def test_hub_request_and_result_fixtures_match_api_models():
    fixtures = Path(__file__).parent / "fixtures"
    request_data = json.loads(
        (fixtures / "elm_classification_request.json").read_text()
    )
    result_data = json.loads((fixtures / "elm_classification_result.json").read_text())

    request = ElmLoqMetadataRequest.model_validate(request_data)
    result = ElmLoqMetadataResponse.model_validate(result_data)

    assert request.options.classifications == ["EQF", "ISCED_F"]
    assert result.classification_suggestions.isced_f[0].code == "0613"


def test_secondary_candidate_does_not_mask_incorrect_dominant_field():
    metrics = score_results(
        [
            {
                "id": "course",
                "expected_isced_f_codes": ["0613"],
                "expected_eqf_levels": [4],
            }
        ],
        [
            {
                "id": "course",
                "isced_f_code": "0211",
                "isced_f_codes": ["0211", "0613"],
                "eqf_level": 4,
            }
        ],
    )
    assert metrics["accepted_code_accuracy"] == 0
    assert metrics["any_candidate_accuracy"] == 1
