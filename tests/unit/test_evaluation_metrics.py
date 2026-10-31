from pathlib import Path

from tests.evaluation.run_rag_eval import evaluate_syntactic_validity


def test_evaluate_syntactic_validity() -> None:
    valid_import = "from langchain_core.language_models import BaseChatModel"
    assert evaluate_syntactic_validity(valid_import) is True

    invalid_import = "from langchain_core.language_models import ("
    assert evaluate_syntactic_validity(invalid_import) is False


def test_golden_dataset_structure() -> None:
    dataset_path = Path("tests/evaluation/golden_dataset.json")
    assert dataset_path.exists()

    import json

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 10
    for item in data:
        assert "question" in item
        assert "expected_symbol" in item
        assert "expected_canonical_import" in item
        assert evaluate_syntactic_validity(item["expected_canonical_import"]) is True
