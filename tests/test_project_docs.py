from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
README_PATH = PROJECT_ROOT / "README.md"


def test_required_project_docs_exist():
    required_docs = [
        "problem_definition.md",
        "data_contract.md",
        "decisions_log.md",
        "model_rubric_matrix.md",
        "feature_audit.md",
    ]

    for doc_name in required_docs:
        assert (DOCS_DIR / doc_name).exists(), f"Missing required doc: {doc_name}"


def test_readme_links_to_key_docs():
    readme_text = README_PATH.read_text(encoding="utf-8")

    for relative_path in [
        "docs/problem_definition.md",
        "docs/data_contract.md",
        "docs/feature_audit.md",
        "docs/model_rubric_matrix.md",
        "docs/decisions_log.md",
    ]:
        assert relative_path in readme_text
