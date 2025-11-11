from pathlib import Path


TESTS_DIR = Path(__file__).parent.resolve()
DATA_DIR = TESTS_DIR / "data"

def data_path(*parts) -> Path:
    """Join parts under Tests/data and return a Path."""
    return DATA_DIR.joinpath(*parts)
