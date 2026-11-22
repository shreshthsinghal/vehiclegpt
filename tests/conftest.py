import pytest

from twin.generate import main


@pytest.fixture(scope="session")
def full_twin(tmp_path_factory):
    """56 days of twin data, generated once per test session (about 1 minute)."""
    out = tmp_path_factory.mktemp("twin56")
    main(days=56, out=out)
    return out
