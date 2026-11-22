import pytest

from remember.build import build
from twin.generate import main


@pytest.fixture(scope="session")
def full_twin(tmp_path_factory):
    """56 days of twin data, generated once per test session (about 1 minute)."""
    out = tmp_path_factory.mktemp("twin56")
    main(days=56, out=out)
    return out


@pytest.fixture(scope="session")
def full_store(full_twin, tmp_path_factory):
    """The on-device store built from the 56 days (SENSE -> REMEMBER), once per session."""
    path = tmp_path_factory.mktemp("store") / "vehicle.db"
    rep = build(twin=full_twin, path=path, quiet=True)
    return path, rep
