"""Shared pytest fixtures. 测试夹具。

Place in the test suite
-----------------------
pytest loads this file automatically before the test modules in ``tests/``.
It makes the in-repository package importable, defines the ``requires_model``
skip marker and provides the fixtures shared by several test modules.

Main objects
------------
``requires_model``
    Marker that skips a test when the trained model weights or the reference
    bathymetry index are missing (for example on a fresh checkout before
    ``scripts/train_model.py`` has been run). Tests that only exercise pure
    functions (geodesy, loaders, storage) do not use it.
``classifier`` / ``planner``
    Session-scoped: the model is loaded once for the whole test run because
    loading the network and the reference index is the slowest step.
``repository``
    Function-scoped: every test gets a new empty SQLite file in its own
    temporary directory, so tests cannot influence each other or the real
    database.
模型相关夹具在整个会话中只加载一次；数据库夹具每个测试使用独立的临时文件。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import ``maritime_route`` from src/ without installing the package.
# 无需安装即可从 src 目录导入。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maritime_route.config import BATHY_INDEX_PATH, MODEL_WEIGHTS_PATH

#: Skip marker for tests that need the trained model files on disk.
#: 缺少已训练模型时跳过相关测试。
requires_model = pytest.mark.skipif(
    not (Path(MODEL_WEIGHTS_PATH).exists() and Path(BATHY_INDEX_PATH).exists()),
    reason="trained model not available - run scripts/train_model.py",
)


@pytest.fixture(scope="session")
def classifier():
    """Shared zone classifier (the process-wide singleton).

    Returns
    -------
    ZoneClassifierService
        Loaded model with its scaler and reference index. The import is done
        inside the fixture so that test modules which do not need the model
        do not import PyTorch.
    """
    from maritime_route.model.inference import ZoneClassifierService
    return ZoneClassifierService.instance()


@pytest.fixture(scope="session")
def planner(classifier):
    """Route planner bound to the shared classifier.

    Parameters
    ----------
    classifier : ZoneClassifierService
        Injected by the ``classifier`` fixture.

    Returns
    -------
    RoutePlanner
        Planner with the default routing configuration.
    """
    from maritime_route.routing.planner import RoutePlanner
    return RoutePlanner(classifier)


@pytest.fixture()
def repository(tmp_path):
    """Fresh SQLite repository in a temporary directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Built-in pytest fixture: a unique empty directory per test.

    Returns
    -------
    RouteRepository
        Repository whose schema has been created in ``tmp_path/test.db``.
    """
    from maritime_route.storage.repository import RouteRepository
    return RouteRepository(tmp_path / "test.db")
