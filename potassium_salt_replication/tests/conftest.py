"""Run the tests against small synthetic inputs, not the GBD extract.

The model's real inputs are GBD 2023 estimates, which are not kept in the
repository.  The tests check the mechanics, and any plausible numbers will do.
So before anything else is imported from the package, this file points it at a
throwaway directory and fills that with the two made-up countries of
``vivarium_potassium_salt.data.synthetic``.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="potassium_salt_tests_"))
os.environ["POTASSIUM_SALT_ROOT"] = str(ROOT)

from vivarium_potassium_salt.data import synthetic  # noqa: E402  (after the line above)

synthetic.write_inputs(ROOT)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(ROOT, ignore_errors=True)
