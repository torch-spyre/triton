# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from pathlib import Path
import pytest

from utils import make_ktir_mod
from scripts._patterns import clean_ir


CONVERSION_DIR = Path(__file__).parent / "Conversion"
CONVERSIONS = sorted(CONVERSION_DIR.glob("*.mlir"))


def run_passes(path: Path) -> str:
    module = make_ktir_mod(path, grid=(32,))
    return clean_ir(module.str())


@pytest.mark.parametrize("conversion", CONVERSIONS, ids=lambda p: p.name)
def test_conversion(conversion, check_ir):
    produced = run_passes(conversion)
    check_ir(produced, conversion)
