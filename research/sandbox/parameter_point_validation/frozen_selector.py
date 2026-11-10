"""Hash-pinned import of the exact k300 selector implementation."""
from pathlib import Path
import hashlib, importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'research/sandbox/k300_preregistration/selector.py'
EXPECTED_SHA256 = '57dbc4c59a26b66ac4f384cb6c7b9accd33d0528561b88b747eb0bbfdcc053b2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


if sha(SOURCE) != EXPECTED_SHA256:
    raise ImportError('Frozen selector source hash mismatch')
spec = importlib.util.spec_from_file_location('_exact_k300_selector', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

build_hamiltonian = module.build_hamiltonian
freeze = module.freeze
greedy = module.greedy
importance = module.importance
indices = module.indices
spread_order = module.spread_order
