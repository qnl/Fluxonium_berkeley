"""Read-only authentication of pinned historical evidence and future k380."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate():
    evidence = json.loads((HERE / 'historical_evidence.json').read_text())
    mismatches = [rel for rel, digest in evidence['files'].items()
                  if not (ROOT / rel).is_file() or sha(ROOT / rel) != digest]
    old = ROOT / 'research/sandbox/epsj011_k340_preregistration'
    prior = load('k380_historical_auth', old / 'authenticate_existing.py').authenticate()
    checkpoint = ROOT / 'research/checkpoints/four-mode-asymmetric-n71-epsj011-k340'
    meta = json.loads((checkpoint / 'checkpoint.json').read_text())
    marker = json.loads((checkpoint / 'COMPLETE').read_text())
    evaluation = json.loads((old / 'evaluation.json').read_text())
    checks = {
        'previous_authentication': prior['status'] == 'PASS',
        'complete': meta['status'] == marker['status'] == 'complete',
        'complete_link': marker['checkpoint_sha256'] == sha(checkpoint / 'checkpoint.json'),
        'configuration': meta['configuration_sha256'] == '10997084ff05005d20219d8d7110fe8751a10ee64a410a031f81e6ec479829a0',
        'payload_hashes': all(sha(checkpoint / name) == digest for name, digest in meta['file_sha256'].items()),
        'quality': meta['validation']['maximum_eigenpair_residual'] <= 1e-7 and meta['validation']['orthogonality_residual_inf'] <= 1e-8,
        'retained_FAIL': evaluation['retained_basis_convergence']['verdict'] == 'FAIL',
        'tail_D': evaluation['tail_hypothesis']['classification'] == 'D_NONMONOTONIC_UNRESOLVED',
        'secondary_PASS': evaluation['selector_secondary']['verdict'] == 'PASS',
    }
    return {'status': 'PASS' if not mismatches and all(checks.values()) else 'FAIL',
            'hash_mismatches': mismatches, 'checks': checks,
            'checkpoint_sha256': sha(checkpoint / 'checkpoint.json')}


def authenticate_new(checkpoint, selection):
    from profile import install_profile, EXPECTED_CONFIGURATION_SHA256, FINAL_DIRECTORY_NAME
    from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner
    if authenticate()['status'] != 'PASS':
        raise ValueError('Historical authentication failed')
    manifest = json.loads((HERE / 'preregistration_manifest.json').read_text())
    seal = json.loads((HERE / 'preregistration_seal.json').read_text())
    if seal['manifest_sha256'] != sha(HERE / 'preregistration_manifest.json'):
        raise ValueError('Manifest seal mismatch')
    if any(sha(ROOT / rel) != digest for rel, digest in manifest['files'].items()):
        raise ValueError('Frozen source changed')
    if Path(checkpoint).resolve() != ROOT / 'research/checkpoints' / FINAL_DIRECTORY_NAME:
        raise ValueError('Unexpected checkpoint path')
    if Path(selection).resolve() != HERE / 'selection.json':
        raise ValueError('Unexpected selection path')
    install_profile(runner)
    meta = runner.validate_complete_checkpoint_directory(checkpoint)
    if meta['configuration_sha256'] != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError('New fingerprint mismatch')
    analysis = json.loads((Path(checkpoint) / 'analysis.json').read_text())
    if sorted(int(k.rsplit('k', 1)[1]) for k in analysis['retained_basis_cases']) != [300,320,340,360,380]:
        raise ValueError('Nested prefixes mismatch')
    selection_seal = json.loads(Path(str(selection) + '.seal.json').read_text())
    if selection_seal['selection_sha256'] != sha(selection) or analysis['selection_freeze']['selection_sha256'] != sha(selection):
        raise ValueError('Selection linkage mismatch')
    if not analysis['selection_freeze']['frozen_before_candidate_or_full_new_shell_dressed_validation']:
        raise ValueError('Selection chronology mismatch')
    criteria = json.loads((HERE / 'criteria.json').read_text())
    if meta['validation']['maximum_eigenpair_residual'] > criteria['quality']['stage1_residual_max'] or meta['validation']['orthogonality_residual_inf'] > criteria['quality']['stage1_orthogonality_inf_max']:
        raise ValueError('Parent quality failed')
    return {'status':'PASS', 'configuration_sha256': EXPECTED_CONFIGURATION_SHA256,
            'checkpoint_sha256':sha(Path(checkpoint) / 'checkpoint.json'),
            'historical_references_preserved':True, 'prefixes':[300,320,340,360,380]}


if __name__ == '__main__':
    import argparse, sys
    sys.path[:0] = [str(HERE), str(ROOT)]
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.selection):
        parser.error('Supply both checkpoint and selection')
    result = authenticate_new(args.checkpoint, args.selection) if args.checkpoint else authenticate()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
