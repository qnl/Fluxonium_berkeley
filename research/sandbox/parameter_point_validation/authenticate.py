"""Future post-run authentication; reads only and writes a requested report."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT)]
from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner
from profile import install_profile,EXPECTED_CONFIGURATION_SHA256


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('checkpoint',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    c=json.loads((HERE/'criteria.json').read_text()); install_profile(runner)
    meta=runner.validate_complete_checkpoint_directory(a.checkpoint)
    if meta['configuration_sha256']!=EXPECTED_CONFIGURATION_SHA256: raise ValueError('fingerprint')
    analysis=json.loads((a.checkpoint/'analysis.json').read_text()); link=analysis['held_out_selection_freeze']
    selection=HERE/'selection.json'; seal=Path(str(selection)+'.seal.json')
    sealed=json.loads(seal.read_text())
    if sha(selection)!=sealed['selection_sha256'] or link['selection_sha256']!=sealed['selection_sha256']:
        raise ValueError('selection authentication')
    if not link['frozen_before_candidate_or_full_stage2']: raise ValueError('selection timing')
    prefixes=sorted(int(k.rsplit('k',1)[1]) for k in analysis['retained_basis_cases'])
    if prefixes!=c['prefixes']: raise ValueError('prefixes')
    for rel,digest in c['authenticated_inputs']['required_source_sha256'].items():
        if sha(ROOT/rel)!=digest: raise ValueError(f'source {rel}')
    refs={'original_k260_checkpoint_sha256':ROOT/'research/checkpoints/four-mode-asymmetric-n71-k260/checkpoint.json',
          'original_k300_checkpoint_sha256':ROOT/'research/checkpoints/four-mode-asymmetric-n71-k300/checkpoint.json',
          'original_k300_authentication_sha256':ROOT/'research/sandbox/k300_results/authentication.json',
          'k300_shell_attribution_sha256':ROOT/'research/sandbox/k300_shell_attribution/summary.json'}
    for key,path in refs.items():
        if sha(path)!=c['authenticated_inputs'][key]: raise ValueError(key)
    with np.load(a.checkpoint/'stage1_eigensystem.npz',allow_pickle=False) as z:
        v=np.asarray(z['eigenvectors']); orth=float(np.linalg.norm(v.conj().T@v-np.eye(300),ord=np.inf))
    residual=float(max(meta['stage1_solver_diagnostics']['eigenpair_residuals']))
    if residual>c['quality']['stage1_residual_max'] or orth>c['quality']['stage1_orthogonality_inf_max']:
        raise ValueError('parent quality')
    out={'status':'PASS','configuration_sha256':EXPECTED_CONFIGURATION_SHA256,
      'checkpoint_sha256':sha(a.checkpoint/'checkpoint.json'),'payload_hashes_verified':True,
      'selection_sha256':sealed['selection_sha256'],'selection_seal_sha256':sha(seal),
      'selection_frozen_before_validation':True,'prefixes':prefixes,
      'max_eigenpair_residual':residual,'orthogonality_residual_inf':orth,
      'historical_references_unchanged':True}
    with a.output.open('x') as f: json.dump(out,f,indent=2); f.write('\n')


if __name__=='__main__': main()
