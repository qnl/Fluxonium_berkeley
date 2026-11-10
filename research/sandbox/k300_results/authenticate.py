"""Post-run authentication for the sealed k300 experiment; no eigensolves."""
from pathlib import Path
import hashlib, json, sys

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
PREG=ROOT/'research/sandbox/k300_preregistration'; CPROOT=ROOT/'research/checkpoints'
sys.path[:0]=[str(PREG),str(ROOT/'research/sandbox/k260_preregistration'),str(ROOT)]
from Simulations.Rabi_3Photon import run_four_mode_spatial71_reference as runner
from k260_profile import install_profile as install_k260
from k300_profile import install_profile as install_k300

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

criteria=json.loads((PREG/'criteria.json').read_text())
manifest=json.loads((PREG/'preregistration_manifest.json').read_text())
assert sha(PREG/'preregistration_manifest.json')=='52229c31c720d54097ccb6e7a5fd74d446f8d5e0c3318d47e20293356cec754e'
assert all(sha(ROOT/rel)==expected for rel,expected in manifest['files'].items())
for rel,expected in criteria['authenticated_inputs']['required_source_sha256'].items(): assert sha(ROOT/rel)==expected
install_k260(runner); install_k300(runner)
references={}
runner.load_verified_n51_checkpoint(CPROOT/'2026-09-26-asymmetric-n51')
for name in ('four-mode-asymmetric-n71-k180','four-mode-asymmetric-n71-k220','four-mode-asymmetric-n71-k260'):
    meta=runner.validate_complete_checkpoint_directory(CPROOT/name)
    references[name]={'checkpoint_json_sha256':sha(CPROOT/name/'checkpoint.json'),
                       'configuration_sha256':meta['configuration_sha256']}
k260criteria=json.loads((ROOT/'research/sandbox/k260_preregistration/criteria.json').read_text())
assert references['four-mode-asymmetric-n71-k220']['checkpoint_json_sha256']==k260criteria['authenticated_inputs']['k220_checkpoint_sha256']
assert references['four-mode-asymmetric-n71-k260']['checkpoint_json_sha256']==criteria['authenticated_inputs']['k260_checkpoint_json_sha256']
cp=CPROOT/'four-mode-asymmetric-n71-k300'; meta=runner.validate_complete_checkpoint_directory(cp)
assert meta['configuration_sha256']==criteria['profile']['configuration_sha256']
assert meta['validation']['maximum_eigenpair_residual']<=criteria['quality']['stage1_residual_max']
assert meta['validation']['orthogonality_residual_inf']<=criteria['quality']['stage1_orthogonality_inf_max']
analysis=json.loads((cp/'analysis.json').read_text()); expected_ks=(180,200,220,240,260,280,300)
assert tuple(int(k.rsplit('k',1)[1]) for k in analysis['retained_basis_cases'])==expected_ks
for k in expected_ks: assert analysis['retained_basis_cases'][f'nested_prefix_k{k}']['retained_indices']==list(range(k))
selection=PREG/'selection.json'; seal=json.loads((PREG/'selection.json.seal.json').read_text())
assert seal['selection_sha256']==sha(selection)
link=analysis['held_out_selection_freeze']
assert link['selection_sha256']==sha(selection) and link['seal_sha256']==sha(PREG/'selection.json.seal.json')
assert link['frozen_before_full_stage2'] and not link['provenance']['selection_used_full_candidate_dressed_results']
selected=json.loads(selection.read_text())
assert set(selected['shells']['S1']['order'])==set(range(260,280))
assert set(selected['shells']['S2']['order'])==set(range(280,300))
for context in ('S1','S2','union'):
    for method in ('physics','prefix','spread','spread_reverse'):
        for budget in (3,6,10,15):
            ranks=selected['spaces'][f'{context}_{method}_{budget}']
            assert len(ranks)==260+(budget if context!='union' else 2*budget)
result={'status':'PASS','complete':(cp/'COMPLETE').is_file(),'configuration_sha256':meta['configuration_sha256'],
        'checkpoint_json_sha256':sha(cp/'checkpoint.json'),'payload_hashes_verified':True,
        'nested_prefixes':list(expected_ks),'selection_sha256':sha(selection),
        'selection_seal_sha256':sha(PREG/'selection.json.seal.json'),'selection_frozen_before_full_stage2':True,
        'max_eigenpair_residual':meta['validation']['maximum_eigenpair_residual'],
        'orthogonality_residual_inf':meta['validation']['orthogonality_residual_inf'],
        'total_runtime_seconds_internal':meta['total_runtime_seconds'],
        'factorization_seconds':meta['stage1_solver_diagnostics']['factorization_seconds'],
        'eigsh_seconds':meta['stage1_solver_diagnostics']['eigsh_seconds'],
        'historical_references':references,'manifest_sha256':sha(PREG/'preregistration_manifest.json')}
with (HERE/'authentication.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
print(json.dumps(result,indent=2))
