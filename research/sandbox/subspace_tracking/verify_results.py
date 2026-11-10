"""Verify completed artifacts and unchanged production diffs; no eigensolves."""
from pathlib import Path
import ast,hashlib,json,re,subprocess
import numpy as np
from scipy import linalg as la

HERE=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def matrix(x):
    return np.asarray(x['real'])+1j*np.asarray(x['imag']) if isinstance(x,dict) else np.asarray(x)

def main():
    manifest=json.loads((HERE/'resume_manifest.json').read_text())
    for name,h in manifest.items():
        if name.endswith('.npz') or name=='validate_gridium.py': assert sha(HERE/name)==h,name
    for args,name in [(['git','diff','--binary'],'initial_worktree.patch'),
                      (['git','diff','--cached','--binary'],'initial_index.patch'),
                      (['git','status','--short'],'initial_status.txt')]:
        assert subprocess.check_output(args)==(HERE/name).read_bytes(),name
    for p in HERE.glob('*.py'): ast.parse(p.read_text(),filename=str(p))
    log=(HERE/'tests.log').read_text()
    assert re.search(r'Ran 19 tests',log) and log.rstrip().endswith('OK')
    pair=json.loads((HERE/'pair45_46_operator_analysis.json').read_text())
    sub=pair['subspace'];s=np.asarray(sub['singular_values'])
    assert np.isclose(sub['mean_subspace_overlap'],np.mean(s*s),rtol=0,atol=1e-12)
    assert np.isclose(sub['projector_frobenius']**2,4-2*sum(s*s),rtol=0,atol=1e-12)
    for block in pair['blocks'].values():
        a=matrix(block['reference_block']);b=matrix(block['candidate_block'])
        delta=matrix(block['comparison']['aligned_candidate'])-a
        for which,mat in [('reference',a),('candidate',b)]:
            inv=block['comparison'][which]
            np.testing.assert_allclose(la.svdvals(mat),inv['singular_values'],rtol=1e-12,atol=1e-14)
            assert np.isclose(la.norm(mat,'fro'),inv['frobenius'],rtol=1e-12,atol=1e-14)
        assert np.isclose(la.norm(delta,'fro'),block['comparison']['aligned_difference_frobenius'],rtol=1e-12,atol=1e-14)
    c=pair['blocks']['to_low11']['summary']
    assert .22<c['aligned_relative_frobenius']<.23
    assert pair['candidate_cluster_in_assigned_order']==[46,45]
    synthetic=json.loads((HERE/'synthetic_summary.json').read_text())
    assert synthetic['maximum_aligned_operator_error']<1e-12
    assert synthetic['maximum_observable_descendant_error']<1e-12
    assert synthetic['coarse_step_all_ambiguous']
    report=(HERE/'REPORT.md').read_text()
    missing=[x for x in re.findall(r'\]\(([^)]+)\)',report)
        if not x.startswith('http') and x!='verification.json' and not (HERE/x).exists()]
    assert not missing,missing
    changed=[name for name,h in manifest.items() if sha(HERE/name)!=h]
    created=sorted(p.name for p in HERE.iterdir() if p.is_file() and p.name not in manifest)
    result=dict(status='PASS',unit_tests=19,unit_tests_failed=0,
        artifact_consistency='PASS',signed_operator_block_checks='PASS',
        cached_npz_files_unchanged=sum(name.endswith('.npz') for name in manifest),
        reconstruction_driver_unchanged=True,production_worktree_and_index_diffs_unchanged=True,
        git_status_unchanged=True,new_Gridium_eigensolves=0,
        changed_existing_files=changed,created_files=sorted(set(created+['verification.json'])),
        report_sha256=sha(HERE/'REPORT.md'))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
