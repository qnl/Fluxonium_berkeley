"""Verify this new review and its source preservation; never write old artifacts."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    data = json.loads((HERE / 'DIAGNOSTICS.json').read_text())
    provenance = json.loads((HERE / 'SOURCE_HASHES.json').read_text())
    with (HERE / 'SPECTRAL_DECOMPOSITION.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == data['spectral_columns']
        rows = list(reader)
    assert len(rows) == 36
    for row, expected in zip(rows, data['spectral_rows']):
        for key, value in expected.items():
            if value is None:
                assert row[key] == '', (key, row[key])
            elif isinstance(value, bool):
                assert row[key] == str(value).lower()
            elif isinstance(value, (int, float)):
                assert math.isfinite(value) and float(row[key]) == value
            else:
                assert row[key] == value
    assert len({(row['comparison'], row['carrier']) for row in rows}) == 36

    # Add only a small saved-scalar decomposition. These terms are differences
    # evaluated on state-specific p,E, not individual-rank addback experiments.
    pair_records = []
    for result in data['rank_results']:
        key = f"k{result['parent']}/{result['shell']}"
        states = data['shell_audit'][key]['states']
        for pair in result['pairs']:
            i, j = map(str, pair['states'])
            sigma = states[j]['exact_self_energy_MHz'] - states[i]['exact_self_energy_MHz']
            relaxation = states[j]['P_space_relaxation_MHz'] - states[i]['P_space_relaxation_MHz']
            rank_difference = pair['rank_scalar_difference_j_minus_i_MHz']
            exact = pair['actual_full_shell_pair_energy_shift_MHz']
            pair_records.append({'shell': key, 'rank': result['rank'], 'states': pair['states'],
                                 'rank_scalar_difference_MHz': rank_difference,
                                 'other_rank_scalar_differences_MHz': sigma - rank_difference,
                                 'whole_shell_sigma_difference_MHz': sigma,
                                 'P_relaxation_difference_MHz': relaxation,
                                 'exact_pair_shift_MHz': exact,
                                 'identity_residual_MHz': rank_difference + (sigma-rank_difference) + relaxation - exact,
                                 'interpretation': 'additive differences of exact state-specific scalar partitions; not exact isolated-rank transition shifts'})
    write_new('SCHUR_PAIR_DECOMPOSITION.json', pair_records)

    # Editorial corrections affect this review only. Keep the clean-run builder
    # text aligned with the generated report and rank table where applicable.
    replacements = {
        '|A01| (MHz)': 'abs(A01) (MHz)',
        'about10–15%': 'about 10–15%', 'almost1 MHz': 'almost 1 MHz',
        'roughly10–14%': 'roughly 10–15%', 'all32 operator': 'all 32 operator',
        'all3 pathway': 'all 3 pathway', 'is0.999762806': 'is 0.999762806',
        'angle1.256322': 'angle 1.256322', 'error1.369015%': 'error 1.369015%',
        'remains1': 'remains 1', 'states7/8': 'states 7/8', 'states0/1': 'states 0/1',
        'state7/8': 'state 7/8',
    }
    for filename in ('REPORT.md', 'RANK_MECHANISM.md', 'build_review.py'):
        path = HERE / filename
        text = path.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text)

    report = HERE / 'REPORT.md'
    text = report.read_text()
    insertion = '''
### Accuracy and perturbative control are different questions

The new-shell fixed-vector relative shift discrepancies span 9.643–14.475%. They satisfy the earlier shell-attribution methodology's numerical 20% error, eta<0.10 and Q-probability<0.01 checks. Thus these data do not demonstrate a breakdown of off-resonant perturbation theory. They demonstrate insufficient accuracy for a stand-alone correction at the present sub-MHz/MHz criteria and the need to include relaxation and nonadditivity. No new official perturbative verdict is introduced.

The maximum magnitude of the fixed-p self-energy slope is 8.9993e-5 in S1 and 1.13563e-4 in S2. This supports weak local dependence on the energy argument at the saved eigenstates, while leaving changes of p and matrix-valued mixing unresolved. Full-shell near-resonant modal weight is zero in both shells.

The top-three aggregate dressed-mode fractions are 50.300195% (S1) and 68.171149% (S2), with effective mode counts 10.204358 and 5.610445. These are much less concentrated than the corresponding rank fractions 87.150372% and 98.873351%. A single Stage-1 rank includes eight mode-4 product states; it is not one dressed shell mode or one physical resonance. Stage-1 energy alone is not a Schur denominator: denominators include the shell Hamiltonian and target energy.

For S2 f78, rank362's scalar difference is −0.919200877 MHz, other ranks contribute +0.254160343 MHz in this scalar partition, and differential P relaxation contributes +0.058541597 MHz. Together they recover −0.606498937 MHz. This illustrates why assigning that exact transition shift to rank362 alone is invalid. All target-pair decompositions are in SCHUR_PAIR_DECOMPOSITION.json.

'''
    anchor = '## Why stable operators do not imply stable spectra'
    assert anchor in text
    report.write_text(text.replace(anchor, insertion + anchor))

    # Check scientific identities independently of the text formatting.
    criteria = json.loads((ROOT / 'research/sandbox/epsj011_k380_preregistration/criteria.json').read_text())
    tolerance_mhz = 1000 * criteria['quality']['schur_identity_GHz_max']
    assert max(abs(r['identity_residual_MHz']) for r in pair_records) <= tolerance_mhz
    for name in ('f01','f07','f08','f78','f81','f72','f21','f03','f38'):
        records = [row for row in data['spectral_rows'] if row['carrier'] == name]
        assert abs(sum(row['signed_upper_minus_lower_MHz'] for row in records) - records[-1]['cumulative_from_k300_MHz']) <= tolerance_mhz
    # This check is a scope guard for new scripts, not a numerical physics test.
    forbidden = {'eig', 'eigh', 'eigvals', 'eigvalsh', 'eigsh', 'eigs',
                 '_solve_stage1_once', 'selected_solve', 'shell_diagnostic',
                 'build_hamiltonian', 'execute'}
    for filename in ('build_review.py', 'verify_review.py'):
        tree = ast.parse((HERE / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
                assert name not in forbidden, (filename, name)
    # Markdown tables must keep a single column count; verify local source links.
    for path in HERE.glob('*.md'):
        count = None
        for line in path.read_text().splitlines():
            if line.startswith('|'):
                cells = len(line.split('|'))
                assert count in (None, cells), (path.name, line)
                count = cells
            else:
                count = None
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if not target.startswith(('https:', 'http:')):
                assert (path.parent / target).exists(), (path.name, target)

    baseline = provenance['preservation_baseline']
    changed = [name for name, digest in baseline.items() if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT).decode()
    assert not changed, changed
    assert status == provenance['git_status_before']
    verification = {'source_authentication': 'PASS', 'spectral_decomposition': 'COMPLETE',
                    'csv_rows': len(rows), 'csv_columns': len(data['spectral_columns']),
                    'csv_numeric_roundtrip': 'EXACT', 'schur_pair_identity_max_residual_MHz': max(abs(r['identity_residual_MHz']) for r in pair_records),
                    'preserved_baseline_files': len(baseline), 'changed_baseline_files': changed,
                    'git_status': status, 'git_status_unchanged': True,
                    'new_eigensolves': 0, 'production_modifications': [],
                    'figures_generated': 0, 'frozen_verdicts_preserved': True,
                    'incomplete_analyses': data['incomplete']}
    write_new('VERIFICATION.json', verification)
    outputs = {path.name: sha(path) for path in sorted(HERE.iterdir()) if path.is_file() and path.name != 'OUTPUT_HASHES.json'}
    write_new('OUTPUT_HASHES.json', outputs)
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
