"""Frozen cancellation-aware verdicts for future parameter-point output."""
from pathlib import Path
import argparse,json,math

HERE=Path(__file__).resolve().parent
CONTEXTS=('S1','S2','union')
FAMILIES={
 'logical_01':('carrier:f01','overlap:0','overlap:1','subspace:0_1','centered:0','centered:1','excitation:1'),
 'comparator_23':('carrier:f21','carrier:f03','overlap:2','overlap:3','subspace:2_3','centered:2','centered:3','excitation:2','excitation:3'),
 'primary_78':('carrier:f78','overlap:7','overlap:8','subspace:7_8','centered:7','centered:8','excitation:7','excitation:8'),
 'cross_carriers':('carrier:f07','carrier:f08','carrier:f81','carrier:f72','carrier:f38'),
 'operators':('prefix:edge:','prefix:block:'),'pathways':('prefix:score:',)}


def members(errors,patterns):
    return [v for k,v in errors.items() if any(k==p or (p.startswith('prefix:') and k.startswith(p[7:])) for p in patterns)]


def family_rms(errors):
    out={}
    for name,patterns in FAMILIES.items():
        vals=members(errors,patterns)
        if not vals: raise ValueError(f'missing family {name}')
        out[name]=math.sqrt(sum(x*x for x in vals)/len(vals))
    return out


def selector_verdict(data,criteria):
    failures=[]; misses=[]; ratios={}; cancellation={}
    failures += [n for n,v in data['hard_gates'].items() if not v]
    for context in CONTEXTS:
        cancellation[context]={}
        for budget in (10,15):
            physics=data['comparisons'][context][str(budget)]['physics']
            if not physics['tracking_safe']: failures.append(f'{context}/b{budget}: unsafe tracking')
            if physics['path_ordering']=='FAIL': failures.append(f'{context}/b{budget}: path ordering')
            elif physics['path_ordering']!='PASS': misses.append(f'{context}/b{budget}: path ordering borderline')
            if any(v['practical']=='FAIL' for v in physics['components'].values()):
                failures.append(f'{context}/b{budget}: practical mandatory failure')
            if budget==15 and any(v['strict']=='FAIL' for v in physics['components'].values()):
                failures.append(f'{context}/b15: strict mandatory failure')
            if budget==15 and any(v['strict']!='PASS' for v in physics['components'].values()):
                misses.append(f'{context}/b15: not all strict PASS')
            flags={n:v for n,v in physics['cancellation'].items() if v['flagged']}
            cancellation[context][str(budget)]=flags
            if any(v['severe'] for v in flags.values()): failures.append(f'{context}/b{budget}: severe cancellation')
            elif flags: misses.append(f'{context}/b{budget}: cancellation flag')
        b10=data['comparisons'][context]['10']; pf=family_rms(b10['physics']['strict_units'])
        controls=[n for n in b10 if n!='physics']; cf={n:family_rms(b10[n]['strict_units']) for n in controls}
        ratios[context]={}
        for family,value in pf.items():
            prefix=cf['prefix'][family]; best=min(v[family] for v in cf.values())
            rp=value/prefix if prefix else (0 if value==0 else math.inf)
            rb=value/best if best else (0 if value==0 else math.inf)
            ratios[context][family]={'vs_prefix':rp,'vs_best_control':rb}
            if rp>=criteria['comparison']['borderline_improvement_ratio']:
                failures.append(f'{context}/{family}: no improvement over prefix')
            if rb>criteria['comparison']['improvement_pass_ratio']:
                misses.append(f'{context}/{family}: <20% improvement over best control')
        for metric,value in b10['physics']['strict_units'].items():
            best=min(b10[n]['strict_units'][metric] for n in controls)
            if value>best+criteria['comparison']['no_harm_slack_strict_units']:
                misses.append(f'{context}/{metric}: no-harm miss')
        b15=data['comparisons'][context]['15']['physics']
        for metric,value in b15['strict_units'].items():
            if value>b10['physics']['strict_units'][metric]+criteria['comparison']['budget_error_growth_slack_strict_units']:
                failures.append(f'{context}/{metric}: b15 regression')
    verdict='FAIL' if failures else 'BORDERLINE' if misses else 'PASS'
    return {'verdict':verdict,'failures':sorted(set(failures)),'pass_requirements_missed':sorted(set(misses)),
            'family_ratios':ratios,'physics_cancellation_flags':cancellation}


def transfer_verdict(selector):
    if selector['verdict']=='FAIL': return {'verdict':'NOT SUPPORTED','reason':'selector validation FAIL'}
    target=('logical_01','comparator_23','primary_78')
    no_prefix=any(selector['family_ratios'][c][g]['vs_prefix']>=1 for c in CONTEXTS for g in target)
    severe=any(v['severe'] for c in selector['physics_cancellation_flags'].values() for b in c.values() for v in b.values())
    if no_prefix or severe: return {'verdict':'NOT SUPPORTED','reason':'target-group or severe-cancellation falsification'}
    strong=selector['verdict']=='PASS' and all(selector['family_ratios'][c][g]['vs_best_control']<=.8 for c in CONTEXTS for g in target)
    no_flags=not any(v for c in selector['physics_cancellation_flags'].values() for v in c.values())
    return {'verdict':'SUPPORTED' if strong and no_flags else 'PARTIAL',
            'reason':'all target groups satisfy strong transfer rule' if strong and no_flags else 'all groups improve prefix but a strong condition is missed'}


def convergence_verdict(data):
    failures=[]; misses=[]
    if not all(data['hard_gates'].values()): failures.append('hard gate')
    for name,row in data['prefix_convergence']['components'].items():
        if row['k260_vs_k300']['practical']=='FAIL' or row['k280_vs_k300']['strict']=='FAIL': failures.append(name)
        if 'step_280_300' in row:
            if abs(row['step_280_300'])>abs(row['step_260_280'])+1e-12: failures.append(f'{name}: final increment did not shrink')
            if row['sign_reversal'] or row['accidental_cancellation']: failures.append(f'{name}: reversal/cancellation')
        if row['k260_vs_k300']['strict']!='PASS' or row['k280_vs_k300']['strict']!='PASS': misses.append(name)
    return {'verdict':'FAIL' if failures else 'BORDERLINE' if misses else 'PASS',
            'failures':sorted(set(failures)),'strict_misses':sorted(set(misses))}


def main():
    p=argparse.ArgumentParser(); p.add_argument('metrics_json',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    criteria=json.loads((HERE/'criteria.json').read_text()); data=json.loads(a.metrics_json.read_text())
    selector=selector_verdict(data,criteria)
    out={'parameter_point_selector_validation':selector,'multi_target_transfer':transfer_verdict(selector),
         'retained_basis_convergence':convergence_verdict(data),
         'endpoint_semantics':'new-point k300 is finite, not ground truth'}
    with a.output.open('x') as f: json.dump(out,f,indent=2,allow_nan=False); f.write('\n')


if __name__=='__main__': main()
