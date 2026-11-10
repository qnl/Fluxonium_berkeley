"""Sandbox state/subspace tracking. Columns are vectors in an explicit common basis.

No implicit padding, energy sorting, or physical parity assumptions. NumPy/SciPy
only. Energies and shift scales must use the same caller-specified units.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import linalg as la
from scipy.optimize import linear_sum_assignment


def _matrix(a):
    a=np.asarray(a,dtype=complex)
    if a.ndim!=2 or min(a.shape)==0 or not np.all(np.isfinite(a)):
        raise ValueError('Expected a finite, nonempty matrix.')
    return a


def normalize_columns(u):
    u=_matrix(u); norms=la.norm(u,axis=0)
    if np.any(norms==0) or not np.all(np.isfinite(norms)): raise ValueError('Invalid vector norm.')
    return u/norms


def overlap_matrix(u,v):
    """Return S_ij=<u_i|v_j>; individual vectors normalized, not mixed."""
    u=normalize_columns(u); v=normalize_columns(v)
    if u.shape[0]!=v.shape[0]: raise ValueError('Supply an explicit common-basis embedding.')
    return u.conj().T@v


@dataclass(frozen=True)
class AmbiguityPolicy:
    # Saved Gridium legacy thresholds were AMPLITUDES .9, margin .25, ratio .5.
    # Keep the amplitude margin explicit: squaring a difference is not valid.
    min_probability: float=.9**2
    min_amplitude_margin: float=.25
    max_competitor_probability_ratio: float=.5**2
    subspace_min_probability: float=.9**2

    def __post_init__(self):
        if not all(0<=x<=1 for x in asdict(self).values()):
            raise ValueError('Policy thresholds must lie in [0,1].')


def assign_scores(probability,method='hungarian'):
    """Maximum total probability vs global-largest-available-edge greedy.

    Greedy tie order is lexicographic. Rectangular cases return -1 for unmatched
    references. A forced poor Hungarian match is NOT a resolved identity.
    """
    p=np.asarray(probability,float)
    if p.ndim!=2 or min(p.shape)==0 or not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1+1e-8):
        raise ValueError('Invalid overlap probabilities.')
    nr,nc=p.shape; assignment=np.full(nr,-1,int)
    if method=='hungarian':
        r,c=linear_sum_assignment(-p); assignment[r]=c
    elif method=='greedy':
        available=p.copy()
        for _ in range(min(nr,nc)):
            i,j=np.unravel_index(np.argmax(available),available.shape)
            assignment[i]=j; available[i,:]=-np.inf; available[:,j]=-np.inf
    else: raise ValueError('Unknown assignment method.')
    return assignment


def track_states(u,v,reference_energies=None,candidate_energies=None,*,
                 policy=AmbiguityPolicy(),method='hungarian',shift_scales=None,
                 previous_assignment=None,score_error_bound=None,
                 reference_energy_indices=None,candidate_energy_indices=None):
    s=overlap_matrix(u,v); p=abs(s)**2; nr,nc=p.shape
    assignment=assign_scores(p,method); rows=[]
    if (reference_energies is None)!=(candidate_energies is None):
        raise ValueError('Provide both energy arrays or neither.')
    er=None if reference_energies is None else np.asarray(reference_energies,float)
    ev=None if candidate_energies is None else np.asarray(candidate_energies,float)
    if er is not None and (er.shape!=(nr,) or ev.shape!=(nc,) or not np.all(np.isfinite(er)) or not np.all(np.isfinite(ev))):
        raise ValueError('Energy dimensions/finiteness invalid.')
    if (reference_energy_indices is None)!=(candidate_energy_indices is None):
        raise ValueError('Provide both global energy-index arrays or neither.')
    if reference_energy_indices is not None:
        rank_u=np.asarray(reference_energy_indices);rank_v=np.asarray(candidate_energy_indices)
        if rank_u.shape!=(nr,) or rank_v.shape!=(nc,): raise ValueError('Energy-index dimensions invalid.')
    elif er is not None and nr==nc:
        rank_u=np.argsort(np.argsort(er));rank_v=np.argsort(np.argsort(ev))
    else: rank_u=rank_v=None  # Unequal candidate windows do not share local ranks.
    scales=None if shift_scales is None else np.broadcast_to(np.asarray(shift_scales,float),(nr,))
    if scales is not None and (er is None or np.any(scales<0) or not np.all(np.isfinite(scales))):
        raise ValueError('Finite nonnegative shift scales require energies.')
    if previous_assignment is not None and np.shape(previous_assignment)!=(nr,):
        raise ValueError('Previous assignment must use the same label/window convention.')
    for i,j in enumerate(assignment):
        reasons=[]
        alternatives=[k for k in range(nc) if k!=j]
        other=max(alternatives,key=lambda k:p[i,k]) if alternatives else None
        score=float(p[i,j]) if j>=0 else 0.
        competitor=float(p[i,other]) if other is not None else 0.
        col=max([p[k,j] for k in range(nr) if k!=i],default=0.) if j>=0 else 0.
        margin=score-competitor; amp_margin=np.sqrt(score)-np.sqrt(competitor)
        if j<0: reasons.append('unmatched')
        if score<policy.min_probability: reasons.append('low_overlap')
        if other is not None and amp_margin<policy.min_amplitude_margin: reasons.append('row_competition')
        if competitor>=policy.max_competitor_probability_ratio*score and competitor>0: reasons.append('row_ratio')
        if col>=policy.max_competitor_probability_ratio*score and col>0: reasons.append('column_competition')
        changed=previous_assignment is not None and int(previous_assignment[i])!=j
        if changed: reasons.append('assignment_changed')
        energy_neighbors=[]
        if scales is not None:
            energy_neighbors=[k for k in range(nr) if k!=i and abs(er[k]-er[i])<=scales[k]+scales[i]]
            if energy_neighbors: reasons.append('gap_comparable_to_step_scale')
        rows.append(dict(reference=i,candidate=int(j),assigned_probability=score,
            assigned_amplitude=float(np.sqrt(score)),best_other_candidate=other,
            second_best_probability=competitor,probability_margin=float(margin),
            amplitude_margin=float(amp_margin),column_competitor_probability=float(col),
            energy_displacement=None if er is None or j<0 else float(ev[j]-er[i]),
            energy_rank_changed=None if rank_u is None or j<0 else bool(rank_u[i]!=rank_v[j]),
            assignment_changed=bool(changed),energy_watch_neighbors=energy_neighbors,
            ambiguous=bool(reasons),reasons=reasons))
    score=sum(p[i,j] for i,j in enumerate(assignment) if j>=0)
    # Exact second-best global score: every distinct matching excludes at least
    # one selected edge. Useful even when each row's best choice looks unique.
    optimum=assign_scores(p); optscore=sum(p[i,j] for i,j in enumerate(optimum) if j>=0)
    best_alternative=-np.inf
    for i,j in enumerate(optimum):
        if j<0: continue
        cost=-p.copy(); cost[i,j]=np.inf
        try:
            rr,cc=linear_sum_assignment(cost)
        except ValueError: continue
        best_alternative=max(best_alternative,float(p[rr,cc].sum()))
    gap=None if not np.isfinite(best_alternative) else float(max(0.,optscore-best_alternative))
    if score_error_bound is not None and (not np.isfinite(score_error_bound) or score_error_bound<0):
        raise ValueError('Score error bound must be finite and nonnegative.')
    robust=None if score_error_bound is None else bool(np.array_equal(assignment,optimum)
        and (gap is None or gap>2*min(nr,nc)*score_error_bound))
    return dict(overlap=s,probability=p,assignment=assignment,rows=rows,method=method,
        total_probability=float(score),optimal_total_probability=float(optscore),
        global_assignment_score_gap=gap,assignment_certified_for_score_error=robust,
        policy=asdict(policy),candidate_window_incomplete=bool(nc<nr))


def orthonormal_frame(u,rcond=1e-12):
    """Symmetric Gram whitening, preserving the span; rejects rank deficiency.

    Returns Q=U C and C so operator blocks can be transformed consistently.
    rcond is a numerical rank cutoff, not a scientific overlap threshold.
    """
    if not np.isfinite(rcond) or not 0<rcond<1: raise ValueError('rcond must be in (0,1).')
    u=_matrix(u); gram=u.conj().T@u
    w,z=la.eigh((gram+gram.conj().T)/2)
    if w[0]<=rcond*w[-1]: raise ValueError('Rank-deficient or ill-conditioned subspace frame.')
    c=(z/np.sqrt(w))@z.conj().T; q=u@c
    return q,c,dict(gram_eigenvalues=w,gram_error=float(la.norm(gram-np.eye(len(w)),2)),
                    gram_condition=float(w[-1]/w[0]))


def track_subspaces(u,v,rcond=1e-12):
    """Principal angles and projector distances without ambient projectors.

    Unequal ranks are allowed, but no square Procrustes rotation is returned.
    A minimum singular value of one does NOT mean equal unequal-rank spaces.
    """
    u=_matrix(u); v=_matrix(v)
    if u.shape[0]!=v.shape[0]: raise ValueError('Supply a common-basis embedding.')
    qu,cu,gu=orthonormal_frame(u,rcond); qv,cv,gv=orthonormal_frame(v,rcond)
    s=qu.conj().T@qv; left,sing,right=la.svd(s,full_matrices=False)
    if sing[0]>1+1e-8: raise ValueError('Invalid orthonormal frames.')
    sing=np.clip(sing,0,1); ru,rv=u.shape[1],v.shape[1]
    angles=np.arccos(sing)
    # Stable near-coincident angles use actual residuals rather than 1-sigma^2.
    residual=qv-qu@s
    if ru==rv:
        sin_angles=np.clip(la.svdvals(residual)[::-1],0,1)
        angles=np.arctan2(sin_angles,sing)
    overlap=float(np.sum(sing**2))
    fro=float(np.sqrt(max(0.,ru+rv-2*overlap)))
    if ru==rv: fro=float(np.sqrt(2)*la.norm(residual,'fro'))
    op=1. if ru!=rv else float(la.norm(residual,2))
    rotation=right.conj().T@left.conj().T if ru==rv else None
    return dict(singular_values=sing,principal_angles_degrees=np.degrees(angles),
        projector_frobenius=fro,projector_operator=op,chordal_distance=fro/np.sqrt(2),
        minimum_singular_value=float(sing[-1]),mean_subspace_overlap=2*overlap/(ru+rv),
        reference_coverage=overlap/ru,candidate_coverage=overlap/rv,
        reference_rank=ru,candidate_rank=rv,reference_gram=gu,candidate_gram=gv,
        reference_frame=qu,candidate_frame=qv,reference_transform=cu,candidate_transform=cv,
        overlap=s,rotation=rotation,
        alignment_unique=bool(ru==rv and sing[-1]>rcond))


def block_invariants(block,hermitian=False):
    b=_matrix(block); s=la.svdvals(b)
    out=dict(singular_values=s,frobenius=float(la.norm(b,'fro')),operator_norm=float(s[0]))
    if hermitian:
        if b.shape[0]!=b.shape[1] or not np.allclose(b,b.conj().T,atol=1e-10,rtol=1e-10):
            raise ValueError('Hermitian internal block required.')
        out.update(trace=np.trace(b),trace_square=np.trace(b@b),eigenvalues=la.eigvalsh(b))
    return out


def compare_blocks(reference,candidate,left_rotation,right_rotation=None,*,hermitian=False):
    """Rotations map candidate orthonormal frames into reference gauges.

    For rectangular source/target blocks use their separate left/right rotations.
    Nonorthogonal input blocks must first be transformed by whitening matrices.
    """
    a=_matrix(reference); b=_matrix(candidate); l=_matrix(left_rotation)
    r=l if right_rotation is None else _matrix(right_rotation)
    for q in [l,r]:
        if q.shape[0]!=q.shape[1] or not np.allclose(q.conj().T@q,np.eye(len(q)),atol=1e-9,rtol=0):
            raise ValueError('Alignment must be unitary.')
    aligned=l.conj().T@b@r
    if aligned.shape!=a.shape: raise ValueError('Incompatible operator block sizes.')
    delta=aligned-a
    return dict(reference=block_invariants(a,hermitian),candidate=block_invariants(b,hermitian),
        aligned_candidate=aligned,aligned_difference_frobenius=float(la.norm(delta,'fro')),
        aligned_difference_operator=float(la.norm(delta,2)))


def ambiguity_clusters(result,reference_vectors,candidate_vectors,policy=None):
    """Competition-connected proposed clusters, never automatic identity claims.

    An energy warning alone proposes joint monitoring. Poor cluster capture asks
    for a larger candidate window or smaller step; it does not certify a label.
    """
    if policy is None: policy=AmbiguityPolicy(**result['policy'])
    p=result['probability']; assignment=result['assignment']; nr,nc=p.shape
    graph={i:set() for i in range(nr)}; owner={int(j):i for i,j in enumerate(assignment) if j>=0}
    for row in result['rows']:
        i=row['reference']; score=row['assigned_probability']
        if not row['ambiguous']: continue
        for j in range(nc):
            if p[i,j]>0 and p[i,j]>=policy.max_competitor_probability_ratio*max(score,1e-30) and j in owner:
                k=owner[j]
                if k!=i: graph[i].add(k); graph[k].add(i)
        j=assignment[i]
        if j>=0:
            for k in range(nr):
                if k!=i and p[k,j]>0 and p[k,j]>=policy.max_competitor_probability_ratio*max(score,1e-30):
                    graph[i].add(k); graph[k].add(i)
        for k in row['energy_watch_neighbors']: graph[i].add(k); graph[k].add(i)
    found=[]; seen=set()
    for root in range(nr):
        if root in seen: continue
        todo=[root]; comp=set()
        while todo:
            i=todo.pop()
            if i in comp: continue
            comp.add(i); todo.extend(graph[i]-comp)
        seen|=comp; refs=sorted(comp)
        if not any(result['rows'][i]['ambiguous'] for i in refs): continue
        cs=sorted({int(assignment[i]) for i in refs if assignment[i]>=0})
        diagnostic=track_subspaces(reference_vectors[:,refs],candidate_vectors[:,cs]) if cs else None
        captured=diagnostic is not None and len(refs)==len(cs) and diagnostic['minimum_singular_value']**2>=policy.subspace_min_probability
        boundary=sorted({j for i in refs for j in range(nc) if j not in cs and p[i,j]>0
            and p[i,j]>=policy.max_competitor_probability_ratio*max(result['rows'][i]['assigned_probability'],1e-30)})
        energy_only=all(set(result['rows'][i]['reasons'])<= {'gap_comparable_to_step_scale'} for i in refs)
        if not captured or boundary: status='expand_window_or_refine_step'
        elif len(refs)>1: status='track_subspace'
        elif energy_only: status='energy_watch'
        else: status='individual_unresolved'
        found.append(dict(reference_states=refs,candidate_states=cs,status=status,
            competing_candidates_outside_cluster=boundary,diagnostic=diagnostic))
    return found


def embed_retained(vectors,indices,total_parents,oscillator_dimension):
    """Explicit (retained-parent, oscillator) -> (full-parent, oscillator) map."""
    v=_matrix(vectors); original=np.asarray(indices);idx=np.asarray(indices,int)
    if np.any(original!=idx): raise ValueError('Parent indices must be integers.')
    if idx.ndim!=1 or len(set(idx.tolist()))!=len(idx) or np.any(idx<0) or np.any(idx>=total_parents):
        raise ValueError('Invalid retained parent indices.')
    if len(v)!=len(idx)*oscillator_dimension: raise ValueError('Tensor dimension mismatch.')
    full=np.zeros((total_parents,oscillator_dimension,v.shape[1]),complex)
    full[idx]=v.reshape(len(idx),oscillator_dimension,-1)
    return full.reshape(total_parents*oscillator_dimension,-1)


def observable_frame(block,gap_tolerance):
    """Observable-defined descendants; eigenvectors need not be energy eigenstates.

    gap_tolerance is caller-supplied observable uncertainty/resolution. Degenerate
    observable eigenvalues remain clusters, rather than falsely restored labels.
    """
    b=_matrix(block)
    block_invariants(b,True)
    if not np.isfinite(gap_tolerance) or gap_tolerance<0: raise ValueError('Invalid gap tolerance.')
    w,v=la.eigh(b); groups=[]; start=0
    for k in range(len(w)-1):
        if w[k+1]-w[k]>gap_tolerance: groups.append(list(range(start,k+1))); start=k+1
    groups.append(list(range(start,len(w))))
    return dict(values=w,rotation=v,groups=groups,individual_labels_resolved=all(len(g)==1 for g in groups))


def pathway_signature(entry,internal,resolvent=None):
    """Gauge-invariant cluster coupling diagnostics, NOT a driven gate rate.

    entry=C=Q_cluster^dagger Phi Q_logical. For R=f(H_cluster), C^dagger R B R C
    is an invariant operator word. Photon orderings/denominators are not inferred.
    """
    c=_matrix(entry); b=_matrix(internal)
    r=np.eye(b.shape[0]) if resolvent is None else _matrix(resolvent)
    return dict(entry=block_invariants(c),internal=block_invariants(b,True),
        logical_gram=c.conj().T@c,operator_word=c.conj().T@r@b@r@c)
