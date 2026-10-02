#!/usr/bin/env python3
"""Supplementary protocol edge-case receipt requested by independent review."""
import json
import numpy as np
from r13_evaluate import Episode, episode_stats, build_fit_cache, fit_cached, FAMILIES, KS

def ep(name, group, t, stage, score):
    return Episode(name,group,'p',np.asarray(t),np.asarray(stage),np.asarray(score,float))

def main():
    checks={}
    # Synthetic role leakage must be rejected by the same set-disjoint invariant used formally.
    roles={'train':[ep('a','shared',[13],[0],[.1])],
           'calibration':[ep('b','shared',[13],[0],[.1])],
           'validation':[ep('c','v',[13],[0],[.1])]}
    sets=[{e.group for e in roles[x]} for x in ('train','calibration','validation')]
    try:
        if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
            raise ValueError('role leakage')
    except ValueError:
        checks['synthetic_cross_role_leakage_rejected']=True
    else: raise AssertionError('leakage was not rejected')
    # First gross segment at a fragment start is left censored; a later terminal event is right-boundary.
    cens=ep('c','g',[13,14,15,16],[2,2,0,2],[.9,.9,.1,.9])
    s=episode_stats(cens,.5,1)
    assert s['left_censored_events']==1 and s['events']==1 and s['right_boundary_events']==1
    checks['left_and_right_censoring']=True
    # Prior alarm can release on first gross frame, so prior-alarm is not synonymous with delay zero.
    rel=ep('r','g',[13,14,15],[0,2,2],[.9,.1,.9])
    s=episode_stats(rel,.5,1)
    assert s['preexisting_alarm_events']==1 and s['preexisting_alarm_delay0_events']==0
    assert s['event_hits']==1 and s['delay_sum']==1
    checks['prior_alarm_release_counterexample']=True
    # Weighted group multiplicity equals literal complete-group duplication; trajectories are never joined.
    base=[ep('a1','g1',[13,14,15],[0,1,2],[.2,.8,.8]),
          ep('a2','g1',[13,14],[0,2],[.7,.7]),
          ep('b1','g2',[13,14,15,16],[0,0,1,2],[.1,.4,.9,.9])]
    weights={'g1':2,'g2':1}
    literal=[ep(f'{e.episode}_copy{j}',e.group,e.t.copy(),e.stage.copy(),e.score.copy())
             for e in base for j in range(weights[e.group])]
    comparisons=0
    for k in KS:
        weighted_cache=build_fit_cache(base,k); literal_cache=build_fit_cache(literal,k)
        for family in FAMILIES:
            for alpha in (.01,.05,.10):
                a=fit_cached(weighted_cache,family,alpha,weights)
                b=fit_cached(literal_cache,family,alpha)
                assert a['status']==b['status'] and a['threshold']==b['threshold']
                comparisons+=1
    checks['literal_complete_group_copy_equivalence']=comparisons
    print(json.dumps({'status':'pass','checks':checks},indent=2))
if __name__=='__main__':main()
