"""Full-source development gate; unknown references never become rewards."""
def evaluate(inputs,episodes,targets):
    ids={x['id'] for x in inputs};expected={(i,a) for i in ids for a in ['Fixed','memory']}
    keys=[(x['id'],x['arm']) for x in episodes];assert len(keys)==len(set(keys)) and set(keys)<=expected
    assert set(targets)=={x['id'] for x in inputs if x['semantic_feedback_allowed']}
    by={(x['id'],x['arm']):x for x in episodes};stats={}
    for role in ['current','anchor']:
        subset=[x for x in inputs if x['gate_role']==role and x['id'] in targets]
        correct=lambda x,a:by.get((x['id'],a),{}).get('status')=='completed' and by.get((x['id'],a),{}).get('predicted')==targets[x['id']]
        gain=sum(not correct(x,'Fixed') and correct(x,'memory') for x in subset)
        loss=sum(correct(x,'Fixed') and not correct(x,'memory') for x in subset)
        stats[role]={'denominator':len(subset),'Fixed_correct':sum(correct(x,'Fixed') for x in subset),'memory_correct':sum(correct(x,'memory') for x in subset),'gain':gain,'loss':loss,'net':gain-loss}
    cost={a:{'recorded':sum(x['arm']==a for x in episodes),'failed_or_invalid':sum(x['arm']==a and (x['status']!='completed' or x.get('predicted') not in list('ABCDEF')) for x in episodes),'tool_calls':sum(x['tool_calls'] for x in episodes if x['arm']==a),'route_calls':sum(x.get('route_calls',0) for x in episodes if x['arm']==a),'seconds':sum(x['seconds'] for x in episodes if x['arm']==a)} for a in ['Fixed','memory']}
    checks={'full_coverage':set(keys)==expected,'sufficient_qualified_sources':stats['current']['denominator']>=6 and stats['anchor']['denominator']>=2,
      'current_gain':stats['current']['net']>=1,'current_loss':stats['current']['loss']<=1,'historical_no_loss':stats['anchor']['loss']==0,
      'failure_nonincrease':cost['memory']['failed_or_invalid']<=cost['Fixed']['failed_or_invalid'],
      'tool_cost':cost['memory']['tool_calls']<=2*cost['Fixed']['tool_calls']+2}
    return {'qualified_scores':stats,'all_source_costs':cost,'checks':checks,'accepted_for_research_development':all(checks.values()),'clinical_deployment':False,'independent_evaluation':False,'unknown_sources':len(inputs)-len(targets)}
