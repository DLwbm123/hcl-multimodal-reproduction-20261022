import re


def normalized(text):
    return ' '.join(text.casefold().split())


def quote_is_observed(quote,notes):
    quote=normalized(quote.strip().strip('"“”\''))
    if quote in ('','none') or len(quote)<4 or not re.search(r'\w',quote):
        return False
    return any(quote in normalized(n['text'][:450]) for n in notes)


def decide(choose,gen,q,answer,trusted,planned,skill):
    if not trusted:
        return False,'continue_no_accepted_evidence'
    evidence='\n'.join(n['name']+': '+n['text'][:450] for n in trusted)
    quote=gen('Question: '+q+'\nAccepted observations:\n'+evidence+
              '\nProposed answer: '+answer+'\nNext tool proposed: '+planned+
              '\nCopy ONE contiguous verbatim passage from an accepted observation that establishes the answer. '
              'If none is sufficient, output NONE. Return only the passage or NONE.',100).strip()
    if not quote_is_observed(quote,trusted):
        return False,'continue_no_matching_quote'
    state='Question: '+q+'\nProposed answer: '+answer+'\nVerbatim evidence: '+quote+'\nNext tool: '+planned
    lead=(skill+'\nDecide whether the quotation establishes all essential facts in the answer that the question asks for. '
          'Mere mention of the subject is insufficient. ')
    # Require agreement after exchanging the token meanings.
    for sufficient in 'AB':
        labels={sufficient:'sufficient',('B' if sufficient=='A' else 'A'):'insufficient'}
        if choose(state,{},lead+'A = '+labels['A']+'. B = '+labels['B']+'. Select one.',list('AB'))!=sufficient:
            return False,'continue_quote_not_verified'
    return True,'stop_verified_quote'


def ablate(choose,state,rule,answer,planned):
    masked=state.replace('\nCurrent answer: '+answer+'\nLast result','\nCurrent answer: (withheld)\nLast result',1)
    assert masked!=state
    outcomes={}
    for shown,s in [('draft',state),('masked',masked)]:
        for action,continuation in [('generic','continue acquiring evidence'),
                                    ('named','execute '+planned+' to obtain its observation before answering')]:
            instruction=rule+'\nA = stop and submit current answer. B = '+continuation+'. Select one.'
            outcomes[shown+'_'+action]=choose(s,{},instruction,list('AB'))=='A'
    return outcomes
