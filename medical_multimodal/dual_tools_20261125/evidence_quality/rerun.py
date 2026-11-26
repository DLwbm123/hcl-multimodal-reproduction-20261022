from grounding import locate
from med import main
from observation import visual_tools
import veto


def experiment(cases,refs,split,context,call,policy,available,visible,answer_from,correct,choose,gen):
    return veto.run(cases,refs,split,context,visual_tools(call,visible,choose,gen),policy,available,visible,
                    answer_from,correct,choose,gen,capture=True)


if __name__=='__main__':main(locator=locate,experiment=experiment)
