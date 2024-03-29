from copy import deepcopy
import random
from deap.base import Toolbox
import numpy
from src.algs.pso.rdpsoOrd.ordering_operators import build_schedule, generate, ordering_update, fitness

from src.algs.pso.rdpsoOrd.rdpso import run_pso, initRankList
from src.algs.pso.rdpsoOrd.mapping_operators import update as mapping_update
from src.core.environment.Utility import Utility
from src.experiments.aggregate_utilities import interval_statistics, interval_stat_string
from src.experiments.cga.utilities.common import repeat
from src.experiments.common import AbstractExperiment
from src.algs.heft.HeftHelper import HeftHelper


class RdpsoBaseExperiment(AbstractExperiment):

    @staticmethod
    def run(**kwargs):
        inst = RdpsoBaseExperiment(**kwargs)
        return inst()

    def __init__(self, wf_name, W, C1, C2, GEN, N):
        super().__init__(wf_name)

        self.W = W
        self.C1 = C1
        self.C2 = C2
        self.GEN = GEN
        self.N = N
        pass

    def __call__(self):

        stats, logbook = self.stats(), self.logbook()
        _wf, rm, estimator = self.env()



        wf_dag = HeftHelper.convert_to_parent_children_map(_wf)
        heft_schedule = self.heft_schedule()
        nodes = rm.get_nodes()
        rankList = initRankList(wf_dag, nodes, estimator)

        toolbox = self.toolbox(rankList)

        pop, log, best = run_pso(
            toolbox=toolbox,
            logbook=logbook,
            stats=stats,
            gen_curr=0, gen_step=self.GEN, invalidate_fitness=True, initial_pop=None,
            w=self.W, c1=self.C1, c2=self.C2, n=self.N, rm=rm, wf=_wf, estimator=estimator, rankList=rankList,
        )

        #schedule = build_schedule(_wf, rm, estimator,  best, mapMatrix, rankList, ordFilter)
        schedule = build_schedule(_wf, rm, estimator,  best, rankList)
        Utility.validate_static_schedule(_wf, schedule)
        makespan = Utility.makespan(schedule)
        if (makespan > best.fitness.values[0]):
            print("DANGER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print("Final makespan: {0}".format(makespan))
        #print("Heft makespan: {0}".format(Utility.makespan(heft_schedule)))
        return makespan

    #def toolbox(self, mapMatrix, rankList, ordFilter):
    def toolbox(self, rankList):
        _wf, rm, estimator = self.env()
        heft_schedule = self.heft_schedule()



        heft_particle = generate(_wf, rm, estimator, rankList, heft_schedule)

        #heft_gen = lambda n: ([deepcopy(heft_particle) if random.random() > 1.00 else generate(_wf, rm, estimator, rankList) for _ in range(n-1)] + [deepcopy(heft_particle)])
        heft_gen = lambda n: ([deepcopy(heft_particle) if random.random() > 1.00 else generate(_wf, rm, estimator, rankList) for _ in range(n)])

        def componoud_update(w, c1, c2, p, best, pop):
            mapping_update(w, c1, c2, p.mapping, best.mapping, pop)
            ordering_update(w, c1, c2, p.ordering, best.ordering, pop)

        toolbox = Toolbox()
        toolbox.register("population", heft_gen)
        toolbox.register("fitness", fitness, _wf, rm, estimator)
        toolbox.register("update", componoud_update)
        return toolbox


    pass

if __name__ == "__main__":
    exp = RdpsoBaseExperiment(wf_name="Epigenomics_24",
                              W=0.1, C1=0.6, C2=0.2,
                              GEN=10, N=20)
    result = repeat(exp, 1)
    print(result)
    sts = interval_statistics(result)
    print("Statistics: {0}".format(interval_stat_string(sts)))
    print("Average: {0}".format(numpy.mean(result)))
    pass

