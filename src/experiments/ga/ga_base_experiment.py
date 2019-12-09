from copy import deepcopy
from functools import partial
from statistics import mean
import uuid
from deap import tools
from deap.base import Toolbox
import numpy
from src.algs.ga.GAFunctions2 import GAFunctions2
from src.algs.ga.common_fixed_schedule_schema import run_ga, fit_converter
from src.algs.heft.DSimpleHeft import run_heft
from src.core.CommonComponents.ExperimentalManager import ExperimentResourceManager, ModelTimeEstimator
from src.core.environment.ResourceManager import Schedule
from src.core.environment.Utility import wf, Utility
from src.core.environment.ResourceGenerator import ResourceGenerator as rg
from src.algs.ga.common_fixed_schedule_schema import generate as ga_generate
from src.algs.common.utilities import unzip_result
from src.core.CommonComponents.utilities import repeat
from src.experiments.common import AbstractExperiment


class GABaseExperiment(AbstractExperiment):

    @staticmethod
    def run(**kwargs):
        inst = GABaseExperiment(**kwargs)
        return inst()

    def __init__(self, ga_params=None):
        wf_name = "Montage_25"
        GA_PARAMS = {
            "kbest": 5,
            "n": 25,
            "cxpb": 0.3,  # 0.8
            "mutpb": 0.9,  # 0.5
            "sweepmutpb": 0.3,  # 0.4
            "gen_curr": 0,
            "gen_step": 300,
            "is_silent": False
        }
        if ga_params is None:
            self.GA_PARAMS = GA_PARAMS
        else:
            self.GA_PARAMS = ga_params
        self.wf_name = wf_name

    def __call__(self):
        _wf = wf(self.wf_name)
        rm = ExperimentResourceManager(rg.r([10, 15, 25, 30]))
        estimator = ModelTimeEstimator(bandwidth=10)

        empty_fixed_schedule_part = Schedule({node: [] for node in rm.get_nodes()})

        heft_schedule = run_heft(_wf, rm, estimator)

        fixed_schedule = empty_fixed_schedule_part

        ga_functions = GAFunctions2(_wf, rm, estimator)

        generate = partial(ga_generate, ga_functions=ga_functions,
                           fixed_schedule_part=fixed_schedule,
                           current_time=0.0, init_sched_percent=0.05,
                           initial_schedule=heft_schedule)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        toolbox = Toolbox()
        toolbox.register("generate", generate)
        toolbox.register("evaluate", fit_converter(ga_functions.build_fitness(empty_fixed_schedule_part, 0.0)))
        toolbox.register("clone", deepcopy)
        toolbox.register("mate", ga_functions.crossover)
        toolbox.register("sweep_mutation", ga_functions.sweep_mutation)
        toolbox.register("mutate", ga_functions.mutation)
        # toolbox.register("select_parents", )
        # toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("select", tools.selRoulette)
        pop, logbook, best = run_ga(toolbox=toolbox,
                                logbook=logbook,
                                stats=stats,
                                **self.GA_PARAMS)

        resulted_schedule = ga_functions.build_schedule(best, empty_fixed_schedule_part, 0.0)

        ga_makespan = Utility.makespan(resulted_schedule)
        return (ga_makespan, logbook)


def fix_schedule(res, heft):
    for item in heft.mapping:
        res.mapping[item] = res.mapping[item].append(heft.mapping[item])
    return res

if __name__ == "__main__":
    exp = GABaseExperiment()
    repeat_count = 1
    result, logbooks = unzip_result(repeat(exp, repeat_count))
    # logbook = logbooks_in_data(logbooks)
    #data_to_file("./CyberShake_30_full.txt", 300, logbook)
    print(result)

# if __name__ == "__main__":
#
#     base_ga_params = {
#             "kbest": 5,
#             "n": 5,
#             "cxpb": 0.3,  # 0.8
#             "mutpb": 0.9,  # 0.5
#             "sweepmutpb": 0.3,  # 0.4
#             "gen_curr": 0,
#             "gen_step": 5,
#             "is_silent": False
#     }
#
#     param_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     params_names = ["cxpb", "mutpb", "sweepmutpb"]
#
#     ga_params = deepcopy(base_ga_params)
#     ga_params["cxpb"] = 0.4
#     ga_params["mutpb"] = 0.5
#     ga_params["sweepmutpb"] = 0.3
#     # def buildGaParams(cxpb, mutpb, sweepmutpb):
#     #     ga_params = deepcopy(base_ga_params)
#     #     ga_params["cxpb"] = cxpb
#     #     ga_params["mutpb"] = mutpb
#     #     ga_params["sweepmutpb"] = sweepmutpb
#     #     return ga_params
#     #
#     #
#     # gaParamsSets = [buildGaParams(cxpb, mutpb, sweepmutpb) for cxpb in param_values
#     #             for mutpb in param_values
#     #             for sweepmutpb in param_values]
#
#     # for ga_params in gaParamsSets:
#     exp = GABaseExperiment(ga_params)
#     print("cxpb: {0}, mutpb: {1}, sweepmutpb: {2}".format(ga_params["cxpb"],
#                                                           ga_params["mutpb"],
#                                                           ga_params["sweepmutpb"]))
#     repeat_count = 1
#     makespans, logbooks = unzip_result(repeat(exp, repeat_count))
#     out_line = "{0}\t{1}\t{2}\t{3}".format(ga_params["cxpb"],
#                                            ga_params["mutpb"],
#                                            ga_params["sweepmutpb"],
#                                            mean(makespans))
#     print(makespans)