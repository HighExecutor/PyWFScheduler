import functools
import numpy

from src.algs.ga.GARunner import MixRunner
from src.algs.heft.DSimpleHeft import run_heft
from src.core.CommonComponents.ExperimentalManagers import ExperimentResourceManager
from src.core.environment.Utility import wf, Utility
from src.experiments.cga.mobjective.utility import SimpleTimeCostEstimator
from src.experiments.cga.utilities.common import UniqueNameSaver, repeat
from src.core.environment.ResourceGenerator import ResourceGenerator as rg


wf_names = ['Montage_100']
# wf_names = ['Montage_50']
# wf_names = ['Montage_500']
# wf_names = ['CyberShake_100']
# wf_names = ['Epigenomics_100']
# wf_names = ["CyberShake_50"]

only_heft = False

PARAMS = {
    "ideal_flops": 20,
    "is_silent": False,
    "is_visualized": False,
    "ga_params": {
        "Kbest": 5,
        "population": 50,
        "crossover_probability": 0.9, #0.3
        "replacing_mutation_probability": 0.9, #0.1
        "sweep_mutation_probability": 0.3, #0.3
        "generations": 300
    },
    "nodes_conf": [10, 15, 25, 30],
    "transfer_time": 100,
    "heft_initial": False
}

run = functools.partial(MixRunner(), **PARAMS)
directory = "../../temp/ga_vs_heft_exp"
saver = UniqueNameSaver("../../temp/ga_vs_heft_exp")

# def do_exp():
#     ga_makespan, heft_makespan, ga_schedule, heft_schedule = run(wf_names[0])
#     saver(ga_makespan)
#     return ga_makespan

def do_exp_schedule(takeHeftSchedule=True):
    saver = UniqueNameSaver("../../temp/ga_vs_heft_exp_heft_schedule")

    ga_makespan, heft_makespan, ga_schedule, heft_schedule, logbook = run(wf_names[0])

    ## TODO: pure hack

    schedule = heft_schedule if takeHeftSchedule else ga_schedule

    mapping = [(item.job.id, node.flops) for node, items in schedule.mapping.items() for item in items]
    mapping = sorted(mapping, key=lambda x: x[0])

    ordering = [(item.job.id, item.start_time) for node, items in heft_schedule.mapping.items() for item in items]
    ordering = [t for t, time in sorted(ordering, key=lambda x: x[1])]

    data = {
        "mapping": mapping,
        "ordering": ordering
    }

    name = saver(data)
    return ga_makespan, heft_makespan, ga_schedule, heft_schedule, name, logbook

def do_exp_heft_schedule():
    res = do_exp_schedule(True)
    return (res[0], res[5])

def do_exp_ga_schedule():
    res = do_exp_schedule(False)
    return (res[0], res[4])


if __name__ == '__main__':
    print("Population size: " + str(PARAMS["ga_params"]["population"]))

    _wf = wf(wf_names[0])
    rm = ExperimentResourceManager(rg.r(PARAMS["nodes_conf"]))
    estimator = SimpleTimeCostEstimator(comp_time_cost=0, transf_time_cost=0, transferMx=None,
                                            ideal_flops=PARAMS["ideal_flops"], transfer_time=PARAMS["transfer_time"])

    heft_schedule = run_heft(_wf, rm, estimator)
    heft_makespan = Utility.makespan(heft_schedule)
    overall_transfer = Utility.overall_transfer_time(heft_schedule, _wf, estimator)
    overall_execution = Utility.overall_execution_time(heft_schedule)

    print("Heft makespan: {0}, Overall transfer time: {1}, Overall execution time: {2}".format(heft_makespan,
                                                                                               overall_transfer,
                                                                                               overall_execution))

    if not only_heft:
        exec_count = 100
        gen = PARAMS["ga_params"]["generations"]
        res_list = [0 for _ in range(gen)]
        result = repeat(do_exp_heft_schedule, exec_count)
        mean = numpy.mean([makespan for (makespan, list) in result])
        for i in range(exec_count):
            cur_list = result[i][1]
            print(str(cur_list))
            for j in range(gen):
                res_list[j] = res_list[j] + cur_list[j]
        for j in range(gen):
                res_list[j] = res_list[j] / exec_count
        print(str(res_list))

        #file = open("C:\Melnik\Experiments\Work\PSO_compare\populations\GA with HEFT cyber.txt", 'w')
        #file.write("#gen    result" + "\n")
        #for i in range(gen):
        #    file.write(str(i) + "   " + str(res_list[i]) + "\n")

        #profit = (1 - mean / heft_makespan) * 100
        #print(result)
        print("Heft makespan: {0}, Overall transfer time: {1}, Overall execution time: {2}".format(heft_makespan,
                                                                                               overall_transfer,
                                                                                               overall_execution))
        print("Mean: {0}".format(mean))
        #print("Profit: {0}".format(profit))


