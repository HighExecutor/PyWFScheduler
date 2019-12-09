from src.algs.heft.DSimpleHeft import run_heft
from src.core.CommonComponents.ExperimentalManager import ExperimentResourceManager, ModelTimeEstimator
from src.core.environment.Utility import wf, Utility
from src.core.environment.ResourceGenerator import ResourceGenerator as rg

ideal_flops = 8.0

rm = ExperimentResourceManager(rg.r([4.0, 8.0, 8.0, 16.0]))

estimator = ModelTimeEstimator(bandwidth=10) # скорость передачи данных 10 MB/sec

def do_exp(wf_name):
    _wf = wf(wf_name)
    heft_schedule = run_heft(_wf, rm, estimator)
    makespan = Utility.makespan(heft_schedule)
    return makespan

if __name__ == "__main__":
    result = do_exp("Montage_25")
    print(result)
