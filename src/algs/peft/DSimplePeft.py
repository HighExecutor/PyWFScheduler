from src.algs.peft.PeftHelper import PeftHelper
from src.core.environment.ResourceManager import Schedule, ScheduleItem
from src.core.environment.Utility import timing
from src.algs.peft.simple_peft import StaticPeftPlanner


class DynamicPeft(StaticPeftPlanner):
    executed_tasks = set()
    def get_nodes(self):
        resources = self.resource_manager.get_resources()
        nodes = PeftHelper.to_nodes(resources)
        return nodes
        # return self.resource_manager.get_nodes()

    def __init__(self, workflow, resource_manager, estimator, oct, ranking=None):
        self.current_schedule = Schedule(dict())
        self.workflow = workflow
        self.resource_manager = resource_manager
        self.estimator = estimator
        self.oct = oct

        self.current_time = 0

        nodes = self.get_nodes()

        self.wf_jobs = self.make_ranking(self.workflow, nodes) if ranking is None else ranking

        # print("A: " + str(self.wf_jobs))

        #TODO: remove it later
        # to_print = ''
        # for job in self.wf_jobs:
        #     to_print = to_print + str(job.id) + " "
        # print(to_print)
        pass

    @timing
    def run(self, current_cleaned_schedule):
        ## current_cleaned_schedule - this schedule contains only
        ## finished and executed tasks, all unfinished and failed have been removed already
        ## current_cleaned_schedule also have down nodes and new added
        ## ALGORITHM DOESN'T CHECK ADDING OF NEW NODES BY ITSELF
        ## nodes contain only available now

        ## 1. get all unscheduled tasks
        ## 2. sort them by rank
        ## 3. map on the existed nodes according to current_cleaned_schedule

        nodes = self.get_nodes()

        for_planning = PeftHelper.get_tasks_for_planning(self.workflow, current_cleaned_schedule)
        ## TODO: check if it sorted properly
        for_planning = set([task.id for task in for_planning])

        sorted_tasks = [task for task in self.wf_jobs if task.id in for_planning]

        # print("P: " + str(sorted_tasks))

        new_sched = self.mapping([(self.workflow, sorted_tasks)], current_cleaned_schedule.mapping, nodes, self.commcost, self.compcost)
        return new_sched

    def endtime(self, job, events):
        """ Endtime of job in list of events """
        for e in events:
            if e.job == job and (e.state == ScheduleItem.FINISHED or e.state == ScheduleItem.EXECUTING or e.state == ScheduleItem.UNSTARTED):
                return e.end_time

def run_peft(workflow, resource_manager, estimator):
    """
    It simply runs peft with empty initial schedule
    and returns complete schedule
    """
    oct = PeftHelper.get_OCT(workflow, resource_manager, estimator)
    peft = DynamicPeft(workflow, resource_manager, estimator, oct)
    nodes = resource_manager.get_nodes()
    init_schedule = Schedule({node: [] for node in nodes})
    return peft.run(init_schedule)



