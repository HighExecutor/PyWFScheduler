##interface Algorithm
import functools
import operator
import itertools
from src.core.environment.BaseElements import Resource


class Algorithm:
    def __init__(self):
        self.resource_manager = None
        self.estimator = None

    def run(self, event):
        pass

##interface ResourceManager
class ResourceManager:
    def __init__(self):
        pass

    ##get all resources in the system
    def get_resources(self):
        raise NotImplementedError()

    def res_by_id(self, id):
        raise NotImplementedError()

    def change_performance(self, node, performance):
        raise NotImplementedError()

    ## TODO: remove duplcate code with HeftHelper
    def get_nodes(self):
        resources = self.get_resources()
        result = set()
        for resource in resources:
            result.update(resource.nodes)
        return result

    def get_nodes_by_resource(self, resource):
        name = resource.name if isinstance(resource, Resource)else resource
        nodes = [node for node in self.get_nodes() if node.resource.name == name]
        ## TODO: debug
        print("Name", name)
        print("Nodes", nodes)

        return nodes

    def byName(self):
        raise NotImplementedError()

##interface Estimator
class Estimator:
    def __init__(self):
        pass

    ##get estimated time of running the task on the node
    def estimate_runtime(self, task, node):
        pass

    ## estimate transfer time between node1 and node2 for data generated by the task
    def estimate_transfer_time(self, node1, node2, task1, task2):
        pass

## element of Schedule
class ScheduleItem:
    UNSTARTED = "unstarted"
    FINISHED = "finished"
    EXECUTING = "executing"
    FAILED = "failed"
    def __init__(self, job, start_time, end_time):
        self.job = job ## either task or service operation like vm up
        self.start_time = start_time
        self.end_time = end_time
        self.state = ScheduleItem.UNSTARTED

    @staticmethod
    def copy(item):
        new_item = ScheduleItem(item.job, item.start_time, item.end_time)
        new_item.state = item.state
        return new_item

    @staticmethod
    def MIN_ITEM():
        return ScheduleItem(None, 10000000, 10000000)

    def is_unstarted(self):
        return self.state == ScheduleItem.UNSTARTED

    def __str__(self):
        return str(self.job.id) + ":" + str(self.start_time) + ":" + str(self.end_time) + ":" + self.state

    def __repr__(self):
        return str(self.job.id) + ":" + str(self.start_time) + ":" + str(self.end_time) + ":" + self.state


class Schedule:
    def __init__(self, mapping):
        ## {
        ##   res1: (task1,start_time1, end_time1),(task2,start_time2, end_time2), ...
        ##   ...
        ## }
        self.mapping = mapping##dict()

    def is_finished(self, task):
        (node, item) = self.place(task)
        if item is None:
            return False
        return item.state == ScheduleItem.FINISHED

    def get_next_item(self, task):
        for (node, items) in self.mapping.items():
            l = len(items)
            for i in range(l):
                if items[i].job.id == task.id:
                    if l > i + 1:
                        return items[i + 1]
                    else:
                        return None
        return None

    def place(self, task):
        for (node, items) in self.mapping.items():
            for item in items:
                if item.job.id == task.id:
                    return (node,item)
        return None

    def change_state_executed(self, task, state):
        for (node, items) in self.mapping.items():
            for item in items:
                if item.job.id == task.id and (item.state == ScheduleItem.EXECUTING or item.state == ScheduleItem.UNSTARTED):
                    item.state = state
        return None

    def place_single(self, task):
        for (node, items) in self.mapping.items():
            for item in items:
                if item.job.id == task.id and (item.state == ScheduleItem.EXECUTING or item.state == ScheduleItem.UNSTARTED):
                    return (node, item)
        return None

    def change_state_executed_with_end_time(self, task, state, time):
        for (node, items) in self.mapping.items():
            for item in items:
                if item.job.id == task.id and item.state == ScheduleItem.EXECUTING:
                    item.state = state
                    item.end_time = time
                    return True
        #print("gotcha_failed_unstarted task: " + str(task))
        return False

    def place_by_time(self, task, start_time):
        for (node, items) in self.mapping.items():
            for item in items:
                if item.job.id == task.id and item.start_time == start_time:
                    return (node,item)
        return None

    def is_executing(self, task):
        for (node, items) in self.mapping.items():
            for item in items:
                if item.job.id == task.id and item.state == ScheduleItem.EXECUTING:
                    return True
        return False


    def change_state(self, task, state):
        (node, item) = self.place(task)
        item.state = state

    # def get_all_unique_tasks_id(self):
    #     ids = set(item.job.id for (node, items) in self.mapping.items() for item in items)
    #     return ids

    def get_all_unique_tasks(self):
        tasks = set(item.job for (node, items) in self.mapping.items() for item in items)
        return tasks

    def get_all_unique_tasks_id(self):
        tasks = self.get_all_unique_tasks()
        ids = set(t.id for t in tasks)
        return ids

    def get_unfailed_taks(self):
        return [item.job for (node, items) in self.mapping.items()
                for item in items if item.state == ScheduleItem.FINISHED or
                item.state == ScheduleItem.EXECUTING or item.state == ScheduleItem.UNSTARTED]

    def get_unfailed_tasks_ids(self):
        return [job.id for job in self.get_unfailed_taks()]

    def task_to_node(self):
        """
        This operation is applicable only for static scheduling.
        i.e. it is assumed that each is "executed" only once and only on one node.
        Also, all tasks must have state "Unstarted".
        """
        all_items = [item for node, items in self.mapping.items() for item in items]
        assert all(it.state == ScheduleItem.UNSTARTED for it in all_items),\
            "This operation is applicable only for static scheduling"
        t_to_n = {item.job: node for (node, items) in self.mapping.items() for item in items}
        return t_to_n

    def tasks_to_node(self):
        ## there can be several instances of a task due to fails of node
        ## we should take all possible occurences
        task_instances = itertools.groupby(((item.job.id, item , node) for (node, items) in self.mapping.items() for item in items),
                          key=lambda x: x[0])
        task_instances = {task_id: [(item, node) for _, item, node in group]
                          for task_id, group in task_instances}
        return task_instances

    ## TODO: there is duplicate functionality Utility.check_and_raise_for_fixed_part
    # def contains(self, other):
    #     for node, other_items in other.mapping.items():
    #         if node not in self.mapping:
    #             return False
    #         this_items = self.mapping[node]
    #         for i, item in enumerate(other_items):
    #             if len(this_items) <= i:
    #                 return False
    #             if item != this_items[i]:
    #                 return False
    #     return True


    @staticmethod
    def insert_item(mapping, node, item):
        result = []
        i = 0
        try:
            while i < len(mapping[node]):
                ## TODO: potential problem with double comparing
                if mapping[node][i].start_time >= item.end_time:
                    break
                i += 1
            mapping[node].insert(i, item)
        except:
            k = 1


    def get_items_in_time(self, time):
        pass

    ## gets schedule consisting of only currently running tasks
    def get_schedule_in_time(self, time):
        pass

    def get_the_most_upcoming_item(self, time):
        pass

    def __str__(self):
        return str(self.mapping)

    def __repr__(self):
        return str(self.mapping)


##interface Scheduler
class Scheduler:
    def __init__(self):
        ##previously built schedule
        self.old_schedule = None
        self.resource_manager = None
        self.estimator = None
        self.executor = None
        self.workflows = None

    ## build and returns new schedule
    def schedule(self):
        pass
