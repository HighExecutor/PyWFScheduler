"""
1) Vector representation
2) generate() - generation of initial solutions
3) fitness(p) (with mofit field)
4) mass(p, worst, best)
5) force_vector_matrix(pop, kbest, G)
6) position(p, velocity) - function of getting new position
7) G(g, i) - changing of G-constant, where i - a number of a current iteration
8) kbest(kbest, i) - changing of kbest
"""
import random
import math

from deap.base import Fitness
from src.algs.common.MapOrdSchedule import build_schedule, MAPPING_SPECIE, ORDERING_SPECIE
from src.algs.common.utilities import mapping_as_vector
from src.core.environment.Utility import Utility
from src.experiments.cga.utilities.common import hamming_distances
from src.algs import SimpleRandomizedHeuristic


def generate(wf, rm, estimator):
    sched = SimpleRandomizedHeuristic(wf, rm.get_nodes(), estimator).schedule()
    return schedule_to_position(sched)



def force_vector_matrix(pop, kbest, G, e=0.0):
    """
    returns matrix of VECTORS size of 'pop_size*kbest'
    distance between two vectors is estimated with hamming distance
    Note: pop is sorted by decreasing of goodness, so to find kbest
    we only need to take first kbest of pop
    """
    sub = lambda seq1, seq2: [0 if s1 == s2 else 1 for s1, s2 in zip(seq1, seq2)]
    zero = lambda: [0 for _ in range(len(pop[0]))]

    def estimate_force(a, b):
        a_string = a.as_vector()
        b_string = b.as_vector()

        R = hamming_distances(a_string, b_string)
        ## TODO: here must be a multiplication of a vector and a number
        val = (G*(a.mass*b.mass)/R + e)
        f = [val * d for d in sub(a_string, b_string)]
        return f

    mat = [[zero() if p == b else estimate_force(p, b) for b in pop[0:kbest]] for p in pop]
    return mat

def position(wf, rm, estimator, position, velocity):
    ## TODO: do normal architecture of relations in the first place
    ## TODO: rework it in an elegant way
    raise NotImplementedError()
    unchecked_tasks = wf.get_all_unique_tasks()
    def change(d):
        if d.startswith("ID"):
            s = set(node.name for node in rm.get_nodes())
            s.remove(d)
            s = list(s)
            new_name = d if len(s) == 0 else s[random.randint(0, len(s) - 1)]
        else:

            s = set(t.id for t in tasks)
            s.remove(d)
            s = [el for el in s if not el.checked]
            ## TODO: add condition for checking of precedence
            if len(s) == 0:
                ## TODO: check case
                new_name = d
            else:
                while len(s) > 0:
                    el = s[random.randint(0, len(s) - 1)]
                    task = wf.byId(el)
                    if all(is_checked(el) for p in task.parents):
                        task.checked = True
                        new_name = el
                        break
                    else:
                        s.remove(el)
        pass
    threshold = 0.4
    new_vector = [change(d) if vd > threshold else d for vd, d in zip(velocity, position.as_vector())]
    new_position = Position.from_vector(new_vector)
    return new_position

def G(ginit, i, iter_number, all_iter_number=None):
    ng = ginit*(1 - i/iter_number) if all_iter_number is None else ginit*(1 - i/all_iter_number)
    return ng

def Kbest(kbest_init, kbest, i, iter_number, all_iter_number=None):
    """
    basic implementation of kbest decreasing
    """
    iter_number = iter_number if all_iter_number is None else all_iter_number
    d = iter_number / kbest_init
    nkbest = math.ceil(abs(kbest_init - i/d))
    return nkbest


def schedule_to_position(schedule):
    """
    this function converts valid schedule
    to mapping and ordering strings
    """
    items = lambda: iter((item, node) for node, items in schedule.mapping.items() for item in items)
    if not all(i.is_unstarted() for i, _ in items()):
        raise ValueError("Schedule is not valid. Not all elements have unstarted state.")

    mapping = {i.job.id: n.name for i, n in items()}
    ordering = sorted([i for i, _ in items()], key=lambda x: x.start_time)
    ordering = [el.job.id for el in ordering]
    return Position(mapping, ordering)