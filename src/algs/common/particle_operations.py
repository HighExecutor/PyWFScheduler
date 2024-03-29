from numbers import Number
from uuid import uuid4
import math
from src.algs.common.individuals import FitAdapter

"""
This file contains classes of particles for PSO and GSA
with methods of transformations from combinatorial to continious spaces
and vice versa
"""

##TODO: it can be upgraded with numpy classes to speed up execution
##TODO: write test cases


class Particle(FitAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uid = uuid4()
        self._velocity = None
        self._best = None

    def _get_best(self): return self._best
    def _set_best(self, b): self._best = b

    def _get_velocity(self): return self._velocity
    def _set_velocity(self, v): self._velocity = v

    best = property(_get_best, _set_best)
    velocity = property(_get_velocity, _set_velocity)
    pass



class MappingParticle(Particle):

    def __init__(self, mapping):
        super().__init__(mapping)
        self.velocity = MappingParticle.Velocity({})
    pass

    def __sub__(self, other):
        # return Position({k: self[k] for k in self.keys() - other.keys()})
        return MappingParticle.Velocity({item: 1.0 for item in self.entity.items()# - other.entity.items()
        })

    def __mul__(self, other):
        if isinstance(other, Number):
            return MappingParticle.Velocity({k: other for k, v in self.entity.items()})
        raise ValueError("Other has not a suitable type for multiplication")

    def emptify(self):
        return MappingParticle.Velocity({})

    class Velocity(dict):
        def __mul__(self, other):
            if isinstance(other, Number):
                if other < 0:
                    raise ValueError("Only positive numbers can be used for operations with velocity")
                return MappingParticle.Velocity({k: 1.0 if v * other > 1.0 else v * other for k, v in self.items()})
            raise ValueError("{0} has not a suitable type for multiplication".format(other))

        def __add__(self, other):
            vel = MappingParticle.Velocity({k: max(self.get(k, 0), other.get(k, 0)) for k in set(self.keys()).union(other.keys())})
            return vel

        def __truediv__(self, denumenator):
           if isinstance(denumenator, Number):
               return self.__mul__(1/denumenator)
           raise ValueError("{0} has not a suitable type for division".format(denumenator))

        def cutby(self, alpha):
            return MappingParticle.Velocity({k: v for k, v in self.items() if v >= alpha})

        def vector_length(self):
            return len(self)


        pass
    pass


class OrderingParticle(Particle):

    @staticmethod
    def _to_limit(val, min, max):
        if val > max:
            return max
        if val < min:
            return min
        return val

    def __init__(self, ordering):
        """
        :param ordering: has the following form
        {
            task_id: value
        }
        """
        super().__init__(ordering)
        pass

    def __sub__(self, other):
        if not isinstance(other, OrderingParticle):
            raise ValueError("Invalid type of the argument for this operation")
        velocity = OrderingParticle.Velocity({task_id: self.entity[task_id] - other.entity[task_id]
                                              for task_id in self.entity})
        return velocity

    def __add__(self, other):
        if not isinstance(other, OrderingParticle.Velocity):
            raise ValueError("Invalid type of the argument for this operation: {0}".format(type(other)))

        if len(other) == 0:
            return OrderingParticle({task_id: self.entity[task_id] for task_id in self.entity})

        velocity = OrderingParticle({task_id: self.entity[task_id] + other[task_id]
                                              for task_id in self.entity})
        return velocity

    def limit_by(self, min=-1, max=-1):
        for t in self.entity:
            self.entity[t] = OrderingParticle._to_limit(self.entity[t], min, max)
        pass

    def emptify(self):
        return OrderingParticle.Velocity({k: 0.0 for k in self.entity})

    class Velocity(dict):

        def __mul__(self, other):
            if isinstance(other, Number):
                if other < 0:
                    raise ValueError("Only positive numbers can be used for operations with velocity")
                return OrderingParticle.Velocity({k: 1.0 if v * other > 1.0 else v * other for k, v in self.items()})
            raise ValueError("{0} has not a suitable type for multiplication".format(other))

        def __add__(self, other):
            if isinstance(other, OrderingParticle.Velocity):
                if len(self) == 0:
                    return OrderingParticle.Velocity({task_id: other[task_id] for task_id in other.keys()})
                vel = OrderingParticle.Velocity({task_id: self[task_id] + other[task_id] for task_id in self.keys()})
                return vel
            raise ValueError("{0} has not a suitable type for adding".format(other))

        def __truediv__(self, denumenator):
            if isinstance(denumenator, Number):
                return self.__mul__(1/denumenator)
            raise ValueError("{0} has not a suitable type for division".format(denumenator))

        def limit_by(self, min=-1, max=1):
            for t in self:
                self[t] = OrderingParticle._to_limit(self[t], min, max)
            pass

        def vector_length(self):
            return math.sqrt(sum(val*val for t, val in self.items()))/len(self)
        pass
    pass


class CompoundParticle(Particle):
    def __init__(self, mapping_particle, ordering_particle):
        super().__init__(None)
        self.mapping = mapping_particle
        self.ordering = ordering_particle
        self._best = None
        pass

    def _get_best(self):
        return self._best

    def _set_best(self, value):
        self._best = value
        if value is not None:
            self.mapping.best = value.mapping
            self.ordering.best = value.ordering
        else:
            self.mapping.best = None
            self.ordering.best = None
        pass

    best = property(_get_best, _set_best)
    pass



