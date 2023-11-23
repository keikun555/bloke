"""Monte Carlo Markov Chain Sampler"""

import math
import random
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeAlias, TypeVar

Point = TypeVar("Point")

Probability: TypeAlias = float


class MonteCarloMarkovChainSample(ABC, Generic[Point]):
    """A generic class for sampling using MCMC"""

    def __init__(self, beta: float):
        self.__beta = beta

    @abstractmethod
    def _next_candidate(self, point: Point) -> tuple[Point, Probability, Probability]:
        """Finds the next candidate of a point with the forward and
        backward probabilities of mutating to the point"""

    @abstractmethod
    def cost(self, point: Point) -> Probability:
        """Calculates the cost of the point within [0, 1]"""

    def sample(
        self, initial_point: Point, initial_cost: Optional[Probability] = None
    ) -> Point:
        """Generates the next points using MCMC"""
        if initial_cost is None:
            initial_cost = self.cost(initial_point)

        candidate_point, fwd, bwd = self._next_candidate(initial_point)
        candidate_cost = self.cost(candidate_point)

        acceptability: Probability
        if initial_cost * bwd == 0:
            acceptability = Probability(0)
        else:
            acceptability = min(
                1.0,
                math.exp(-self.__beta * (candidate_cost * fwd) / (initial_cost * bwd)),
            )

        if random.random() < acceptability:
            return candidate_point

        return initial_point
