"""Optimize using the Bloke sampler"""
import copy
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import random
import sys
import threading
import time
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Generator, Iterable, TypeAlias, cast

import click
import numpy as np
import numpy.typing as npt

from bloke.bril_equivalence import (
    EquivalenceAnalysisResult,
    z3_prove_equivalence_or_find_counterexample,
)
from bloke.mcmc import MonteCarloMarkovChainSample, Probability
from bloke.sample import BlokeSample, State
from bril.bril2z3 import COMPATIBLE_OPS
from bril.bril_constants import OPERATORS, GenericType
from bril.brili import Brili, Brilirs, SubprocessBrili
from bril.briltxt import prints_prog
from bril.typing_bril import (
    BrilType,
    Effect,
    Instruction,
    Operation,
    PrimitiveType,
    Program,
    Value,
    Variable,
)

logger = logging.getLogger(__name__)

MIN_PHASES = 2
MAX_PHASES = 5


class Bloke(object):
    """Namespace for all the Bloke functions"""

    @staticmethod
    def process_thread(
        correct_state_queue: queue.Queue[State | None],
        program: Program,
        beta: float,
        performance_correctness_ratio: float,
        max_iterations: int = 10000,
    ) -> None:
        """Samples from bloke and pass in correct and better programs into queue"""
        logger.debug("Starting process thread")
        local_program_set: set[str] = set()

        sampler = BlokeSample(Brilirs(), program, 1, 1, 1, 1, 0.1, beta)
        sampler.performance_correctness_ratio = performance_correctness_ratio

        state = State(program, True, 1.0)
        state.cost = sampler.cost(state)
        best_state = state

        local_program_set.add(json.dumps(state.program))
        correct_state_queue.put(state, block=True)

        i = 0
        while i < max_iterations:
            if (
                state.correct
                and (program_string := json.dumps(state.program))
                not in local_program_set
            ):
                local_program_set.add(program_string)
                if state.cost < best_state.cost:
                    best_state = state
                    correct_state_queue.put(state, block=True)

            state = sampler.sample(state, state.cost)
            i += 1

        correct_state_queue.put(best_state, block=True)
        correct_state_queue.put(None, block=True)
        logger.debug("Finished process thread")

    @staticmethod
    def process(
        out_queue: queue.Queue[State | None],
        program: Program,
        beta: float,
        ratio: float,
    ) -> int:
        """Runs Bloke on program, sends unique programs into out_queue"""
        logger.debug("Starting process")

        correct_state_queue: queue.Queue[State | None] = queue.Queue(maxsize=1)

        process_thread = threading.Thread(
            target=Bloke.process_thread,
            args=(correct_state_queue, program, beta, ratio),
        )
        process_thread.start()

        count = 0
        while (state := correct_state_queue.get(block=True)) is not None:
            out_queue.put(state, block=True)
            count += 1

        process_thread.join()

        logger.debug("Finished process")
        return count

    @staticmethod
    def phase_thread(
        pool: "mp.pool.Pool",
        beta: float,
        ratio: float,
        in_queue: queue.Queue[State | None],
        out_queue: queue.Queue[State | None],
    ):
        """Accepts from in_queue and starts process threads"""
        logger.debug("Starting phase thread")
        processes: list["mp.pool.ApplyResult[int]"] = []
        program_set: set[str] = set()

        while (state := in_queue.get(block=True)) is not None:
            if (program_string := json.dumps(state.program)) not in program_set:
                program_set.add(program_string)
                logger.info(
                    "PHASE %.2f (beta=%.2f) received:\n%s",
                    ratio,
                    beta,
                    prints_prog(state.program),
                )
                process = pool.apply_async(
                    Bloke.process, (out_queue, state.program, beta, ratio)
                )
                processes.append(process)

        # We're done
        total_count = 0
        for process in processes:
            process.wait()
            total_count += process.get()

        out_queue.put(None, block=True)

        logger.info(
            "PHASE %.2f (beta=%.2f) sent %d programs",
            ratio,
            beta,
            total_count,
        )
        logger.debug("Finished phase thread")

    @staticmethod
    def optimize(
        program: Program, beta_range: tuple[float, float], num_phases: int
    ) -> Program:
        """Optimize program, beta is for MCMC, num_phases for performance factor smoothing"""
        assert MIN_PHASES <= num_phases <= MAX_PHASES

        min_beta, max_beta = beta_range
        beta_step = (max_beta - min_beta) / (num_phases - 1)
        betas = [min_beta + beta_step * i for i in range(num_phases)]
        betas[-1] = max_beta

        if (cpu_count := os.cpu_count()) is None:
            cpu_count = 1
        # Make num_phases phases from 0 to 1
        ratios = [i / (num_phases - 1) for i in range(num_phases)]

        # Make input and output queues
        manager = mp.Manager()
        queues: list[queue.Queue[State | None]] = [
            manager.Queue(maxsize=cpu_count) for _ in range(num_phases + 1)
        ]
        in_queues: list[queue.Queue[State | None]] = queues[:-1]
        out_queues: list[queue.Queue[State | None]] = queues[1:]

        # Phase worker threads
        threads: list[threading.Thread] = []

        with mp.Pool() as pool:
            # Start phase threads
            for beta, ratio, in_queue, out_queue in zip(
                betas, ratios, in_queues, out_queues
            ):
                thread = threading.Thread(
                    target=Bloke.phase_thread,
                    args=(pool, beta, ratio, in_queue, out_queue),
                )
                thread.start()
                threads.append(thread)

            # Put initial program into first queue
            initial_state = State(program, True, 1.0)
            in_queues[0].put(initial_state, block=True)
            in_queues[0].put(None)

            # Get optimal program from last phase thread
            state: State | None = out_queues[-1].get(block=True)
            best_correct_state = state
            while state is not None:
                if best_correct_state is None or state.cost < best_correct_state.cost:
                    best_correct_state = state
                    logger.info(
                        "Found better program with cost %.2f \n%s",
                        state.cost,
                        prints_prog(state.program),
                    )
                state = out_queues[-1].get(block=True)

            for thread in threads:
                thread.join()

        if best_correct_state is None:
            return program
        return best_correct_state.program


def validate_num_phases(ctx, param, value):
    if MIN_PHASES <= value <= MAX_PHASES:
        return value
    raise click.BadParameter(
        f"num-phases must be between {MIN_PHASES} and {MAX_PHASES}"
    )


@click.command()
@click.option(
    "--min-beta",
    type=float,
    default=1.0,
    help="Minimum beta value for MCMC",
)
@click.option(
    "--max-beta",
    type=float,
    default=10.0,
    help="Maximum beta value for MCMC",
)
@click.option(
    "--num-phases",
    type=int,
    default=2,
    callback=validate_num_phases,
    help="Number of phases, determines gamma",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    type=bool,
    help="Verbose output",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    type=bool,
    help="Debug output",
)
def main(
    min_beta: float, max_beta: float, num_phases: int, verbose: bool, debug: bool
) -> None:
    logging_config: dict[str, Any] = {
        "stream": sys.stderr,
        "encoding": "utf-8",
        "format": "%(message)s",
    }

    if verbose:
        logging_config["level"] = logging.INFO
    if debug:
        logging_config["level"] = logging.DEBUG

    logging.basicConfig(**logging_config)
    gamma = 1.0 / num_phases
    logger.debug("GAMMA: %f", gamma)

    program: Program = json.load(sys.stdin)
    sys.stdin = open(
        "/dev/tty", encoding="utf-8"
    )  # pylint: disable=consider-using-with

    optimized_program = Bloke.optimize(program, (min_beta, max_beta), num_phases)
    print(json.dumps(optimized_program))


if __name__ == "__main__":
    main()
