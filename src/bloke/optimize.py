"""Optimize using the Bloke sampler"""
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from typing import Any

import click
import numpy as np
import numpy.typing as npt

from bloke.sample import BlokeSample, State
from bril.brili import Brilirs
from bril.briltxt import prints_prog
from bril.typing_bril import Program

logger = logging.getLogger(__name__)

MIN_PHASES = 2
MAX_PHASES = 10


def handle_exception(args):
    """Thread exception handling"""
    if issubclass(args.exc_type, KeyboardInterrupt):
        sys.__excepthook__(*args.exc_type)
        return

    logger.error(
        "Uncaught exception",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


threading.excepthook = handle_exception


class Bloke(object):
    """Namespace for all the Bloke functions"""

    @staticmethod
    def phase_optimize_thread(
        sampler: BlokeSample,
        correct_state_queue: queue.Queue[State | None],
        program: Program,
        max_iterations: int = 100000,
    ) -> None:
        """Samples from bloke and pass in correct and better programs into queue"""
        logger.debug("Starting optimize thread")

        state = State(program, True, 1.0)
        state.cost = sampler.cost(state)
        correct_state_queue.put(state, block=True)

        best_state = state

        i = 0
        while i < max_iterations:
            last_state = state
            state = sampler.sample(last_state, state.cost)
            if state.correct:
                if state.cost < last_state.cost:
                    correct_state_queue.put(state, block=True)
                if state.cost < best_state.cost:
                    best_state = state
            i += 1

        correct_state_queue.put(best_state, block=True)
        correct_state_queue.put(None, block=True)
        logger.debug("Finished optimize thread")

    @staticmethod
    def phase_optimize(
        out_queue: queue.Queue[State | None],
        program: Program,
        beta: float,
        ratio: float,
        threads_per_program: int = 1,  # TODO: 2 or more doesn't work because of Z3 race
    ) -> int:
        """Runs Bloke on program, sends unique programs into out_queue"""
        logger.debug("Starting phase optimize")
        program_set: set[str] = set()

        correct_state_queue: queue.Queue[State | None] = queue.Queue(
            maxsize=threads_per_program
        )

        sampler = BlokeSample(Brilirs(), program, 1, 1, 1, 1, 0.1, beta)
        sampler.performance_correctness_ratio = ratio

        threads: list[threading.Thread] = []
        for _ in range(threads_per_program):
            thread = threading.Thread(
                target=Bloke.phase_optimize_thread,
                args=(sampler, correct_state_queue, program),
            )
            thread.start()
            threads.append(thread)

        count = 0
        done_count = 0
        while done_count < threads_per_program:
            state = correct_state_queue.get(block=True)
            if state is None:
                done_count += 1
                continue

            if (program_string := json.dumps(state.program)) not in program_set:
                program_set.add(program_string)
                out_queue.put(state, block=True)
                count += 1

        for thread in threads:
            thread.join()

        logger.debug("Finished phase optimize")
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
                    "Phase %.2f (beta=%.2f) received:\n%s",
                    ratio,
                    beta,
                    prints_prog(state.program),
                )
                process = pool.apply_async(
                    Bloke.phase_optimize, (out_queue, state.program, beta, ratio)
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

        beta_min, beta_max = beta_range
        beta_step = (beta_max - beta_min) / (num_phases - 1)
        betas = [beta_min + beta_step * i for i in range(num_phases)]
        betas[-1] = beta_max

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
    "--beta-min",
    type=float,
    default=1.0,
    help="Minimum beta value for MCMC",
)
@click.option(
    "--beta-max",
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
    beta_min: float, beta_max: float, num_phases: int, verbose: bool, debug: bool
) -> None:
    logging_config: dict[str, Any] = {
        "stream": sys.stderr,
        "encoding": "utf-8",
        "format": "%(asctime)s %(levelname)s: %(message)s",
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

    start_time = time.time()
    optimized_program = Bloke.optimize(program, (beta_min, beta_max), num_phases)
    logger.info("Completed in %.2f seconds", time.time() - start_time)
    print(json.dumps(optimized_program))


if __name__ == "__main__":
    main()
