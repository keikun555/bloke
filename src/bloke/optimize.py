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
    def sample_thread(
        sampler: BlokeSample,
        state_queue: queue.Queue[State | None],
        program: Program,
        max_iterations: int,
    ) -> None:
        """Samples from bloke and pass in correct and better programs into queue"""
        logger.debug("Starting sample thread")

        state = State(program, True, 1.0)
        state.cost = sampler.cost(state)
        state_queue.put(state, block=True)

        best_correct_state = state

        i = 0
        while i < max_iterations:
            last_state = state
            state = sampler.sample(last_state, state.cost)
            if state.correct:
                if state.cost < last_state.cost:
                    state_queue.put(state, block=True)
                if state.cost <= best_correct_state.cost:
                    # Allow equality for a radom walk
                    best_correct_state = state
            i += 1

        state_queue.put(best_correct_state, block=True)
        state_queue.put(None, block=True)

        logger.debug("Finished sample thread")

    @staticmethod
    def sample(
        out_queue: queue.Queue[State | None],
        program: Program,
        beta: float,
        ratio: float,
        max_iterations: int,
    ) -> int:
        """Runs Bloke on program, sends unique programs into out_queue"""
        logger.debug("Starting sample")

        program_set: set[str] = set()

        state_queue: queue.Queue[State | None] = queue.Queue(maxsize=1)

        sampler = BlokeSample(Brilirs(), program, 1, 1, 1, 1, 0.1, beta)
        sampler.performance_correctness_ratio = ratio

        start_time = time.time()
        thread = threading.Thread(
            target=Bloke.sample_thread,
            args=(sampler, state_queue, program, max_iterations),
        )
        thread.start()

        count = 0
        while (state := state_queue.get(block=True)) is not None:
            if (program_string := json.dumps(state.program)) not in program_set:
                program_set.add(program_string)
                out_queue.put(state, block=True)
                count += 1

        thread.join()

        logger.info(
            "Phase %.2f (beta=%.2f), finished sample process in %.2f seconds\n"
            "\t%.2f samples per second\n"
            "\t%d test cases\n"
            "\t%d programs found",
            ratio,
            beta,
            time.time() - start_time,
            max_iterations / (time.time() - start_time),
            len(sampler.test_cases),
            count,
        )
        return count

    @staticmethod
    def phase_thread(
        pool: "mp.pool.Pool",
        beta: float,
        ratio: float,
        in_queue: queue.Queue[State | None],
        out_queue: queue.Queue[State | None],
        sample_processes_per_program: int,
        samples_per_program: int,
    ):
        """Accepts from in_queue and starts sample processes"""
        logger.debug("Starting phase thread")
        processes: list["mp.pool.ApplyResult[int]"] = []
        program_set: set[str] = set()
        receive_count = 0

        while (state := in_queue.get(block=True)) is not None:
            if (program_string := json.dumps(state.program)) not in program_set:
                program_set.add(program_string)
                receive_count += 1
                logger.info(
                    "Phase %.2f (beta=%.2f) received:\n%s",
                    ratio,
                    beta,
                    prints_prog(state.program),
                )

                # Start sample processes
                for _ in range(sample_processes_per_program):
                    process = pool.apply_async(
                        Bloke.sample,
                        (out_queue, state.program, beta, ratio, samples_per_program),
                    )
                    processes.append(process)

        # We're done
        send_count = 0
        for process in processes:
            process.wait()
            send_count += process.get()

        out_queue.put(None, block=True)

        logger.info(
            "Phase %.2f (beta=%.2f), received %d programs and sent %d programs",
            ratio,
            beta,
            receive_count,
            send_count,
        )

    @staticmethod
    def optimize(
        program: Program,
        beta_range: tuple[float, float],
        num_phases: int,
        samples: int,
        processes: int,
    ) -> Program:
        """Optimize program, beta is for MCMC, num_phases for performance factor smoothing"""
        assert MIN_PHASES <= num_phases <= MAX_PHASES

        beta_min, beta_max = beta_range
        beta_step = (beta_max - beta_min) / (num_phases - 1)
        betas = [beta_min + beta_step * i for i in range(num_phases)]
        betas[-1] = beta_max

        if (cpu_count := os.cpu_count()) is None:
            cpu_count = 1
        # Make num_phases ratios from 0 to 1
        ratios = [i / (num_phases - 1) for i in range(num_phases)]

        # Make input and output queues
        manager = mp.Manager()
        queues: list[queue.Queue[State | None]] = [
            manager.Queue() for _ in range(num_phases + 1)
        ]
        in_queues: list[queue.Queue[State | None]] = queues[:-1]
        out_queues: list[queue.Queue[State | None]] = queues[1:]

        # Number of sample processes to run per program received on phase thread
        processes_per_program = [16] + [1 for _ in range(num_phases - 1)]

        # Phase worker threads
        threads: list[threading.Thread] = []

        with mp.Pool(processes) as pool:
            # Start phase threads
            for beta, ratio, in_queue, out_queue, processes in zip(
                betas, ratios, in_queues, out_queues, processes_per_program
            ):
                thread = threading.Thread(
                    target=Bloke.phase_thread,
                    args=(pool, beta, ratio, in_queue, out_queue, processes, samples),
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
    show_default=True,
    help="Minimum beta value for MCMC.",
)
@click.option(
    "--beta-max",
    type=float,
    default=10.0,
    show_default=True,
    help="Maximum beta value for MCMC.",
)
@click.option(
    "--num-phases",
    type=int,
    default=2,
    show_default=True,
    callback=validate_num_phases,
    help="Number of phases, determines gamma. "
    f"Must be between {MIN_PHASES} and {MAX_PHASES}, inclusive.",
)
@click.option(
    "--samples",
    type=int,
    default=10000,
    show_default=True,
    help="Number of samples per program.",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=os.cpu_count(),
    show_default=True,
    help="Number jobs to run simultaneously.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="Verbose output",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="Debug output",
)
def main(
    beta_min: float,
    beta_max: float,
    num_phases: int,
    samples: int,
    jobs: int,
    verbose: bool,
    debug: bool,
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
    optimized_program = Bloke.optimize(
        program, (beta_min, beta_max), num_phases, samples, jobs
    )
    logger.info("Completed in %.2f seconds", time.time() - start_time)
    print(json.dumps(optimized_program))


if __name__ == "__main__":
    main()
