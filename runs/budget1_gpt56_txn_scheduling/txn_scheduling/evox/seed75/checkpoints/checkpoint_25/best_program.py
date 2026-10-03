import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Build true-cost greedy seeds, descend relocation/swap neighborhoods, then perturb and re-optimize."""
    n = workload.num_txns
    if n <= 1:
        schedule = list(range(n))
        return workload.get_opt_seq_cost(schedule), schedule

    rng = random.Random(7919 * n + 17)
    candidate_count = min(n, 24)
    constructed = []

    for restart in range(max(4, num_seqs)):
        remaining = list(range(n))
        schedule = [remaining.pop((restart if restart < n else rng.randrange(n)) % n)]
        while remaining:
            candidates = remaining if len(remaining) <= candidate_count else rng.sample(remaining, candidate_count)
            txn = min(candidates, key=lambda t: workload.get_opt_seq_cost(schedule + [t]))
            schedule.append(txn)
            remaining.remove(txn)
        constructed.append((workload.get_opt_seq_cost(schedule), schedule))

    insertion_seeds = 2 if n <= 30 else 1
    position_limit = n + 1 if n <= 30 else 10
    for restart in range(insertion_seeds):
        remaining = list(range(n))
        schedule = [remaining.pop((restart * 7) % n)]
        while remaining:
            candidates = remaining if len(remaining) <= candidate_count else rng.sample(remaining, candidate_count)
            if len(schedule) + 1 <= position_limit:
                positions = range(len(schedule) + 1)
            else:
                positions = [0] + rng.sample(range(1, len(schedule)), position_limit - 2) + [len(schedule)]

            best_cost, best_txn, best_position = None, None, None
            for txn in candidates:
                for position in positions:
                    trial = schedule[:]
                    trial.insert(position, txn)
                    trial_cost = workload.get_opt_seq_cost(trial)
                    if best_cost is None or trial_cost < best_cost:
                        best_cost, best_txn, best_position = trial_cost, txn, position
            schedule.insert(best_position, best_txn)
            remaining.remove(best_txn)
        constructed.append((workload.get_opt_seq_cost(schedule), schedule))

    def relocate_descent(schedule, cost, rounds):
        """Perform strict best-improvement relocation descent using exact simulated makespan."""
        for _ in range(rounds):
            move_cost, move_schedule = cost, None
            for source in range(n):
                for destination in range(n):
                    if source == destination:
                        continue
                    trial = schedule[:]
                    txn = trial.pop(source)
                    trial.insert(destination, txn)
                    trial_cost = workload.get_opt_seq_cost(trial)
                    if trial_cost < move_cost:
                        move_cost, move_schedule = trial_cost, trial
            if move_schedule is None:
                break
            schedule, cost = move_schedule, move_cost
        return cost, schedule

    constructed.sort(key=lambda item: item[0])
    best_cost, best_schedule = constructed[0][0], constructed[0][1][:]

    for initial_cost, initial_schedule in constructed[:min(3, len(constructed))]:
        schedule, cost = initial_schedule[:], initial_cost
        for _ in range(6):
            move_cost, move_schedule = relocate_descent(schedule, cost, 1)
            if move_schedule != schedule:
                schedule, cost = move_schedule, move_cost
                continue

            swap_cost, swap_schedule = cost, None
            for left in range(n):
                for right in range(left + 1, n):
                    trial = schedule[:]
                    trial[left], trial[right] = trial[right], trial[left]
                    trial_cost = workload.get_opt_seq_cost(trial)
                    if trial_cost < swap_cost:
                        swap_cost, swap_schedule = trial_cost, trial

            if swap_schedule is None:
                break
            schedule, cost = swap_schedule, swap_cost

        if cost < best_cost:
            best_cost, best_schedule = cost, schedule

    # Strict descent cannot cross equal-cost plateaus.  Small multi-position
    # shuffles change several conflict orientations at once; each shuffled
    # schedule is repaired by exact relocation search and accepted only if it
    # improves the incumbent.
    perturbations = 5 if n <= 40 else 3
    for _ in range(perturbations):
        trial = best_schedule[:]
        positions = rng.sample(range(n), min(n, 3 + rng.randrange(4)))
        values = [trial[position] for position in positions]
        rng.shuffle(values)
        for position, txn in zip(positions, values):
            trial[position] = txn

        trial_cost = workload.get_opt_seq_cost(trial)
        trial_cost, trial = relocate_descent(trial, trial_cost, 3)

        if trial_cost < best_cost:
            best_cost, best_schedule = trial_cost, trial

    return best_cost, best_schedule

# EVOLVE-BLOCK-END

def get_random_costs():
    workload_size = 100
    workload = Workload(WORKLOAD_1)

    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)
    print(cost1, cost2, cost3)
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3]


if __name__ == "__main__":
    makespan, schedule = get_random_costs()
    print(f"Makespan: {makespan}")
