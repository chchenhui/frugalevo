import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Construct sampled true-cost greedy orders and use alternating relocation/swap descent."""
    n = workload.num_txns
    if n <= 1:
        schedule = list(range(n))
        return workload.get_opt_seq_cost(schedule), schedule

    rng = random.Random(7919 * n + 17)
    constructed = []
    candidate_count = min(n, 24)

    # Keep the original append-greedy seeds: they are cheap and provide a
    # dependable baseline for the subsequent exact local search.
    for restart in range(max(4, num_seqs)):
        remaining = list(range(n))
        schedule = [remaining.pop((restart if restart < n else rng.randrange(n)) % n)]

        while remaining:
            candidates = remaining if len(remaining) <= candidate_count else rng.sample(remaining, candidate_count)
            txn = min(candidates, key=lambda t: workload.get_opt_seq_cost(schedule + [t]))
            schedule.append(txn)
            remaining.remove(txn)

        constructed.append((workload.get_opt_seq_cost(schedule), schedule))

    # Appending a transaction irrevocably fixes it behind every transaction
    # already chosen.  Build a few additional seeds by evaluating the actual
    # prefix makespan at possible insertion positions.  This can place a
    # writer before the readers/writers that form its critical conflict chain,
    # an arrangement append-only construction cannot discover.
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
                interior = rng.sample(range(1, len(schedule)), position_limit - 2)
                positions = [0] + interior + [len(schedule)]

            best_cost = None
            best_txn = None
            best_position = None
            for txn in candidates:
                for position in positions:
                    trial = schedule[:]
                    trial.insert(position, txn)
                    trial_cost = workload.get_opt_seq_cost(trial)
                    if best_cost is None or trial_cost < best_cost:
                        best_cost = trial_cost
                        best_txn = txn
                        best_position = position

            schedule.insert(best_position, best_txn)
            remaining.remove(best_txn)

        constructed.append((workload.get_opt_seq_cost(schedule), schedule))

    constructed.sort(key=lambda item: item[0])
    best_cost, best_schedule = constructed[0][0], constructed[0][1][:]

    # A swap can expose a new beneficial relocation, while a relocation can
    # expose a new useful swap.  Alternate the two exact neighborhoods instead
    # of stopping after one swap, retaining strict improvements only.
    for initial_cost, initial_schedule in constructed[:min(3, len(constructed))]:
        schedule, cost = initial_schedule[:], initial_cost

        for _ in range(4):
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

            if move_schedule is not None:
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
