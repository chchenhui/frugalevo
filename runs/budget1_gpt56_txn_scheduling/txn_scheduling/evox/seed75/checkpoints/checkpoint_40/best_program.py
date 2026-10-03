import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Build true-cost append/insertion seeds and apply exact relocation/swap descent."""
    n = workload.num_txns
    if n <= 1:
        schedule = list(range(n))
        return workload.get_opt_seq_cost(schedule), schedule

    rng = random.Random(7919 * n + 17)
    candidate_count = min(n, 24)
    constructed = []

    # Append-greedy starts cheaply generate several distinct schedules while
    # evaluating the simulator's real prefix makespan at every choice.
    for restart in range(max(4, num_seqs)):
        remaining = list(range(n))
        schedule = [remaining.pop((restart if restart < n else rng.randrange(n)) % n)]

        while remaining:
            candidates = remaining if len(remaining) <= candidate_count else rng.sample(remaining, candidate_count)
            txn = min(candidates, key=lambda t: workload.get_opt_seq_cost(schedule + [t]))
            schedule.append(txn)
            remaining.remove(txn)

        constructed.append((workload.get_opt_seq_cost(schedule), schedule))

    # Insertion construction avoids the irreversible append-only decision:
    # a writer may need to precede a reader/writer chain already in the order.
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
                        best_cost, best_txn, best_position = trial_cost, txn, position

            schedule.insert(best_position, best_txn)
            remaining.remove(best_txn)

        constructed.append((workload.get_opt_seq_cost(schedule), schedule))

    constructed.sort(key=lambda item: item[0])
    best_cost, best_schedule = constructed[0][0], constructed[0][1][:]

    # Exhaustively test the complete relocation neighborhood.  If it has no
    # improvement, test swaps; each neighborhood can expose improvements in
    # the other, so alternate them from the strongest constructed schedules.
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

    # Strict local descent can become trapped when several transactions must
    # cross a conflict chain together.  Destroy a small part of the incumbent
    # and greedily recreate it using the real simulator cost at every possible
    # insertion point.  This is more directed than merely shuffling positions:
    # it permits each removed writer or reader to move across the whole order.
    perturbations = 5 if n <= 40 else 3
    for _ in range(perturbations):
        trial = best_schedule[:]
        remove_count = min(n - 1, 4 + rng.randrange(5))
        remove_positions = sorted(rng.sample(range(n), remove_count), reverse=True)
        removed = [trial.pop(position) for position in remove_positions]
        rng.shuffle(removed)

        # Recreate the destroyed portion one transaction at a time.  Evaluating
        # all positions is practical because only a few transactions are
        # removed, and preserves actual read/write-conflict effects.
        for txn in removed:
            insertion_cost, insertion_position = None, None
            for position in range(len(trial) + 1):
                candidate = trial[:]
                candidate.insert(position, txn)
                candidate_cost = workload.get_opt_seq_cost(candidate)
                if insertion_cost is None or candidate_cost < insertion_cost:
                    insertion_cost, insertion_position = candidate_cost, position
            trial.insert(insertion_position, txn)

        trial_cost = workload.get_opt_seq_cost(trial)

        # Use variable-neighborhood descent after reconstruction.  Relocation
        # is best for moving a transaction through a long conflict chain;
        # swaps can make a paired precedence change that no improving single
        # relocation exposes.  A successful swap is followed by relocation
        # again, allowing the two neighborhoods to unlock one another.
        for _ in range(4):
            move_cost, move_schedule = trial_cost, None
            for source in range(n):
                for destination in range(n):
                    if source == destination:
                        continue
                    candidate = trial[:]
                    txn = candidate.pop(source)
                    candidate.insert(destination, txn)
                    candidate_cost = workload.get_opt_seq_cost(candidate)
                    if candidate_cost < move_cost:
                        move_cost, move_schedule = candidate_cost, candidate

            if move_schedule is not None:
                trial, trial_cost = move_schedule, move_cost
                continue

            swap_cost, swap_schedule = trial_cost, None
            for left in range(n):
                for right in range(left + 1, n):
                    candidate = trial[:]
                    candidate[left], candidate[right] = candidate[right], candidate[left]
                    candidate_cost = workload.get_opt_seq_cost(candidate)
                    if candidate_cost < swap_cost:
                        swap_cost, swap_schedule = candidate_cost, candidate

            if swap_schedule is None:
                break
            trial, trial_cost = swap_schedule, swap_cost

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
