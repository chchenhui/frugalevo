import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Sampled exact insertion construction plus exact local window ordering."""
    import itertools
    import time

    n = workload.num_txns
    if n == 0:
        return 0, []

    # The caller evaluates three workloads, so each invocation must leave room
    # for the other two.  A valid identity schedule is available immediately.
    deadline = time.monotonic() + 105.0
    cache = {}

    def cost(seq):
        key = tuple(seq)
        value = cache.get(key)
        if value is None:
            value = workload.get_opt_seq_cost(list(seq))
            cache[key] = value
        return value

    def construct(start):
        """Build a schedule by sampled regret-guided exact insertion."""
        order = [start]
        remaining = set(range(n))
        remaining.remove(start)

        while remaining and time.monotonic() < deadline:
            options = list(remaining)
            random.shuffle(options)
            options = options[:min(8, len(options))]

            chosen_txn = None
            chosen_order = None
            chosen_best = None
            chosen_regret = None

            # Prefer transactions whose good insertion opportunities are
            # scarce, rather than merely taking the cheapest current move.
            for txn in options:
                insertion_values = []
                insertion_orders = []
                for position in range(len(order) + 1):
                    trial = order[:]
                    trial.insert(position, txn)
                    insertion_orders.append(trial)
                    insertion_values.append(cost(trial))

                ranking = sorted(range(len(insertion_values)),
                                 key=lambda index: insertion_values[index])
                best_index = ranking[0]
                best_value = insertion_values[best_index]
                if len(ranking) > 1:
                    regret = (insertion_values[ranking[1]] - best_value)
                else:
                    regret = 0

                if (chosen_txn is None or regret > chosen_regret or
                        (regret == chosen_regret and
                         best_value < chosen_best)):
                    chosen_txn = txn
                    chosen_order = insertion_orders[best_index]
                    chosen_best = best_value
                    chosen_regret = regret

            if chosen_order is None:
                break
            order = chosen_order
            remaining.remove(chosen_txn)

        # This is also the deadline-safe completion path.
        order.extend(sorted(remaining))
        return order

    def improve(seq):
        """Exhaustively optimize sixteen adaptive contiguous windows, then relocate."""
        current = cost(seq)

        if n >= 2:
            width = min(5, n)
            all_starts = list(range(n - width + 1))
            random.shuffle(all_starts)

            # Keep the bounded exact-search budget, but make later windows
            # adaptive: after an accepted reorder, inspect nearby untouched
            # regions before spending evaluations elsewhere.
            pending = all_starts[:]
            examined = set()
            accepted_windows = 0

            while pending and len(examined) < 16:
                if time.monotonic() >= deadline:
                    break

                left = pending.pop(0)
                if left in examined:
                    continue
                examined.add(left)

                original = tuple(seq[left:left + width])
                best_value = current
                best_window = None

                for perm in itertools.permutations(original):
                    if time.monotonic() >= deadline:
                        break
                    if perm == original:
                        continue

                    trial = seq[:left] + list(perm) + seq[left + width:]
                    value = cost(trial)
                    if value < best_value:
                        best_value = value
                        best_window = perm

                if best_window is not None:
                    seq = (seq[:left] + list(best_window) +
                           seq[left + width:])
                    current = best_value
                    accepted_windows += 1

                    # Reordering a window can change the best ordering of
                    # overlapping windows.  Prioritize those windows while
                    # retaining the hard limit of sixteen distinct windows.
                    nearby = [
                        candidate for candidate in (left - 2, left - 1,
                                                    left + 1, left + 2)
                        if 0 <= candidate <= n - width
                        and candidate not in examined
                    ]
                    random.shuffle(nearby)
                    pending = nearby + pending

        accepted = 0
        while accepted < 12 and time.monotonic() < deadline:
            moves = [(i, j) for i in range(n) for j in range(n)
                     if i != j and j != i + 1]
            random.shuffle(moves)
            changed = False

            for i, j in moves:
                if time.monotonic() >= deadline:
                    break
                trial = seq[:]
                txn = trial.pop(i)
                trial.insert(j - (j > i), txn)
                value = cost(trial)
                if value < current:
                    seq, current = trial, value
                    accepted += 1
                    changed = True
                    break

            if not changed:
                break

        return seq, current

    best_seq = list(range(n))
    best_cost = cost(best_seq)

    starts = max(1, min(4, int(num_seqs), n))
    for start in random.sample(range(n), starts):
        if time.monotonic() >= deadline:
            break
        candidate = construct(start)
        candidate, value = improve(candidate)
        if value < best_cost:
            best_seq, best_cost = candidate, value

    return best_cost, best_seq

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