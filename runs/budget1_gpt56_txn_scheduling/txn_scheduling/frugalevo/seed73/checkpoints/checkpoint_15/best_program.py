import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Build multistart cheapest-insertion permutations, then refine by bounded VND."""
    import time

    n = workload.num_txns
    if n == 0:
        return 0, []

    # This is per workload; the unchanged caller evaluates exactly three.
    # Three workloads are evaluated by the caller, so retain a conservative
    # per-workload cap while allowing the insertion search more evaluations.
    # Reserve a small margin for caller overhead while using nearly the full
    # three-workload budget for true-cost insertion and refinement searches.
    deadline = time.monotonic() + 119.0
    cache = {}

    def cost(seq):
        key = tuple(seq)
        value = cache.get(key)
        if value is None:
            value = workload.get_opt_seq_cost(list(seq))
            cache[key] = value
        return value

    # A complete valid answer is available before any search work.
    best_seq = list(range(n))
    best_cost = cost(best_seq)

    """Construct multistart orders by globally cheapest insertion.

    Each bounded start repeatedly chooses the unplaced transaction and every
    insertion position with the lowest true simulator cost.  Cyclic candidate
    windows provide distinct deterministic starts while limiting simulator
    calls, and unfinished constructions are completed as valid permutations.
    """
    starts = min(10, max(3, num_seqs))
    seed_order = [0] if n == 1 else [0, n - 1]
    stride = max(1, n // max(1, starts - 2))
    seed_order.extend(
        (2 + i * stride) % n for i in range(max(0, starts - 2))
    )

    seen = set()
    for seed in seed_order:
        if seed in seen or time.monotonic() >= deadline:
            continue
        seen.add(seed)
        sequence = [seed]
        remaining = [txn for txn in range(n) if txn != seed]

        while remaining and time.monotonic() < deadline:
            width = min(12, len(remaining))
            # Rotate the bounded candidate window by both restart and depth.
            # This preserves the compute bound while exposing different
            # transactions to the globally cheapest insertion decision.
            offset = (seed * 11 + len(sequence) * 7) % len(remaining)
            candidates = [
                remaining[(offset + j) % len(remaining)]
                for j in range(width)
            ]

            selected_txn = None
            selected_position = 0
            selected_cost = None
            for txn in candidates:
                if time.monotonic() >= deadline:
                    break
                for position in range(len(sequence) + 1):
                    trial = (
                        sequence[:position]
                        + [txn]
                        + sequence[position:]
                    )
                    value = cost(trial)
                    if (
                        selected_cost is None
                        or value < selected_cost
                        or (
                            value == selected_cost
                            and (
                                selected_txn is None
                                or txn < selected_txn
                                or (
                                    txn == selected_txn
                                    and position < selected_position
                                )
                            )
                        )
                    ):
                        selected_cost = value
                        selected_txn = txn
                        selected_position = position

            if selected_txn is None:
                break
            sequence.insert(selected_position, selected_txn)
            remaining.remove(selected_txn)

        sequence.extend(remaining)
        value = cost(sequence)
        if value < best_cost:
            best_cost, best_seq = value, sequence

    # Bounded variable-neighborhood descent.  A relocation can cross an
    # adjacent-swap local minimum, while all candidates remain permutations.
    # The longer insertion phase is balanced by limiting refinement rounds.
    # Spend the additional bounded search time on another VND pass.  Each
    # round still stops immediately at the per-workload deadline, and every
    # accepted move remains strictly better under the true simulator cost.
    max_rounds = 4
    for _ in range(max_rounds):
        if time.monotonic() >= deadline:
            break
        improved = False

        # First improving bounded relocation, checking both directions.
        for i in range(n):
            if time.monotonic() >= deadline or improved:
                break
            item = best_seq[i]
            rest = best_seq[:i] + best_seq[i + 1:]
            for distance in range(1, min(12, n - 1) + 1):
                for target in (i - distance, i + distance):
                    if target < 0 or target >= n:
                        continue
                    candidate = rest[:target] + [item] + rest[target:]
                    value = cost(candidate)
                    if value < best_cost:
                        best_cost, best_seq = value, candidate
                        improved = True
                        break
                if improved or time.monotonic() >= deadline:
                    break

        if improved:
            continue

        # Non-adjacent swaps expose pairwise conflict exchanges without an
        # exhaustive O(n^2) scan.
        for distance in range(2, min(6, n - 1) + 1):
            if improved or time.monotonic() >= deadline:
                break
            for i in range(n - distance):
                candidate = best_seq[:]
                candidate[i], candidate[i + distance] = (
                    candidate[i + distance],
                    candidate[i],
                )
                value = cost(candidate)
                if value < best_cost:
                    best_cost, best_seq = value, candidate
                    improved = True
                    break

        if improved:
            continue

        # Short reversals provide a distinct cluster-level local move.
        for length in range(3, min(8, n) + 1):
            if improved or time.monotonic() >= deadline:
                break
            for start in range(n - length + 1):
                candidate = (
                    best_seq[:start]
                    + list(reversed(best_seq[start:start + length]))
                    + best_seq[start + length:]
                )
                value = cost(candidate)
                if value < best_cost:
                    best_cost, best_seq = value, candidate
                    improved = True
                    break

        if not improved:
            break

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