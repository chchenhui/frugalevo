import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Seed with simulator-cost beam search, then use bounded exact DFS with
    cached prefix costs and safe canonicalization of nonconflicting transactions."""
    import re
    import time
    from functools import lru_cache

    n = workload.num_txns
    if n <= 1:
        seq = list(range(n))
        return workload.get_opt_seq_cost(seq), seq

    @lru_cache(maxsize=None)
    def cost(prefix):
        # Prefix order is significant: caching by a selected set would be wrong.
        return workload.get_opt_seq_cost(list(prefix))

    # First obtain a good complete upper bound cheaply.  The exact search below
    # can consequently discard any prefix whose already-fixed makespan cannot
    # improve this incumbent.
    width = max(1, min(8, num_seqs))
    beam = [((), tuple(range(n)))]
    for _ in range(n):
        expanded = []
        for prefix, remaining in beam:
            for txn in remaining:
                candidate = prefix + (txn,)
                expanded.append((
                    cost(candidate), candidate,
                    tuple(other for other in remaining if other != txn),
                ))
        expanded.sort(key=lambda x: x[0])
        beam = [(prefix, remaining) for _, prefix, remaining in expanded[:width]]

    best_seq = list(min(beam, key=lambda x: cost(x[0]))[0])
    best_cost = cost(tuple(best_seq))

    # Exact insertion descent makes the branch-and-bound upper bound tighter.
    for _ in range(3):
        next_seq, next_cost = best_seq, best_cost
        for source, txn in enumerate(best_seq):
            partial = best_seq[:source] + best_seq[source + 1:]
            for destination in range(n):
                candidate = tuple(partial[:destination] + [txn] + partial[destination:])
                value = cost(candidate)
                if value < next_cost:
                    next_seq, next_cost = list(candidate), value
        if next_cost >= best_cost:
            break
        best_seq, best_cost = next_seq, next_cost

    # Recover documented transaction text where possible.  Without it, DFS is
    # still valid, but no commutation equivalences can safely be assumed.
    values = None
    for value in list(getattr(workload, "__dict__", {}).values()):
        if isinstance(value, (list, tuple)) and len(value) == n:
            values = list(value)
            break
        if isinstance(value, dict) and len(value) == n:
            ordered = [None] * n
            for key, txn in value.items():
                match = re.search(r"(\d+)$", str(key))
                if match and int(match.group(1)) < n:
                    ordered[int(match.group(1))] = txn
            if all(txn is not None for txn in ordered):
                values = ordered
                break

    access = [set() for _ in range(n)]
    if values is not None:
        for i, txn in enumerate(values):
            for mode, key in re.findall(r"([rRwW])\s*-\s*(\d+)", str(txn)):
                access[i].add((int(key), mode.lower() == "w"))

    @lru_cache(maxsize=None)
    def conflicts(i, j):
        # RR sharing is explicitly not a conflict.
        if not access[i] or not access[j]:
            return True
        left = {}
        right = {}
        for key, write in access[i]:
            left[key] = left.get(key, False) or write
        for key, write in access[j]:
            right[key] = right.get(key, False) or write
        return any(left[key] or right[key] for key in left.keys() & right.keys())

    # Full enumeration is factorial, so retain the strong incumbent if the
    # bounded proof/search cannot finish.  Prefix makespan is monotone under
    # the simulator's precedence scheduling semantics and is therefore a lower
    # bound for every extension.
    deadline = time.monotonic() + 18.0
    node_limit = 300000 if n <= 12 else 120000
    nodes = [0]

    def dfs(prefix, remaining):
        nonlocal best_cost, best_seq
        if nodes[0] >= node_limit or time.monotonic() >= deadline:
            return
        nodes[0] += 1
        lower_bound = cost(prefix) if prefix else 0
        if lower_bound >= best_cost:
            return
        if not remaining:
            best_cost, best_seq = lower_bound, list(prefix)
            return

        candidates = []
        for txn in remaining:
            # For independent transactions, require increasing transaction ID.
            # Their relative order can be swapped without changing any conflict
            # direction, so this removes duplicate commutation classes.
            if any(other < txn and not conflicts(other, txn) for other in remaining):
                continue
            child = prefix + (txn,)
            candidates.append((cost(child), txn, child))
        candidates.sort()
        for _, txn, child in candidates:
            dfs(child, tuple(other for other in remaining if other != txn))

    dfs((), tuple(range(n)))
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
