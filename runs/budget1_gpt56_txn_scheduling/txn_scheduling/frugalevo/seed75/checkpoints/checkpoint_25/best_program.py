import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Construct true-cost regret schedules and improve them by exact repair."""
    import time

    now = time.perf_counter()
    global _schedule_search_deadline
    if "_schedule_search_deadline" not in globals():
        _schedule_search_deadline = now + 330.0
    deadline = min(_schedule_search_deadline, now + 105.0)

    n = workload.num_txns
    if n == 0:
        return 0, []

    cache = {}

    def cost(seq):
        key = tuple(seq)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(list(key))
        return cache[key]

    identity = list(range(n))
    best_seq = identity[:]
    best_cost = cost(best_seq)

    # Preserve enough time for complete, global insertion construction before
    # applying more expensive destruction/repair moves.
    construct_deadline = min(deadline, now + 48.0)

    def construct(reverse):
        """Build a weighted conflict-tournament order and polish it by exact insertion."""
        if not reverse:
            partial = []
            remaining = list(range(n))
            while remaining and time.perf_counter() < construct_deadline:
                selected = None
                for txn in remaining:
                    options = []
                    for pos in range(len(partial) + 1):
                        trial = partial[:pos] + [txn] + partial[pos:]
                        options.append((cost(trial), pos))
                    value, pos = min(options)
                    item = (value, txn, pos)
                    if selected is None or item < selected:
                        selected = item
                if selected is None:
                    break
                _, txn, pos = selected
                partial.insert(pos, txn)
                remaining.remove(txn)
            partial.extend(remaining)
            return partial

        # Each pair contributes a directed preference.  The edge weight is
        # the exact penalty for choosing the less favorable direction.
        preference = [[0] * n for _ in range(n)]
        for left in range(n):
            if time.perf_counter() >= construct_deadline:
                break
            for right in range(left + 1, n):
                if time.perf_counter() >= construct_deadline:
                    break
                forward = cost([left, right])
                backward = cost([right, left])
                if forward <= backward:
                    preference[right][left] = backward - forward
                else:
                    preference[left][right] = forward - backward

        # Insert strongly ordered transactions first.  A large net score means
        # that the transaction is preferred before many other transactions,
        # making it a more informative anchor for subsequent insertions.
        strength = []
        for txn in range(n):
            outgoing = sum(preference[txn])
            incoming = sum(preference[other][txn] for other in range(n))
            strength.append((outgoing - incoming, txn))
        insertion_order = [
            txn for _, txn in sorted(strength, key=lambda item: (-item[0], item[1]))
        ]

        order = []
        for txn in insertion_order:
            best = None
            for pos in range(len(order) + 1):
                trial = order[:pos] + [txn] + order[pos:]
                penalty = 0
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        penalty += preference[trial[i]][trial[j]]
                candidate = (penalty, pos, txn)
                if best is None or candidate < best[0]:
                    best = (candidate, trial)
            order = best[1]

        # Exact polishing converts the structural seed into a true-cost seed.
        for txn in order[:]:
            if time.perf_counter() >= construct_deadline:
                break
            partial = [x for x in order if x != txn]
            best_trial = None
            for pos in range(len(partial) + 1):
                trial = partial[:pos] + [txn] + partial[pos:]
                candidate = (cost(trial), pos, trial)
                if best_trial is None or candidate[:2] < best_trial[:2]:
                    best_trial = candidate
            order = best_trial[2]

        # Continue exact relocate improvement while the reserved construction
        # budget remains.  Unlike pairwise scoring, this evaluates the complete
        # schedule and can repair interactions among three or more transactions.
        improved = True
        while improved and time.perf_counter() < construct_deadline:
            improved = False
            current_cost = cost(order)
            for index, txn in enumerate(order):
                if time.perf_counter() >= construct_deadline:
                    break
                partial = order[:index] + order[index + 1:]
                best_value = current_cost
                best_order = order
                for pos in range(len(partial) + 1):
                    if pos == index:
                        continue
                    trial = partial[:pos] + [txn] + partial[pos:]
                    value = cost(trial)
                    if value < best_value:
                        best_value = value
                        best_order = trial
                if best_value < current_cost:
                    order = best_order
                    improved = True
                    break
        return order

    for reverse in (False, True):
        if time.perf_counter() >= construct_deadline:
            break
        candidate = construct(reverse)
        candidate_cost = cost(candidate)
        if candidate_cost < best_cost:
            best_seq, best_cost = candidate, candidate_cost

    ruin_count = min(8, max(3, n // 8), n)

    for attempt in range(4):
        if time.perf_counter() >= deadline:
            break

        impacts = []
        for txn in best_seq:
            if time.perf_counter() >= deadline:
                break
            remainder = [x for x in best_seq if x != txn]
            impacts.append((best_cost - cost(remainder), txn))

        if len(impacts) != n:
            break

        impacts.sort(
            key=lambda item: (-item[0], (item[1] - attempt) % max(1, n))
        )
        removed = {txn for _, txn in impacts[:ruin_count]}
        partial = [txn for txn in best_seq if txn not in removed]

        while removed and time.perf_counter() < deadline:
            choice = None
            for txn in sorted(
                removed, key=lambda x: (x - attempt) % max(1, n)
            ):
                for pos in range(len(partial) + 1):
                    trial = partial[:pos] + [txn] + partial[pos:]
                    candidate = (cost(trial), txn, pos, trial)
                    if choice is None or candidate[:3] < choice[:3]:
                        choice = candidate

            if choice is None:
                break
            _, txn, _, partial = choice
            removed.remove(txn)

        partial.extend(sorted(removed))
        candidate = partial
        candidate_cost = cost(candidate)

        # First-improvement mesoscopic relocation.  Try triples before pairs
        # because a complete dependency cluster is less likely to be split
        # when the repair has exposed a three-transaction bottleneck.
        block_moved = False
        for block_len in (3, 2):
            if block_moved or block_len >= n:
                break
            for start in range(n - block_len + 1):
                if block_moved or time.perf_counter() >= deadline:
                    break
                block = candidate[start:start + block_len]
                remainder = candidate[:start] + candidate[start + block_len:]
                for target in range(len(remainder) + 1):
                    if time.perf_counter() >= deadline:
                        break
                    # target == start recreates the original schedule and
                    # cannot improve it, so avoid an unnecessary simulation.
                    if target == start:
                        continue
                    trial = remainder[:target] + block + remainder[target:]
                    value = cost(trial)
                    if value < candidate_cost:
                        candidate, candidate_cost = trial, value
                        block_moved = True
                        break

        # Use adjacent swaps only as a follow-up polish after the cluster move.
        # The deadline check ensures this stage cannot overrun the shared
        # budget reserved for later workload repairs.
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for pos in range(n - 1):
                if time.perf_counter() >= deadline:
                    break
                trial = candidate[:]
                trial[pos], trial[pos + 1] = trial[pos + 1], trial[pos]
                value = cost(trial)
                if value < candidate_cost:
                    candidate, candidate_cost = trial, value
                    improved = True
                    break

        if candidate_cost < best_cost:
            best_seq, best_cost = candidate[:], candidate_cost

    if len(best_seq) != n or set(best_seq) != set(range(n)):
        best_seq = identity
        best_cost = cost(identity)

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