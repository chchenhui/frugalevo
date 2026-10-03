import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Construct sampled best-insertion schedules and apply bounded local search."""
    import time

    n = workload.num_txns
    if n == 0:
        return 0, []

    deadline = time.monotonic() + 105.0
    cache = {}

    def cost(seq):
        key = tuple(seq)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(list(seq))
        return cache[key]

    def best_insertion_order(start):
        """Insert sampled transactions at the cheapest exact-cost position."""
        order = [start]
        remaining = set(range(n))
        remaining.remove(start)

        while remaining and time.monotonic() < deadline:
            candidates = list(remaining)
            random.shuffle(candidates)
            candidates = candidates[:7]

            best_value = None
            best_order = None
            best_txn = None
            for txn in candidates:
                for position in range(len(order) + 1):
                    trial = order[:]
                    trial.insert(position, txn)
                    value = cost(trial)
                    if best_value is None or value < best_value:
                        best_value = value
                        best_order = trial
                        best_txn = txn

            if best_order is None:
                break
            order = best_order
            remaining.remove(best_txn)

        if remaining:
            order.extend(sorted(remaining))
        return order

    def improve(seq):
        """Use 16 bounded two-transaction ejection chains, then 20 strict relocations."""
        current = cost(seq)

        # A coordinated pair move can cross a barrier that single-item
        # relocation descent cannot cross.  Every trial keeps both removed
        # transaction IDs and tests both possible reinsertion orders.
        for attempt in range(16):
            if n < 2 or time.monotonic() >= deadline:
                break

            # Cycle through pair choices while retaining a small random
            # perturbation between starts, avoiding repeatedly sampling only
            # the same few transaction pairs.
            offset = (attempt * 7) % n
            first_pos = offset
            second_pos = (offset + 1 + random.randrange(n - 1)) % n
            if first_pos == second_pos:
                second_pos = (second_pos + 1) % n

            hi, lo = sorted((first_pos, second_pos), reverse=True)
            first = seq[hi]
            second = seq[lo]
            base = seq[:]
            base.pop(hi)
            base.pop(lo)

            # Sample a duplicate-free ejection neighborhood.  Favor positions
            # near the transactions' original locations and the sequence
            # boundaries, while retaining randomized interior alternatives.
            # This preserves the bounded 24-pair (48 arrangement) budget.
            length = len(base)
            limit = min(24, (length + 1) * (length + 2))
            pairs = []
            seen = set()

            # Positions immediately around the removed locations often expose
            # the conflict chain that made a one-transaction move impossible.
            anchors = set()
            for position in (lo, hi, lo - 1, hi - 1, lo + 1, hi + 1):
                for delta in (-1, 0, 1):
                    p = position + delta
                    if 0 <= p <= length:
                        anchors.add(p)

            for p in sorted(anchors):
                for q in (0, p, p + 1, length + 1):
                    if 0 <= q <= length + 1 and (p, q) not in seen:
                        seen.add((p, q))
                        pairs.append((p, q))
                        if len(pairs) >= limit:
                            break
                if len(pairs) >= limit:
                    break

            candidates = [(p, q)
                          for p in range(length + 1)
                          for q in range(length + 2)
                          if (p, q) not in seen]
            random.shuffle(candidates)
            pairs.extend(candidates[:limit - len(pairs)])

            best_trial = None
            best_value = current
            for a, b in ((first, second), (second, first)):
                for p, q in pairs:
                    if time.monotonic() >= deadline:
                        break
                    trial = base[:]
                    trial.insert(p, a)
                    trial.insert(q, b)
                    value = cost(trial)
                    if value < best_value:
                        best_value = value
                        best_trial = trial
                if time.monotonic() >= deadline:
                    break

            if best_trial is not None:
                seq, current = best_trial, best_value

        # Exploit any ordinary improvements exposed by the paired moves.
        accepted = 0
        while accepted < 20 and time.monotonic() < deadline:
            moves = [(i, j) for i in range(n) for j in range(n)
                     if i != j and j != i + 1]
            random.shuffle(moves)
            changed = False
            for i, j in moves:
                if time.monotonic() >= deadline:
                    break
                trial = seq[:]
                item = trial.pop(i)
                trial.insert(j - (j > i), item)
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

    starts = max(1, min(4, int(num_seqs)))
    chosen = random.sample(range(n), min(starts, n))
    for start in chosen:
        if time.monotonic() >= deadline:
            break
        candidate = best_insertion_order(start)
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
