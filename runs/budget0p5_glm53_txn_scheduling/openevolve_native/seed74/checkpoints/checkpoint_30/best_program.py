import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Greedy construction + iterated local search (ILS) with insertion moves.

    Approach:
    1. Greedy: at each step, evaluate every remaining transaction by
       appending it to the partial sequence and computing the true cost
       via workload.get_opt_seq_cost; pick the cheapest (ties random).
    2. Local search: first-improvement over adjacent swaps AND insertion
       moves (remove a txn, reinsert at another position), which is a
       strictly stronger neighborhood than swaps alone.
    3. ILS: perturb the best solution (random swaps / segment reversal),
       re-run local search, accept if better; repeat under a time budget
       to escape local optima.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    import time

    n = workload.num_txns
    deadline = time.time() + 90.0  # global time budget (seconds)

    def greedy_from(start_txn):
        txn_seq = [start_txn]
        remaining = [t for t in range(n) if t != start_txn]
        while remaining:
            best_cost = None
            best_txns = []
            for t in remaining:
                cand = txn_seq + [t]
                cost = workload.get_opt_seq_cost(cand)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_txns = [t]
                elif cost == best_cost:
                    best_txns.append(t)
            pick = random.choice(best_txns)
            txn_seq.append(pick)
            remaining.remove(pick)
        return workload.get_opt_seq_cost(txn_seq), txn_seq

    def local_search(seq, cost, ls_deadline):
        """First-improvement local search over swaps + insertions.

        Bounded by ls_deadline so a single call cannot consume the
        whole time budget; alternates adjacent swaps with insertion
        moves (a strictly larger neighborhood than swaps alone).
        """
        while time.time() < ls_deadline:
            improved = False
            # adjacent swaps
            for i in range(len(seq) - 1):
                cand = seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = workload.get_opt_seq_cost(cand)
                if c is not None and c < cost:
                    seq, cost = cand, c
                    improved = True
                if time.time() > ls_deadline:
                    return cost, seq
            # insertion moves
            for i in range(len(seq)):
                for j in range(len(seq)):
                    if i == j:
                        continue
                    cand = seq[:]
                    t = cand.pop(i)
                    cand.insert(j, t)
                    c = workload.get_opt_seq_cost(cand)
                    if c is not None and c < cost:
                        seq, cost = cand, c
                        improved = True
                    if time.time() > ls_deadline:
                        return cost, seq
            if not improved:
                break
        return cost, seq

    def perturb(seq):
        """Random perturbation: swaps, segment reversal, or block move."""
        cand = seq[:]
        mode = random.random()
        if mode < 0.4 and len(cand) >= 4:
            # block move: relocate a small contiguous block elsewhere
            i = random.randrange(len(cand) - 1)
            blen = min(3, len(cand) - i)
            block = cand[i:i + blen]
            rest = cand[:i] + cand[i + blen:]
            j = random.randrange(len(rest) + 1)
            cand = rest[:j] + block + rest[j:]
        else:
            for _ in range(3):
                i, j = random.sample(range(len(cand)), 2)
                cand[i], cand[j] = cand[j], cand[i]
            if len(cand) >= 4:
                i = random.randrange(len(cand) - 3)
                j = i + random.randrange(3, min(len(cand) - i, 8) + 1)
                cand[i:j] = cand[i:j][::-1]
        return cand

    # Greedy restarts
    if n <= 15:
        starts = list(range(n))
    else:
        starts = random.sample(range(n), min(6, n))

    best_cost = None
    best_seq = None
    for s in starts:
        cost, seq = greedy_from(s)
        cost, seq = local_search(seq, cost, deadline)
        if best_cost is None or cost < best_cost:
            best_cost, best_seq = cost, seq
        if time.time() > deadline:
            break

    # Iterated local search around the best solution found.
    # Accept equal-cost moves (plateau walking) to keep the search
    # moving when it is stuck, while always tracking the true best.
    cur_seq = best_seq
    cur_cost = best_cost
    start_time = time.time()
    while time.time() < deadline:
        cand = perturb(cur_seq)
        c = workload.get_opt_seq_cost(cand)
        if c is None:
            continue
        c, cand = local_search(cand, c, deadline)
        # threshold acceptance: allow small worsening that shrinks over time
        frac = (deadline - time.time()) / max(1e-9, deadline - start_time)
        threshold = max(0, int(round(2 * frac)))
        if c < cur_cost + threshold:
            cur_cost, cur_seq = c, cand
            if c < best_cost:
                best_cost, best_seq = c, cand
        # occasionally restart from the best to refocus the search
        if random.random() < 0.05:
            cur_seq, cur_cost = best_seq, best_cost

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
