import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def _greedy_construct(workload, n, first):
    """Greedy: repeatedly append the txn that minimizes makespan increase."""
    seq = [first]
    rem = [x for x in range(n) if x != first]
    cur_cost = workload.get_opt_seq_cost(seq)
    while rem:
        best_t = None
        best_c = float('inf')
        for t in rem:
            c = workload.get_opt_seq_cost(seq + [t])
            if c < best_c:
                best_c = c
                best_t = t
        seq.append(best_t)
        rem.remove(best_t)
        cur_cost = best_c
    return cur_cost, seq


def _local_search(workload, seq, cost, deadline_rounds):
    """ILS local search with insertion, swap, and or-opt (2-3) block moves."""
    n = len(seq)
    improved = True
    rounds = 0
    while improved and rounds < deadline_rounds:
        improved = False
        rounds += 1
        # insertion (or-1) moves
        for i in range(n):
            t = seq[i]
            rest = seq[:i] + seq[i + 1:]
            for j in range(n - 1):
                cand = rest[:j] + [t] + rest[j:]
                c = workload.get_opt_seq_cost(cand)
                if c < cost:
                    seq, cost = cand, c
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # swap moves
        for i in range(n):
            for j in range(i + 1, n):
                cand = seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = workload.get_opt_seq_cost(cand)
                if c < cost:
                    seq, cost = cand, c
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # or-opt: relocate blocks of length 2-3
        for length in (2, 3):
            for i in range(n - length + 1):
                block = seq[i:i + length]
                rest = seq[:i] + seq[i + length:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    cand = rest[:j] + block + rest[j:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < cost:
                        seq, cost = cand, c
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return seq, cost


def get_best_schedule(workload, num_seqs):
    """
    Greedy construction from multiple seeds, then ILS with insertion, swap,
    and or-opt (2-3) segment moves; perturbation restarts escape local optima.
    """
    n = workload.num_txns
    if n == 0:
        return 0, []

    best_cost = float('inf')
    best_seq = None

    num_restarts = max(2, min(6, 150 // max(1, n)))
    deadline_rounds = max(20, min(300, 3000 // max(1, n)))

    for restart in range(num_restarts):
        if restart == 0:
            first = 0
        elif restart == 1 and n > 1:
            first = n - 1
        else:
            first = random.randrange(n)
        cost, seq = _greedy_construct(workload, n, first)
        seq, cost = _local_search(workload, seq, cost, deadline_rounds)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

    # perturbation-based ILS from the best solution found so far
    cur_seq = best_seq[:]
    cur_cost = best_cost
    for _ in range(deadline_rounds):
        cand = cur_seq[:]
        # perturb: random segment reinsertion of a small block
        i = random.randrange(n)
        length = random.choice((1, 2, 3))
        end = min(n, i + length)
        block = cand[i:end]
        rest = cand[:i] + cand[end:]
        if not rest:
            break
        j = random.randrange(len(rest) + 1)
        cand = rest[:j] + block + rest[j:]
        c = workload.get_opt_seq_cost(cand)
        seq2, cost2 = _local_search(workload, cand, c, 40)
        if cost2 < cur_cost:
            cur_seq, cur_cost = seq2, cost2
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_seq = cur_seq[:]

    assert best_seq is not None and len(set(best_seq)) == n
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