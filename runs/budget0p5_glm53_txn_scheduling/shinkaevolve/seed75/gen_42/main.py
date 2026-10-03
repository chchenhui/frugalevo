import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using greedy construction + insertion-based
    simulated annealing local search.
    """
    _cost_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        c = _cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(seq)
            _cost_cache[key] = c
        return c

    def greedy_from(start_txn):
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(0, workload.num_txns)]
        remaining_txns.remove(start_txn)
        while remaining_txns:
            min_cost = float("inf")
            min_txn = -1
            for t in remaining_txns:
                test_seq = txn_seq + [t]
                cost = eval_cost(test_seq)
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
            txn_seq.append(min_txn)
            remaining_txns.remove(min_txn)
        return txn_seq

    def local_search_insertion(seq, cost, rounds=60):
        # Hybrid neighborhood (insertion + nearby swap) with simulated
        # annealing acceptance. Global best is tracked separately so
        # worsening moves never destroy the best solution found.
        n = len(seq)
        temp = 1.0
        global_best_cost = cost
        global_best_seq = seq[:]
        cur = seq[:]
        cur_cost = cost
        stagnation = 0
        rng = random.Random(1234)
        for r in range(rounds):
            found_any = False
            best_cost = cur_cost
            best_move = None
            # scan a random subset of moves from both neighborhoods
            num_moves = min(300, n * 6)
            for _ in range(num_moves):
                if rng.random() < 0.2:
                    # Or-opt: move a segment of 2-3 txns (kept in order)
                    seg = rng.choice((2, 3))
                    if n <= seg + 1:
                        continue
                    i = rng.randint(0, n - seg)
                    j = rng.randint(0, n - seg)
                    if i == j:
                        continue
                    seg_txns = cur[i:i + seg]
                    del cur[i:i + seg]
                    cur[j:j] = seg_txns
                    c = eval_cost(cur)
                    if c < best_cost or (temp > 0 and c < cur_cost * (1 + temp * 0.01)):
                        if c < best_cost:
                            best_cost = c
                        best_move = ("seg", i, j, seg, c)
                    # undo: remove at j, restore at i
                    del cur[j:j + seg]
                    cur[i:i] = seg_txns
                elif rng.random() < 0.5:
                    # insertion move
                    i = rng.randint(0, n - 1)
                    j = rng.randint(0, n - 1)
                    if i == j:
                        continue
                    t = cur.pop(i)
                    cur.insert(j, t)
                    c = eval_cost(cur)
                    # annealing acceptance: allow slight worsening early
                    if c < best_cost or (temp > 0 and c < cur_cost * (1 + temp * 0.01)):
                        if c < best_cost:
                            best_cost = c
                        best_move = ("ins", i, j, c)
                    # undo
                    t2 = cur.pop(j)
                    cur.insert(i, t2)
                else:
                    # nearby swap move (conflicts are mostly local)
                    if n < 2:
                        continue
                    i = rng.randint(0, n - 2)
                    span = rng.choice((1, 1, 2, 3, 5, 8))
                    j = min(i + span, n - 1)
                    if i == j:
                        continue
                    cur[i], cur[j] = cur[j], cur[i]
                    c = eval_cost(cur)
                    if c < best_cost or (temp > 0 and c < cur_cost * (1 + temp * 0.01)):
                        if c < best_cost:
                            best_cost = c
                        best_move = ("swp", i, j, c)
                    # undo
                    cur[i], cur[j] = cur[j], cur[i]
            if best_move is None:
                # try pure best-improvement full scan fallback
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        t = cur.pop(i)
                        cur.insert(j, t)
                        c = eval_cost(cur)
                        if c < best_cost:
                            best_cost = c
                            best_move = (i, j, c)
                        t2 = cur.pop(j)
                        cur.insert(i, t2)
            if best_move is not None:
                kind, i, j, c = best_move
                if kind == "ins":
                    t = cur.pop(i)
                    cur.insert(j, t)
                elif kind == "seg":
                    seg = best_move[3]
                    seg_txns = cur[i:i + seg]
                    del cur[i:i + seg]
                    cur[j:j] = seg_txns
                else:
                    cur[i], cur[j] = cur[j], cur[i]
                cur_cost = c
                if c < global_best_cost:
                    global_best_cost = c
                    global_best_seq = cur[:]
                found_any = True
            temp *= 0.95
            if temp < 0.05:
                temp = 0.05
            if found_any:
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= 5:
                    break
        return global_best_cost, global_best_seq

    best_cost = float("inf")
    best_seq = None
    num_restarts = max(1, num_seqs)
    for i in range(num_restarts):
        if i < workload.num_txns:
            start_txn = i
        else:
            start_txn = random.randint(0, workload.num_txns - 1)
        seq = greedy_from(start_txn)
        cost = eval_cost(seq)
        cost, seq = local_search_insertion(seq, cost)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

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