import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using greedy construction + systematic
    first-improvement local search + adaptive ruin-and-recreate.
    """

    def get_full_greedy(perturb_rate=0.10):
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(0, workload.num_txns)
                          if x != start_txn]
        while remaining_txns:
            min_cost = None
            min_txn = -1
            if random.random() < perturb_rate:
                idx = random.randint(0, len(remaining_txns) - 1)
                min_txn = remaining_txns[idx]
                test_seq = txn_seq + [min_txn]
                min_cost = workload.get_opt_seq_cost(test_seq)
            else:
                for t in remaining_txns:
                    test_seq = txn_seq + [t]
                    cost = workload.get_opt_seq_cost(test_seq)
                    if min_cost is None or cost < min_cost:
                        min_cost = cost
                        min_txn = t
            txn_seq.append(min_txn)
            remaining_txns.remove(min_txn)
        assert len(set(txn_seq)) == workload.num_txns
        return workload.get_opt_seq_cost(txn_seq), txn_seq

    def systematic_local_search(seq, cost, deadline):
        # First-improvement sweeps over full relocation + swap
        # neighborhoods (also or-opt with segment length up to 3).
        n = workload.num_txns
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # --- relocations (segment length 1..3) ---
            for seg_len in (1, 2, 3):
                if time.time() >= deadline:
                    break
                for i in range(n - seg_len + 1):
                    if time.time() >= deadline:
                        break
                    seg = seq[i:i + seg_len]
                    base = seq[:i] + seq[i + seg_len:]
                    for pos in range(len(base) + 1):
                        if pos == i:
                            continue
                        cand = base[:pos] + seg + base[pos:]
                        c = workload.get_opt_seq_cost(cand)
                        if c < cost:
                            seq = cand
                            cost = c
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved or time.time() >= deadline:
                continue
            # --- swaps ---
            for i in range(n - 1):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n):
                    cand = seq[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    c = workload.get_opt_seq_cost(cand)
                    if c < cost:
                        seq = cand
                        cost = c
                        improved = True
                        break
                if improved:
                    break
        return cost, seq

    def ruin_and_recreate(seq, cost, deadline, accept_equal=True):
        n = workload.num_txns
        ruin_min, ruin_max = 2, max(3, n // 8)
        while time.time() < deadline:
            k = random.randint(ruin_min, min(ruin_max, n - 1))
            removed_idx = random.sample(range(n), k)
            removed = [seq[i] for i in removed_idx]
            base = [seq[i] for i in range(n) if i not in set(removed_idx)]
            cur_cost = workload.get_opt_seq_cost(base)
            ok = True
            random.shuffle(removed)
            for t in removed:
                best_c = None
                best_pos = -1
                for pos in range(len(base) + 1):
                    cand = base[:pos] + [t] + base[pos:]
                    c = workload.get_opt_seq_cost(cand)
                    if best_c is None or c < best_c:
                        best_c = c
                        best_pos = pos
                if time.time() >= deadline:
                    ok = False
                    break
                base.insert(best_pos, t)
                cur_cost = best_c
            if not ok:
                break
            if cur_cost < cost or (accept_equal and cur_cost == cost):
                seq = base
                cost = min(cost, cur_cost)
        return cost, seq

    time_budget = 20.0
    deadline = time.time() + time_budget
    best_cost = None
    best_seq = None
    while time.time() < deadline:
        cost, seq = get_full_greedy(0.10)
        cost, seq = systematic_local_search(
            seq, cost, min(deadline, time.time() + 4.0))
        cost, seq = ruin_and_recreate(seq, cost, deadline)
        cost, seq = systematic_local_search(
            seq, cost, min(deadline, time.time() + 2.0))
        if best_cost is None or cost < best_cost:
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