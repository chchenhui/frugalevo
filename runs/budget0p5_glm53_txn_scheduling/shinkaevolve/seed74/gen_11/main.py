import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    n = workload.num_txns
    cost_fn = workload.get_opt_seq_cost

    def greedy_construct(start_txn):
        txn_seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            best_t, best_c = None, float('inf')
            for t in remaining:
                test = txn_seq + [t]
                c = cost_fn(test)
                if c < best_c:
                    best_c, best_t = c, t
            txn_seq.append(best_t)
            remaining.remove(best_t)
        return cost_fn(txn_seq), txn_seq

    # Warm start: best of a few random greedy constructions
    best_cost, best_seq = float('inf'), None
    for s in random.sample(range(n), min(n, 5)):
        c, sq = greedy_construct(s)
        if c < best_cost:
            best_cost, best_seq = c, sq

    cur_cost, cur_seq = best_cost, best_seq[:]

    # Simulated annealing with mixed move operators
    T0 = max(best_cost * 0.05, 1.0)
    T = T0
    cooling = 0.995
    iters = 4000
    time_budget = iters

    for it in range(time_budget):
        T = T0 * (cooling ** (it / 10.0))
        if T < 0.05:
            T = 0.05
        move = random.random()
        seq = cur_seq
        if move < 0.5 and n > 2:
            # reinsertion: pull out one txn, insert elsewhere
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j:
                continue
            cand = seq[:i] + seq[i+1:]
            cand.insert(j if j < i else j - 0, seq[i])
            # simpler: rebuild
            t = seq[i]
            rest = seq[:i] + seq[i+1:]
            jj = j if j < len(rest) + 1 else len(rest)
            cand = rest[:jj] + [t] + rest[jj:]
        elif move < 0.8 and n > 1:
            # adjacent / nearby swap
            i = random.randrange(n - 1)
            gap = random.choice([1, 2, 3])
            j = min(i + gap, n - 1)
            cand = seq[:]
            cand[i], cand[j] = cand[j], cand[i]
        else:
            # block reversal
            i = random.randrange(n)
            j = random.randrange(n)
            if i > j:
                i, j = j, i
            if j - i < 1:
                continue
            cand = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]

        c = cost_fn(cand)
        delta = c - cur_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            cur_cost, cur_seq = c, cand
            if c < best_cost:
                best_cost, best_seq = c, cand[:]
        # occasional reheat if stuck
        if it % 1500 == 1499:
            cur_cost, cur_seq = best_cost, best_seq[:]

    # final deterministic polish: pass of improving moves
    improved = True
    rounds = 0
    while improved and rounds < 10:
        improved = False
        rounds += 1
        for i in range(n):
            t = best_seq[i]
            rest = best_seq[:i] + best_seq[i+1:]
            for j in range(n):
                cand = rest[:j] + [t] + rest[j:]
                if cand == best_seq:
                    continue
                c = cost_fn(cand)
                if c < best_cost:
                    best_cost, best_seq = c, cand
                    improved = True
                    break
            if improved:
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
