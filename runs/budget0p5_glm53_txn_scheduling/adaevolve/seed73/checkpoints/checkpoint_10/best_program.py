import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Multi-start randomized greedy construction (exhaustive candidate scan
    when affordable) followed by Iterated Local Search combining insertion
    and swap neighborhood moves, all evaluated with the exact makespan
    cost function, run under a per-workload time budget.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    import time

    n = workload.num_txns
    deadline = time.time() + 100.0
    exhaustive = (n * n <= 400)

    def greedy_construct(num_samples):
        start_txn = random.randint(0, n - 1)
        txn_seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            if exhaustive:
                candidates = list(remaining)
            else:
                k = min(num_samples, len(remaining))
                candidates = random.sample(remaining, k)
            min_cost = float('inf')
            min_txn = candidates[0]
            for t in candidates:
                cost = workload.get_opt_seq_cost(txn_seq + [t])
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
            txn_seq.append(min_txn)
            remaining.remove(min_txn)
        return txn_seq

    def local_search(seq, cost):
        # Insertion + swap + 2-opt segment-reversal neighborhoods,
        # first-improvement descent until local optimum or deadline.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # insertion (relocate) moves
            for i in range(n):
                if time.time() >= deadline:
                    return cost
                t = seq.pop(i)
                for j in range(n):
                    if j == i:
                        continue
                    seq.insert(j, t)
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                        break
                    seq.pop(j)
                if improved:
                    break
                seq.insert(i, t)
            if improved:
                continue
            # swap moves
            for i in range(n - 1):
                if time.time() >= deadline:
                    return cost
                for j in range(i + 1, n):
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                        break
                    seq[i], seq[j] = seq[j], seq[i]
                if improved:
                    break
            if improved:
                continue
            # 2-opt: reverse segment [i..j], often fixes many
            # conflict orderings at once
            for i in range(n - 1):
                if time.time() >= deadline:
                    return cost
                for j in range(i + 2, n):
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                        break
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                if improved:
                    break
        return cost

    def perturb(seq, strength):
        s = seq[:]
        for _ in range(strength):
            if random.random() < 0.5:
                # relocate move
                i = random.randint(0, n - 1)
                t = s.pop(i)
                s.insert(random.randint(0, n - 1), t)
            else:
                # swap move (preserves relative structure more)
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                s[i], s[j] = s[j], s[i]
        return s

    best_seq = None
    best_cost = float('inf')
    cur_seq = None
    cur_cost = float('inf')
    stagnation = 0
    strength = max(2, n // 10)
    while time.time() < deadline:
        if best_seq is None:
            # fresh start from greedy construction
            seq = greedy_construct(8)
            stagnation = 0
        elif cur_seq is None or stagnation >= 30:
            # heavy perturbation from best to diversify
            seq = perturb(best_seq, max(4, n // 5))
            stagnation = 0
        else:
            # ILS: perturb current solution
            seq = perturb(cur_seq, strength)
        cost = workload.get_opt_seq_cost(seq)
        cost = local_search(seq, cost)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq[:]
            cur_seq = seq[:]
            cur_cost = cost
            stagnation = 0
            strength = max(2, n // 10)  # reset kick size on improvement
        elif cost < cur_cost:
            # accept improving move relative to current
            cur_seq = seq[:]
            cur_cost = cost
            stagnation += 1
        elif cost <= cur_cost + 2 and random.random() < 0.5:
            # mild annealing: accept near-equal/worse solutions to
            # traverse cost ridges instead of restarting blindly
            cur_seq = seq[:]
            cur_cost = cost
            stagnation += 1
        elif cost == cur_cost and random.random() < 0.3:
            # plateau drift: accept equal-cost solution to wander
            cur_seq = seq[:]
            cur_cost = cost
            stagnation += 1
            strength = min(strength + 1, n // 2)  # escalate kick on stagnation
        else:
            stagnation += 1
            strength = min(strength + 1, n // 2)

    if best_seq is None:
        best_seq = list(range(n))
        best_cost = workload.get_opt_seq_cost(best_seq)
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
