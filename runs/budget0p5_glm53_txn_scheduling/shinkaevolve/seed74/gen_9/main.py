import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

class ScheduleEvaluator:
    """Wraps the workload cost oracle."""

    def __init__(self, workload):
        self.workload = workload
        self.cache = {}

    def cost(self, seq):
        key = tuple(seq)
        c = self.cache.get(key)
        if c is None:
            c = self.workload.get_opt_seq_cost(list(seq))
            self.cache[key] = c
        return c


class GreedyConstructor:
    """Multi-start greedy with random tie-breaking and candidate sampling."""

    def __init__(self, evaluator, num_txns):
        self.ev = evaluator
        self.n = num_txns

    def build(self, num_samples=12):
        seq = []
        remaining = list(range(self.n))
        start = random.randrange(self.n)
        seq.append(start)
        remaining.remove(start)
        while remaining:
            k = min(num_samples, len(remaining))
            candidates = random.sample(remaining, k)
            best_t, best_c = None, None
            for t in candidates:
                c = self.ev.cost(seq + [t])
                if best_c is None or c < best_c:
                    best_t, best_c = t, c
            seq.append(best_t)
            remaining.remove(best_t)
        return self.ev.cost(seq), seq


class LocalRefiner:
    """Hybrid local search: adjacent swap, random swap, and reinsertion moves."""

    def __init__(self, evaluator):
        self.ev = evaluator

    def refine(self, cost, seq, deadline):
        best_cost, best_seq = cost, list(seq)
        n = len(best_seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # adjacent swap pass
            for i in range(n - 1):
                if time.time() > deadline:
                    return best_cost, best_seq
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = self.ev.cost(cand)
                if c < best_cost:
                    best_cost, best_seq = c, cand
                    improved = True
            if improved:
                continue
            # reinsertion pass: move one txn to another position
            for i in range(n):
                if time.time() > deadline:
                    return best_cost, best_seq
                for j in random.sample(range(n), min(n, 10)):
                    if j == i:
                        continue
                    cand = best_seq[:]
                    t = cand.pop(i)
                    cand.insert(j if j < i else j - 1, t)
                    c = self.ev.cost(cand)
                    if c < best_cost:
                        best_cost, best_seq = c, cand
                        improved = True
                        break
                if improved:
                    break
        return best_cost, best_seq


def get_best_schedule(workload, num_seqs):
    """
    Pipeline: multi-start greedy construction -> hybrid local refinement,
    alternating under a time budget (GRASP-style).
    """
    ev = ScheduleEvaluator(workload)
    constructor = GreedyConstructor(ev, workload.num_txns)
    refiner = LocalRefiner(ev)

    deadline = time.time() + 8.0

    best_cost, best_seq = constructor.build()
    best_cost, best_seq = refiner.refine(best_cost, best_seq, deadline)

    while time.time() < deadline:
        c, s = constructor.build()
        if c < best_cost:
            best_cost, best_seq = c, s
        # short refinement burst
        burst = min(deadline, time.time() + 2.0)
        c2, s2 = refiner.refine(best_cost, best_seq, burst)
        if c2 < best_cost:
            best_cost, best_seq = c2, s2

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