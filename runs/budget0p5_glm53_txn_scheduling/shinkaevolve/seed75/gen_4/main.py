import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

class ScheduleBuilder:
    """Constructs candidate schedules via cost-aware greedy with
    randomized tie-breaking and limited lookahead."""

    def __init__(self, workload, samples_per_step=6, explore_rate=0.15):
        self.workload = workload
        self.n = workload.num_txns
        self.samples_per_step = samples_per_step
        self.explore_rate = explore_rate

    def build(self):
        txn_seq = []
        remaining = set(range(self.n))
        while remaining:
            if len(remaining) == 1:
                txn_seq.append(remaining.pop())
                break
            # occasionally explore: pick random candidate among sampled
            cands = random.sample(remaining,
                                  min(self.samples_per_step, len(remaining)))
            best_t, best_cost = None, None
            for t in cands:
                cost = self.workload.get_opt_seq_cost(txn_seq + [t])
                if best_cost is None or cost < best_cost:
                    best_cost, best_t = cost, t
            if random.random() < self.explore_rate:
                best_t = random.choice(cands)
            txn_seq.append(best_t)
            remaining.remove(best_t)
        return txn_seq


class LocalSearch:
    """Improves a schedule with insertion and swap neighborhood moves."""

    def __init__(self, workload, max_rounds=200):
        self.workload = workload
        self.n = workload.num_txns
        self.max_rounds = max_rounds

    def improve(self, seq):
        best = seq[:]
        best_cost = self.workload.get_opt_seq_cost(best)
        improved = True
        rounds = 0
        while improved and rounds < self.max_rounds:
            improved = False
            rounds += 1
            # insertion moves (shift a txn elsewhere)
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        continue
                    cand = best[:]
                    t = cand.pop(i)
                    cand.insert(j, t)
                    c = self.workload.get_opt_seq_cost(cand)
                    if c < best_cost:
                        best, best_cost = cand, c
                        improved = True
            if improved:
                continue
            # pairwise swaps
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    cand = best[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    c = self.workload.get_opt_seq_cost(cand)
                    if c < best_cost:
                        best, best_cost = cand, c
                        improved = True
                        break
                if improved:
                    break
        return best_cost, best


def get_best_schedule(workload, num_seqs):
    """Pipeline: greedy restarts -> local search -> select best.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    builder = ScheduleBuilder(workload)
    searcher = LocalSearch(workload)
    best_cost, best_seq = None, None

    time_budget = 5.0  # seconds of refinement
    start = time.time()
    for k in range(max(4, num_seqs)):
        seq = builder.build()
        cost, seq = searcher.improve(seq)
        if best_cost is None or cost < best_cost:
            best_cost, best_seq = cost, seq
        if time.time() - start > time_budget:
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
