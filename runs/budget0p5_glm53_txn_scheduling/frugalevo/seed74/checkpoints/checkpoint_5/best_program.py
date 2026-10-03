import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time

# Shared deadline across the three-workload evaluation (hard limit 360s,
# leave margin for startup / caller-side cost recomputation).
_DEADLINE = time.time() + 300.0


_cost_cache = {}


def _cost(workload, seq):
    """Memoized wrapper around the simulator's true cost; greedy prefixes,
    pair costs, and local-search candidates repeat heavily, so caching
    buys extra search iterations within the same deadline."""
    key = tuple(seq)
    c = _cost_cache.get(key)
    if c is None:
        c = workload.get_opt_seq_cost(list(seq))
        _cost_cache[key] = c
    return c


_MAX_PAIR_EVALS = 60


def _build_conflict_graph(workload, deadline):
    """Bounded directed conflict graph from true simulator pair costs.

    For sampled unordered pairs, compare cost([a,b]) vs cost([b,a]): the
    cheaper direction yields an edge weighted by the cost difference, and
    each node's 'pressure' is the sum of incident weights (a hot-key /
    conflict-delay proxy). Pair evaluations are hard-capped at
    _MAX_PAIR_EVALS and abort on deadline; partial weights are fine since
    they only need to rank seeds. Sampled pairs start from a deterministic
    RNG so different workloads behave reproducibly.
    """
    n = workload.num_txns
    weight = [[0.0] * n for _ in range(n)]
    pressure = [0.0] * n
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng = random.Random(999)
    rng.shuffle(pairs)
    done = 0
    for a, b in pairs:
        if done >= _MAX_PAIR_EVALS or time.time() > deadline:
            break
        cab = _cost(workload, [a, b])
        cba = _cost(workload, [b, a])
        w = abs(cab - cba)
        done += 1
        if w > 0.0:
            pressure[a] += w
            pressure[b] += w
            if cab <= cba:
                weight[a][b] = w
            else:
                weight[b][a] = w
    return weight, pressure


def _graph_seed(n, weight, pressure, rng):
    """Kahn-style topological seed: repeatedly pick, among nodes with no
    unplaced predecessors, the one with the largest noisy conflict
    pressure. Multiplicative lognormal noise makes the seeds genuinely
    distinct orderings rather than near-copies. Leftovers from cycles are
    appended by descending pressure."""
    indeg = [0] * n
    succs = [[] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if weight[a][b] > 0:
                succs[a].append(b)
                indeg[b] += 1
    placed = [False] * n
    seq = []
    avail = [v for v in range(n) if indeg[v] == 0]
    noise = [rng.lognormvariate(0.0, 0.6) for _ in range(n)]
    while avail:
        best = max(avail, key=lambda v: pressure[v] * noise[v])
        avail.remove(best)
        placed[best] = True
        seq.append(best)
        for s in succs[best]:
            indeg[s] -= 1
            if indeg[s] == 0 and not placed[s]:
                avail.append(s)
    rest = [v for v in range(n) if not placed[v]]
    rest.sort(key=lambda v: -pressure[v])
    seq.extend(rest)
    return seq


def _local_search(workload, seq, cost, deadline):
    """First-improvement local search over shift + adjacent-swap moves.

    Shift moves remove one transaction and reinsert it at the best of all
    n positions (a mesoscopic move no single swap can realize in one step);
    an adjacent-swap sweep is a cheap complementary pass. Both preserve
    permutation validity; costs are computed via the simulator's true cost.
    """
    n = len(seq)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shift pass: relocate each transaction once, first-improvement.
        for i in range(n):
            if time.time() > deadline:
                return seq, cost
            item = seq[i]
            rest = seq[:i] + seq[i + 1:]
            best_j, best_c = i, cost
            for j in range(n):
                if j == i:
                    continue
                cand = rest[:j] + [item] + rest[j:]
                c = workload.get_opt_seq_cost(cand)
                if c < best_c:
                    best_c, best_j = c, j
                    break  # first improvement
            if best_c < cost:
                rest = seq[:i] + seq[i + 1:]
                seq = rest[:best_j] + [item] + rest[best_j:]
                cost = best_c
                improved = True
        # Adjacent-swap pass (cheap, O(n) evaluations per sweep).
        for i in range(n - 1):
            if time.time() > deadline:
                return seq, cost
            cand = seq[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = workload.get_opt_seq_cost(cand)
            if c < cost:
                seq, cost = cand, c
                improved = True
    return seq, cost


def get_best_schedule(workload, num_seqs):
    """
    Multi-start full greedy + swap local search under a shared time budget.

    Runs full greedy (evaluating every candidate with the simulator's true
    cost) from several random starting transactions, then applies pairwise
    swap hill climbing. Keeps and returns the best valid permutation found
    before the deadline. Time is split conservatively across workloads.
    """
    n = workload.num_txns
    # Reserve time for up to 3 workloads; give each a slice of the budget.
    remaining = _DEADLINE - time.time()
    deadline = time.time() + max(30.0, remaining / 3.0)

    rng = random.Random(12345)
    best_seq = None
    best_cost = None

    # Stage 1 (strictly bounded to 30% of this slice): conflict-graph seeds.
    graph_deadline = time.time() + (deadline - time.time()) * 0.30
    seed_stage_end = graph_deadline
    try:
        weight, pressure = _build_conflict_graph(workload, graph_deadline)
    except Exception:
        weight = pressure = None
    if weight is not None:
        for k in range(3):
            now = time.time()
            if now > seed_stage_end:
                break
            seq = _graph_seed(n, weight, pressure, rng)
            cost = _cost(workload, seq)
            if time.time() < seed_stage_end:
                seq, cost = _local_search(workload, seq, cost, seed_stage_end)
            if best_cost is None or cost < best_cost:
                best_seq, best_cost = seq, cost

    # Stage 2: parent's proven multi-start greedy on the remaining ~70%,
    # guaranteeing we never do worse than the incumbent mechanism.
    starts = list(range(n))
    rng.shuffle(starts)

    for start in starts:
        if time.time() > deadline:
            break
        seq = [start]
        rest = [x for x in range(n) if x != start]
        while rest:
            best_t, best_c, best_idx = None, None, -1
            for idx, t in enumerate(rest):
                c = _cost(workload, seq + [t])
                if best_c is None or c < best_c:
                    best_c, best_t, best_idx = c, t, idx
            seq.append(best_t)
            rest.pop(best_idx)
        cost = _cost(workload, seq)
        if time.time() < deadline:
            seq, cost = _local_search(workload, seq, cost, deadline)
        if best_cost is None or cost < best_cost:
            best_seq, best_cost = seq, cost

    # Guarantee a feasible complete schedule even if the budget expired early.
    if best_seq is None:
        best_seq = list(range(n))
        best_cost = workload.get_opt_seq_cost(best_seq)

    assert len(set(best_seq)) == n
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
