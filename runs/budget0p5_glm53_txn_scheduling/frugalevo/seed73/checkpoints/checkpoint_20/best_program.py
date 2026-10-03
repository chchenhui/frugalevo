import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time

def get_best_schedule(workload, num_seqs):
    """
    Approach: breadth-limited beam search over partial permutations with
    last-transaction diversity enforcement, a cached true-cost oracle, and
    an iterated local search tail (or-opt reinsertion + adjacent swaps +
    random 3-perturbation restarts).

    Mechanism (beam-search partial-schedule expansion): the beam maintains
    k diverse partial prefixes alive simultaneously. At each depth, every
    state is expanded with a sampled set of next transactions, each
    successor evaluated with the simulator's TRUE cost
    (workload.get_opt_seq_cost on the extended prefix). Selection enforces
    diversity: at most 2 survivors may share the same last transaction, so
    the beam retains genuinely different orderings rather than clones
    differing only in one slot; the cheapest full permutation wins.

    Refinement tail: identical to before — or-opt fixes long-range
    misplacements, adjacent swaps fix local inversions, and random
    3-position perturbations diversify when stuck, all
    accept-only-if-better, until the per-workload deadline.

    Budget: ~30s per workload (hard per-workload deadline), keeping the
    three-workload total well inside the 360s limit. A cache keyed on the
    sequence tuple avoids recomputing repeated prefixes. A valid identity
    permutation is always available as a fallback before any search, and
    every beam extension appends a distinct unplaced transaction, so any
    completed state is a full permutation of range(n).
    """
    n = workload.num_txns
    t0 = time.time()
    # ~85s per workload x3 = 255s, safely inside the 360s hard limit.
    # Construction is hard-capped at ~10s and only runs when the workload
    # structure was successfully parsed (verified by nonzero affinity).
    # All extra budget flows to the ILS tail (accept-only-if-better), the
    # stage with the best empirical cost-per-second yield.
    deadline = t0 + 85.0
    construct_deadline = t0 + 10.0
    sample_size = max(3, n // 3)
    beam_k = 16
    max_per_last = 3

    cache = {}

    def seq_cost(seq):
        key = tuple(seq)
        c = cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(seq)
            cache[key] = c
        return c

    best_seq = list(range(n))
    best_cost = seq_cost(best_seq)
    completions = []

    # --- Beam search over partial permutations ---
    # State: (cost_of_prefix, prefix_list). Initialized with each single
    # start transaction (capped at beam_k starts for diversity).
    starts = list(range(min(beam_k, n)))
    beam = []
    for s in starts:
        beam.append((seq_cost([s]), [s]))

    depth = 1
    while beam and time.time() < deadline:
        successors = []
        for cost, seq in beam:
            placed = set(seq)
            # Sample unplaced transactions.
            unplaced = [x for x in range(n) if x not in placed]
            if len(unplaced) <= sample_size:
                cands = unplaced
            else:
                cands = random.sample(unplaced, sample_size)
            for t in cands:
                ext = seq + [t]
                c = seq_cost(ext)
                successors.append((c, ext))
            if time.time() > deadline:
                break
        if not successors:
            break
        # Sort by true cost; dedupe identical prefixes; keep beam_k best,
        # capping survivors sharing the same last transaction for diversity.
        successors.sort(key=lambda p: p[0])
        new_beam = []
        seen = set()
        last_count = {}
        for c, seq in successors:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            lt = seq[-1]
            if last_count.get(lt, 0) >= max_per_last:
                continue
            last_count[lt] = last_count.get(lt, 0) + 1
            new_beam.append((c, seq))
            if len(new_beam) >= beam_k:
                break
        # Harvest any completed permutations; keep diverse completions as
        # seeds for the local-search refinement stage.
        for c, seq in new_beam:
            if len(seq) == n:
                completions.append((c, seq))
                if c < best_cost:
                    best_cost, best_seq = c, seq
        beam = [(c, seq) for c, seq in new_beam if len(seq) < n]
        depth += 1
        # Later depths have fewer states; widen the candidate sample.
        if len(beam) <= 4:
            sample_size = max(sample_size, n // 2)

    # --- Conflict-key-clustering greedy construction (extra ILS seeds) ---
    # Transactions sharing data keys should be adjacent so conflict delays
    # are not repeatedly re-armed across distant positions. Build candidate
    # permutations by greedy conflict-affinity insertion with randomized
    # starts/tie-breaks; evaluate with the true oracle; keep only
    # near-competitive orderings as additional ILS seeds.
    def parse_txn(ops):
        rk, wk = set(), set()
        if not isinstance(ops, (list, tuple)):
            ops = str(ops).split()
        for op in ops:
            if isinstance(op, str):
                for p in op.split():
                    if len(p) > 2 and p[1] == '-':
                        (wk if p[0] == 'w' else rk).add(p[2:])
            elif isinstance(op, (tuple, list)) and len(op) >= 2:
                key = str(op[1])
                (wk if op[0] == 'w' else rk).add(key)
        return rk, wk

    raw = None
    for attr in ('txns', 'transactions', 'txn_ops', 'ops', 'workload',
                 '_txns', '_workload'):
        obj = getattr(workload, attr, None)
        if obj:
            raw = obj
            break

    keys = []
    if raw is not None:
        try:
            if hasattr(raw, 'items'):
                items = sorted(raw.items(), key=lambda kv: str(kv[0]))
                tx_op_lists = [it[1] for it in items]
            else:
                tx_op_lists = list(raw)
            for ops in tx_op_lists:
                keys.append(parse_txn(ops))
        except Exception:
            keys = []
    if len(keys) != n:
        keys = None

    # Gate: construction only proceeds if the parse succeeded AND the
    # conflict graph is non-trivial (otherwise seeds would be noise).
    total_aff = 0
    if keys is not None:
        deg = [len(rk) + 2 * len(wk) for rk, wk in keys]

        def aff(i, j):
            rk_i, wk_i = keys[i]
            rk_j, wk_j = keys[j]
            return (len(rk_i & rk_j) + 2 * len(wk_i & wk_j)
                    + len(wk_i & rk_j) + len(rk_i & wk_j))
        for a in range(n):
            for b in range(a + 1, n):
                total_aff += aff(a, b)
    else:
        deg = [1] * n

        def aff(i, j):
            return 0

    constructed = []
    # --- Conflict-graph layered decomposition (extra ILS seeds) ---
    # Union-find splits the conflict graph (edge = pairwise key affinity)
    # into connected components. Inside each component a
    # nearest-conflict-neighbor Hamiltonian path (2-opt improved on the
    # affinity matrix) chains transactions sharing hot keys contiguously,
    # expressing whole-cluster orderings the prefix-greedy beam cannot.
    # Restart 0 uses the deterministic heavy-first component order; later
    # restarts shuffle component order so the TRUE oracle arbitrates
    # clustering vs interleaving. Every full permutation is scored with
    # the true oracle (accept-only-if-better on the incumbent) and pooled
    # with the greedy-insertion candidates below as ILS seeds. Bounded by
    # the same 10s construct deadline; 3 restarts, 2-opt capped at 50
    # improvement sweeps.
    if keys is not None and total_aff > 0:
        parent_uf = list(range(n))

        def uf_find(x):
            while parent_uf[x] != x:
                parent_uf[x] = parent_uf[parent_uf[x]]
                x = parent_uf[x]
            return x

        for a in range(n):
            for b in range(a + 1, n):
                if aff(a, b) > 0:
                    ra, rb = uf_find(a), uf_find(b)
                    if ra != rb:
                        parent_uf[ra] = rb
        comps = {}
        for x in range(n):
            comps.setdefault(uf_find(x), []).append(x)
        comp_list = sorted(comps.values(),
                           key=lambda c: -sum(deg[x] for x in c))

        def chain_path(nodes, start):
            """Nearest-conflict-neighbor path over `nodes` from `start`."""
            path = [start]
            rem = [t for t in nodes if t != start]
            while rem:
                last = path[-1]
                nxt = max(rem, key=lambda t: (aff(t, last), deg[t], -t))
                path.append(nxt)
                rem.remove(nxt)
            return path

        def two_opt(path):
            """Segment reversals raising adjacent-affinity sum; capped."""
            improved = True
            it = 0
            while improved and it < 50 and time.time() < construct_deadline:
                improved = False
                it += 1
                for i in range(len(path) - 1):
                    if time.time() > construct_deadline:
                        return path
                    for j in range(i + 1, len(path)):
                        a0 = path[i - 1] if i > 0 else None
                        a1, b0 = path[i], path[j]
                        b1 = path[j + 1] if j + 1 < len(path) else None
                        before = after = 0
                        if a0 is not None:
                            before += aff(a0, a1)
                            after += aff(a0, b0)
                        if b1 is not None:
                            before += aff(b0, b1)
                            after += aff(a1, b1)
                        if after > before:
                            path[i:j + 1] = path[i:j + 1][::-1]
                            improved = True
            return path

        for r in range(3):
            if time.time() > construct_deadline:
                break
            order = list(comp_list)
            if r > 0:
                random.shuffle(order)
            full = []
            for comp in order:
                if r == 0:
                    start = max(comp, key=lambda x: deg[x])
                else:
                    top = sorted(comp, key=lambda x: -deg[x])[:max(1, len(comp) // 2)]
                    start = random.choice(top)
                p = chain_path(comp, start)
                if len(p) > 3:
                    p = two_opt(p)
                full.extend(p)
            c = seq_cost(full)
            if c < best_cost:
                best_cost, best_seq = c, list(full)
            constructed.append((c, list(full)))
    if keys is not None and total_aff > 0:
        attempts = 0
        while attempts < n and time.time() < construct_deadline:
            attempts += 1
            start_pool = sorted(range(n), key=lambda x: -deg[x])[:max(1, n // 4)]
            cur = [random.choice(start_pool)]
            placed = {cur[0]}
            while len(cur) < n:
                tail = cur[-min(5, len(cur)):]
                best_t, best_v = None, -1.0
                for t in range(n):
                    if t in placed:
                        continue
                    v = sum(aff(t, p) for p in tail) + random.random() * 0.01 * deg[t]
                    if v > best_v:
                        best_v, best_t = v, t
                # Positional insertion: place best_t at the slot (including
                # interior slots) maximizing affinity with its two neighbors,
                # so clustered transactions end up truly adjacent rather than
                # merely appended at the tail.
                if len(cur) == 1:
                    cur.append(best_t)
                else:
                    best_pos, best_pv = 0, -1.0
                    for pos in range(len(cur) + 1):
                        pv = 0.0
                        if pos > 0:
                            pv += aff(best_t, cur[pos - 1])
                        if pos < len(cur):
                            pv += aff(best_t, cur[pos])
                        pv += random.random() * 0.01 * (deg[best_t] + 1)
                        if pv > best_pv:
                            best_pv, best_pos = pv, pos
                    cur.insert(best_pos, best_t)
                placed.add(best_t)
            c = seq_cost(cur)
            if c < best_cost:
                best_cost, best_seq = c, list(cur)
            constructed.append((c, cur))
    # Keep only constructed orderings competitive with the incumbent so the
    # ILS tail is never wasted polishing clearly inferior seeds.
    cutoff = best_cost * 1.02
    best_constructed = [list(s) for c, s in sorted(constructed) if c <= cutoff][:3]

    # --- Iterated local search refinement until deadline ---
    def one_pass(seq, cost):
        """One first-improvement sweep: or-opt reinsertion, bounded segment
        reversal (2-opt style), then adjacent swaps.
        Returns (seq, cost, improved_flag)."""
        # Or-opt reinsertion.
        for i in range(n):
            if time.time() > deadline:
                return seq, cost, False
            item = seq[i]
            rest = seq[:i] + seq[i + 1:]
            for j in range(n):
                if j == i:
                    continue
                cand = rest[:j] + [item] + rest[j:]
                c = seq_cost(cand)
                if c < cost:
                    return cand, c, True
        # Segment reversal: fixes inverted blocks that or-opt/adjacent
        # swaps cannot, common when a transaction is misplaced mid-schedule.
        for i in range(n - 1):
            if time.time() > deadline:
                return seq, cost, False
            for j in range(i + 1, min(n, i + 8)):
                cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                c = seq_cost(cand)
                if c < cost:
                    return cand, c, True
        # Adjacent swaps.
        for i in range(n - 1):
            if time.time() > deadline:
                return seq, cost, False
            cand = list(seq)
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = seq_cost(cand)
            if c < cost:
                return cand, c, True
        return seq, cost, False

    def local_search(seq, cost):
        while time.time() < deadline:
            seq, cost, imp = one_pass(seq, cost)
            if not imp:
                break
        return seq, cost

    best_seq, best_cost = local_search(best_seq, best_cost)
    # ILS threads: polish other diverse beam completions first, then
    # perturb-and-reoptimize the incumbent (3 random transpositions).
    # Beam completions first (proven best path), then only near-competitive
    # clustered constructions; incumbent perturbation fills the remaining time.
    seeds = ([s for _, s in sorted(completions)[:3]] + best_constructed)
    cur_seq, cur_cost = list(best_seq), best_cost
    while time.time() < deadline:
        if seeds:
            cand = seeds.pop(0)
            c = seq_cost(cand)
        else:
            # Ruin-and-recreate perturbation: remove a short random
            # segment and reinsert it at a random position. This matches
            # the or-opt neighborhood used by local search, so the ILS
            # escapes local optima coherently instead of via noisy swaps.
            i = random.randrange(n)
            seg_len = min(random.randint(2, 4), n - i)
            seg = cur_seq[i:i + seg_len]
            rest = cur_seq[:i] + cur_seq[i + seg_len:]
            j = random.randrange(len(rest) + 1)
            cand = rest[:j] + seg + rest[j:]
            c = seq_cost(cand)
        if time.time() > deadline:
            break
        cand, c = local_search(cand, c)
        if c < cur_cost:
            cur_seq, cur_cost = cand, c
        if cur_cost < best_cost:
            best_cost, best_seq = cur_cost, list(cur_seq)

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
