import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Build several conflict-aware prefix orders, then refine them with exact
    variable-neighborhood descent and iterated perturbation.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    if n == 0:
        return 0, []

    cost_cache = {}

    def cost(order):
        key = tuple(order)
        value = cost_cache.get(key)
        if value is None:
            value = workload.get_opt_seq_cost(list(order))
            cost_cache[key] = value
        return value

    # Exhaustive search is inexpensive at this scale and is useful both as an
    # exact solver and as a guard against heuristic behavior on tiny inputs.
    if n <= 8:
        best_value = float("inf")
        best_order = None

        def visit(prefix, remaining):
            nonlocal best_value, best_order
            if not remaining:
                value = cost(prefix)
                if value < best_value:
                    best_value = value
                    best_order = prefix[:]
                return
            for index, txn in enumerate(remaining):
                visit(prefix + [txn], remaining[:index] + remaining[index + 1:])

        visit([], list(range(n)))
        return best_value, best_order

    # A wider beam than num_seqs is deliberate: num_seqs is normally small,
    # while conflict-equivalent prefixes often need several alternatives before
    # their later writes reveal which ordering is best.
    beam_width = max(12, min(36, max(1, int(num_seqs)) * 3))
    transactions = tuple(range(n))
    beam = [(0, (), transactions)]

    for depth in range(n):
        expanded = []
        for _, prefix, remaining in beam:
            candidates = list(remaining)
            # Randomization only decides otherwise equal-looking beam paths;
            # all retained paths are still ranked by exact makespan.
            random.shuffle(candidates)
            for txn in candidates:
                next_prefix = prefix + (txn,)
                next_remaining = tuple(item for item in remaining if item != txn)
                expanded.append((cost(next_prefix), next_prefix, next_remaining))

        expanded.sort(key=lambda entry: entry[0])

        # Keep the lowest-cost states, but explicitly retain tied boundary
        # states in a shuffled order before trimming.  This prevents txn-id
        # ordering from repeatedly selecting the same conflict-equivalent path.
        if len(expanded) > beam_width:
            cutoff = expanded[beam_width - 1][0]
            strict = [entry for entry in expanded if entry[0] < cutoff]
            tied = [entry for entry in expanded if entry[0] == cutoff]
            random.shuffle(tied)
            beam = strict + tied[:beam_width - len(strict)]
        else:
            beam = expanded

    beam.sort(key=lambda entry: entry[0])
    finalists = []
    seen = set()
    for value, prefix, _ in beam:
        if prefix not in seen:
            seen.add(prefix)
            finalists.append((value, list(prefix)))

    def best_neighborhood(order, current):
        """Return the strongest improving move among four exact neighborhoods."""
        best_value = current
        best_order = order
        length = len(order)

        # One transaction can move across the transaction responsible for its
        # current critical-path delay.
        for source in range(length):
            txn = order[source]
            reduced = order[:source] + order[source + 1:]
            for destination in range(length):
                if destination == source:
                    continue
                candidate = reduced[:destination] + [txn] + reduced[destination:]
                value = cost(candidate)
                if value < best_value:
                    best_value = value
                    best_order = candidate

        # Exchanges can improve schedules requiring two distant transactions to
        # trade positions without either single insertion being favorable.
        for left in range(length - 1):
            for right in range(left + 1, length):
                candidate = order[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                value = cost(candidate)
                if value < best_value:
                    best_value = value
                    best_order = candidate

        # Ordered cooperating pairs and triples are useful when their internal
        # read/write relationship should remain unchanged while crossing a
        # conflicting region.
        for block_size in (2, 3):
            if length <= block_size:
                continue
            for source in range(length - block_size + 1):
                block = order[source:source + block_size]
                reduced = order[:source] + order[source + block_size:]
                for destination in range(len(reduced) + 1):
                    if destination == source:
                        continue
                    candidate = (
                        reduced[:destination] + block + reduced[destination:]
                    )
                    value = cost(candidate)
                    if value < best_value:
                        best_value = value
                        best_order = candidate

        return best_value, best_order

    def descend(order, initial_value=None):
        value = cost(order) if initial_value is None else initial_value
        while True:
            next_value, next_order = best_neighborhood(order, value)
            if next_value >= value:
                return value, order
            value, order = next_value, next_order

    # Refine multiple independent beam completions rather than only the first.
    # A prefix that is marginally worse can lead to a substantially better
    # conflict graph after all transactions are placed.
    refine_count = min(len(finalists), max(4, min(10, int(num_seqs))))
    best_value = float("inf")
    best_order = None
    local_optima = []

    for initial_value, order in finalists[:refine_count]:
        value, improved = descend(order, initial_value)
        local_optima.append((value, improved))
        if value < best_value:
            best_value, best_order = value, improved

    # Escape local minima by disrupting short regions.  Each kicked schedule is
    # completely re-optimized, while the incumbent is retained unless a true
    # exact-cost improvement is found.
    kick_budget = max(4, min(14, int(num_seqs) + 4))
    sources = [entry[1] for entry in local_optima]
    if not sources:
        sources = [best_order]

    for iteration in range(kick_budget):
        base = sources[iteration % len(sources)]
        candidate = base[:]

        if iteration % 3 == 0:
            # Reverse a medium interval, changing several conflict precedences.
            width = min(n, 3 + (iteration % max(1, min(4, n - 2))))
            start = random.randrange(0, n - width + 1)
            candidate[start:start + width] = reversed(candidate[start:start + width])
        elif iteration % 3 == 1:
            # Move an ordered pair together to another region.
            start = random.randrange(0, n - 1)
            pair = candidate[start:start + 2]
            reduced = candidate[:start] + candidate[start + 2:]
            destination = random.randrange(0, len(reduced) + 1)
            candidate = reduced[:destination] + pair + reduced[destination:]
        else:
            # A nonlocal exchange is a compact way of crossing two conflict
            # barriers simultaneously.
            left, right = random.sample(range(n), 2)
            candidate[left], candidate[right] = candidate[right], candidate[left]

        value, improved = descend(candidate)
        if value < best_value:
            best_value, best_order = value, improved
            sources.append(improved)

    return cost(best_order), best_order

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