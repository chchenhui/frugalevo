import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction ordering with simulator-guided adaptive
    large-neighborhood search.  All construction, repair, and local-search
    decisions are evaluated by the actual schedule-cost objective.
    """
    n = workload.num_txns

    if n == 0:
        return 0, []
    if n == 1:
        return workload.get_opt_seq_cost([0]), [0]

    cost_cache = {}

    def score(sequence):
        key = tuple(sequence)
        value = cost_cache.get(key)
        if value is None:
            value = workload.get_opt_seq_cost(sequence)
            cost_cache[key] = value
        return value

    # Exhaustive evaluation is inexpensive for very small workloads and gives
    # an actual optimum rather than a heuristic result.
    if n <= 8:
        from itertools import permutations

        best_cost = float("inf")
        best_schedule = None
        for ordering in permutations(range(n)):
            current_cost = score(ordering)
            if current_cost < best_cost:
                best_cost = current_cost
                best_schedule = list(ordering)
        return best_cost, best_schedule

    def ranked_choices(values, limit):
        """Return all values when possible, otherwise a diverse sample."""
        if len(values) <= limit:
            return values[:]
        return random.sample(values, limit)

    def greedy_seed(first_txn=None, noise=0.0):
        """
        Randomized objective-guided construction.  At each step the next
        transaction is selected by the simulator cost of the resulting prefix.
        Noise lets different starts retain useful diversity.
        """
        if first_txn is None:
            first_txn = random.randrange(n)

        schedule = [first_txn]
        remaining = list(range(n))
        remaining.remove(first_txn)

        while remaining:
            # Testing all late choices is particularly useful: nearly complete
            # prefixes expose conflict chains much more accurately.
            if len(remaining) <= 32:
                choices = remaining[:]
            else:
                choices = ranked_choices(remaining, min(30, len(remaining)))

            scored = []
            for txn in choices:
                candidate_cost = score(schedule + [txn])
                scored.append((candidate_cost, txn))

            scored.sort(key=lambda entry: entry[0])
            best_value = scored[0][0]

            # Permit a controlled choice among near-equivalent alternatives.
            # This produces different conflict arrangements for later repair.
            threshold = best_value + noise
            near_best = [txn for value, txn in scored if value <= threshold]
            chosen = random.choice(near_best)

            schedule.append(chosen)
            remaining.remove(chosen)

        return schedule

    def random_insertion_seed():
        """
        A structurally different constructor: choose random transactions and
        insert each at its best tested position, rather than always appending.
        """
        order = list(range(n))
        random.shuffle(order)
        schedule = [order.pop()]

        while order:
            # A sampled set keeps this constructor affordable on large inputs.
            if len(order) <= 12:
                txn_candidates = order[:]
            else:
                txn_candidates = random.sample(order, 12)

            best_cost = float("inf")
            best_moves = []

            # Early construction uses all positions; later schedules use a
            # representative positional sample plus both boundaries.
            if len(schedule) <= 38:
                positions = list(range(len(schedule) + 1))
            else:
                positions = {0, len(schedule)}
                positions.update(
                    random.sample(
                        range(1, len(schedule)),
                        min(28, len(schedule) - 1),
                    )
                )
                positions = list(positions)

            for txn in txn_candidates:
                for position in positions:
                    candidate = schedule[:]
                    candidate.insert(position, txn)
                    candidate_cost = score(candidate)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_moves = [(txn, position)]
                    elif candidate_cost == best_cost:
                        best_moves.append((txn, position))

            chosen_txn, chosen_position = random.choice(best_moves)
            schedule.insert(chosen_position, chosen_txn)
            order.remove(chosen_txn)

        return schedule

    def repair(base, removed):
        """
        Greedy exact-cost repair.  Each step jointly selects a removed
        transaction and its insertion location, allowing the destruction phase
        to reorder the removed transactions instead of restoring their old
        relative order.
        """
        schedule = base[:]
        pending = removed[:]

        while pending:
            best_cost = float("inf")
            best_moves = []

            # Removed neighborhoods are deliberately small, so considering all
            # pending transactions and all insertion positions is practical.
            for txn in pending:
                for position in range(len(schedule) + 1):
                    candidate = schedule[:]
                    candidate.insert(position, txn)
                    candidate_cost = score(candidate)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_moves = [(txn, position)]
                    elif candidate_cost == best_cost:
                        best_moves.append((txn, position))

            txn, position = random.choice(best_moves)
            schedule.insert(position, txn)
            pending.remove(txn)

        return schedule, score(schedule)

    def destroy(schedule, amount, mode):
        """
        Remove a neighborhood.  Different neighborhood shapes expose different
        classes of conflict-order changes.
        """
        if amount >= n:
            amount = n - 1

        if mode == 0:
            # Contiguous removal rearranges an entire local conflict region.
            start = random.randrange(n - amount + 1)
            indices = set(range(start, start + amount))
        elif mode == 1:
            # Spread removal changes distant interacting chains together.
            indices = set(random.sample(range(n), amount))
        else:
            # Remove positions around a random pivot, with a few random extras.
            pivot = random.randrange(n)
            indices = set()
            radius = 0
            while len(indices) < max(1, amount - 2):
                left = pivot - radius
                right = pivot + radius
                if left >= 0:
                    indices.add(left)
                if right < n:
                    indices.add(right)
                radius += 1
            while len(indices) < amount:
                indices.add(random.randrange(n))

        removed = [txn for index, txn in enumerate(schedule) if index in indices]
        retained = [txn for index, txn in enumerate(schedule) if index not in indices]
        return retained, removed

    def local_polish(schedule, current_cost, attempts):
        """
        Simulator-scored first-improvement relocation and swap moves.  The
        moves are sampled on large workloads so that effort remains available
        for larger destroy-and-repair changes.
        """
        current = schedule[:]

        for _ in range(attempts):
            improved = False

            if n <= 34:
                move_count = n * (n - 1)
            else:
                move_count = min(420, 7 * n)

            for _ in range(move_count):
                source = random.randrange(n)
                target = random.randrange(n)
                if source == target:
                    continue

                candidate = current[:]
                txn = candidate.pop(source)
                candidate.insert(target, txn)
                candidate_cost = score(candidate)

                if candidate_cost < current_cost:
                    current = candidate
                    current_cost = candidate_cost
                    improved = True
                    break

            if improved:
                continue

            swap_count = min(260, 5 * n)
            for _ in range(swap_count):
                left, right = random.sample(range(n), 2)
                if left > right:
                    left, right = right, left

                candidate = current[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_cost = score(candidate)

                if candidate_cost < current_cost:
                    current = candidate
                    current_cost = candidate_cost
                    improved = True
                    break

            if not improved:
                break

        return current_cost, current

    # Build a diversified population.  It contains both append-oriented greedy
    # schedules and best-position insertion schedules.
    seed_count = max(8, min(20, num_seqs * 2))
    seeds = []
    starts = list(range(n))
    random.shuffle(starts)

    for seed_index in range(seed_count):
        if seed_index % 4 == 3:
            schedule = random_insertion_seed()
        else:
            # Increasing tiny noise avoids repeated tie-driven schedules.
            schedule = greedy_seed(
                starts[seed_index % n],
                noise=seed_index % 3,
            )
        seeds.append((score(schedule), schedule))

    seeds.sort(key=lambda entry: entry[0])

    # Preserve a compact elite population.  Distinct schedules are important:
    # each one provides a different neighborhood for later destruction.
    elite_limit = min(8, max(4, num_seqs))
    elite = []
    seen = set()
    for seed_cost, seed_schedule in seeds:
        key = tuple(seed_schedule)
        if key not in seen:
            seen.add(key)
            elite.append((seed_cost, seed_schedule))
        if len(elite) >= elite_limit:
            break

    best_cost, best_schedule = elite[0][0], elite[0][1][:]

    # First polish elite seeds so ALNS begins from several stable basins.
    polished = []
    for seed_cost, seed_schedule in elite:
        refined_cost, refined_schedule = local_polish(
            seed_schedule,
            seed_cost,
            3 if n <= 45 else 2,
        )
        polished.append((refined_cost, refined_schedule))
        if refined_cost < best_cost:
            best_cost, best_schedule = refined_cost, refined_schedule[:]

    elite = polished
    elite.sort(key=lambda entry: entry[0])

    # Adaptive large-neighborhood search.  Failed large removals lead to
    # smaller neighborhoods; successful changes permit broader disruption.
    iterations = max(18, min(54, num_seqs * 5))
    neighborhood = max(3, min(6, n // 12 + 2))
    stale = 0

    for iteration in range(iterations):
        # Prefer elite schedules but occasionally repair a less dominant basin.
        parent_index = 0 if random.random() < 0.55 else random.randrange(len(elite))
        parent_cost, parent = elite[parent_index]

        amount = min(n - 1, max(2, neighborhood))
        retained, removed = destroy(parent, amount, iteration % 3)
        candidate, candidate_cost = repair(retained, removed)

        if candidate_cost <= parent_cost:
            candidate_cost, candidate = local_polish(
                candidate,
                candidate_cost,
                2 if n <= 45 else 1,
            )

        candidate_key = tuple(candidate)
        known = {tuple(schedule) for _, schedule in elite}

        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_schedule = candidate[:]
            stale = 0
            neighborhood = min(n - 1, neighborhood + 1)
        else:
            stale += 1

        if candidate_key not in known:
            elite.append((candidate_cost, candidate))
            elite.sort(key=lambda entry: entry[0])
            elite = elite[:elite_limit]

        # Periodically alter the destruction size to escape the basin reached
        # by repeated repairs.
        if stale >= 5:
            neighborhood = random.randint(
                2,
                min(n - 1, max(4, n // 7 + 2)),
            )
            stale = 0

    # A final local refinement of the global incumbent is cheap relative to the
    # full search and catches improvements created near the final repair.
    best_cost, best_schedule = local_polish(
        best_schedule,
        best_cost,
        4 if n <= 45 else 2,
    )

    if workload.debug:
        print("best:", best_cost, best_schedule)

    return best_cost, best_schedule

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