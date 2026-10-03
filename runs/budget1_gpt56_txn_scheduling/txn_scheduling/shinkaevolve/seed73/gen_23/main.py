import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Build several conflict-aware greedy schedules, then improve the best one
    with moves evaluated against the complete makespan.
    """
    if workload.num_txns == 0:
        return 0, []

    cost_cache = {}

    def sequence_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    def greedy_from(start_txn):
        sequence = [start_txn]
        remaining = set(range(workload.num_txns))
        remaining.remove(start_txn)

        # At every position, score every feasible next transaction using the
        # simulator's real prefix cost rather than a length-based proxy or a
        # random subset of candidates.
        while remaining:
            candidate_costs = []
            best_prefix_cost = None
            for txn in remaining:
                cost = sequence_cost(sequence + [txn])
                if best_prefix_cost is None or cost < best_prefix_cost:
                    best_prefix_cost = cost
                    candidate_costs = [txn]
                elif cost == best_prefix_cost:
                    candidate_costs.append(txn)

            # Random tie breaking gives repeated starts useful diversification
            # without accepting a candidate with a worse measured cost.
            selected = random.choice(candidate_costs)
            sequence.append(selected)
            remaining.remove(selected)

        return sequence_cost(sequence), sequence

    # Probe pairwise precedence directly with the simulator.  A low-cost
    # two-transaction order is a useful conflict signal that greedy prefix
    # construction cannot recover when its early prefix costs are tied.
    n = workload.num_txns
    preference = [0] * n
    wins = [0] * n
    singleton_cost = [sequence_cost([txn]) for txn in range(n)]

    all_pairs = [(left, right)
                 for left in range(n - 1)
                 for right in range(left + 1, n)]
    # Pair probes are inexpensive prefixes, but cap them for unusually large
    # workloads while retaining a deterministic, evenly distributed sample.
    pair_limit = 1500
    if len(all_pairs) > pair_limit:
        step = len(all_pairs) / float(pair_limit)
        pairs = [all_pairs[int(index * step)] for index in range(pair_limit)]
    else:
        pairs = all_pairs

    for left, right in pairs:
        left_first = sequence_cost([left, right])
        right_first = sequence_cost([right, left])
        advantage = right_first - left_first
        preference[left] += advantage
        preference[right] -= advantage
        if advantage > 0:
            wins[left] += 1
            wins[right] -= 1
        elif advantage < 0:
            wins[left] -= 1
            wins[right] += 1

    # Different stable rankings preserve several interpretations of pairwise
    # evidence.  Randomized tertiary ties prevent transaction identifiers from
    # becoming an accidental precedence rule.
    pairwise_seeds = []
    ranking_modes = (
        lambda txn: (-preference[txn], -wins[txn], singleton_cost[txn]),
        lambda txn: (-wins[txn], -preference[txn], singleton_cost[txn]),
        lambda txn: (-preference[txn], singleton_cost[txn], -wins[txn]),
        lambda txn: (singleton_cost[txn], -preference[txn], -wins[txn]),
    )
    for ranking in ranking_modes:
        tie_break = list(range(n))
        random.shuffle(tie_break)
        tie_position = {txn: index for index, txn in enumerate(tie_break)}
        pairwise_seeds.append(sorted(
            range(n), key=lambda txn: ranking(txn) + (tie_position[txn],)
        ))

    # Starting with different transactions is especially important for
    # read/write hotspots: the first writer or reader can determine the
    # critical dependency chain for the entire schedule.
    starts = list(range(n))
    random.shuffle(starts)
    attempts = max(1, num_seqs)
    best_cost = None
    best_schedule = None

    for schedule in pairwise_seeds:
        cost = sequence_cost(schedule)
        if best_cost is None or cost < best_cost:
            best_cost, best_schedule = cost, schedule

    for attempt in range(attempts):
        if attempt < len(starts):
            start = starts[attempt]
        else:
            start = random.randrange(n)
        cost, schedule = greedy_from(start)
        if cost < best_cost:
            best_cost, best_schedule = cost, schedule

    # Greedy prefixes cannot always see a conflict that becomes critical only
    # after later transactions are added.  Relocation and swap moves evaluate
    # complete schedules and repair those decisions directly.
    for _ in range(4):
        improved_cost = best_cost
        improved_schedule = best_schedule
        n = len(best_schedule)

        for source in range(n):
            shortened = best_schedule[:source] + best_schedule[source + 1:]
            moved_txn = best_schedule[source]
            for destination in range(n):
                candidate = shortened[:destination] + [moved_txn] + shortened[destination:]
                if candidate == best_schedule:
                    continue
                cost = sequence_cost(candidate)
                if cost < improved_cost:
                    improved_cost, improved_schedule = cost, candidate

        for left in range(n):
            for right in range(left + 1, n):
                candidate = best_schedule[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                cost = sequence_cost(candidate)
                if cost < improved_cost:
                    improved_cost, improved_schedule = cost, candidate

        if improved_cost >= best_cost:
            break
        best_cost, best_schedule = improved_cost, improved_schedule

    # A schedule can be locally optimal for one-transaction relocations while
    # still requiring several conflicting transactions to move together.  Use
    # a small destroy-and-repair neighborhood to cross those local minima.
    # Contiguous destruction repairs a congested dependency region, while
    # random destruction can reconnect transactions sharing distant hotspots.
    n = len(best_schedule)
    if n >= 3:
        repair_attempts = max(6, min(16, num_seqs * 2))
        max_destroy = min(6, n)

        for attempt in range(repair_attempts):
            destroy_size = random.randint(3, max_destroy)

            if attempt % 3 == 0:
                # Preserve the rest of the schedule while reopening one
                # complete local region for coordinated reordering.
                first = random.randint(0, n - destroy_size)
                removed = best_schedule[first:first + destroy_size]
                repaired = best_schedule[:first] + best_schedule[first + destroy_size:]
            else:
                removed_positions = set(random.sample(range(n), destroy_size))
                removed = [txn for index, txn in enumerate(best_schedule)
                           if index in removed_positions]
                repaired = [txn for index, txn in enumerate(best_schedule)
                            if index not in removed_positions]

            # At each repair step consider both which transaction should be
            # restored next and where it belongs.  This uses the simulator's
            # measured cost for every candidate partial schedule.
            while removed:
                repair_cost = None
                repair_choices = []

                for txn in removed:
                    for destination in range(len(repaired) + 1):
                        candidate = (repaired[:destination] + [txn] +
                                     repaired[destination:])
                        cost = sequence_cost(candidate)
                        if repair_cost is None or cost < repair_cost:
                            repair_cost = cost
                            repair_choices = [(txn, candidate)]
                        elif cost == repair_cost:
                            repair_choices.append((txn, candidate))

                selected_txn, repaired = random.choice(repair_choices)
                removed.remove(selected_txn)

            repaired_cost = sequence_cost(repaired)
            if repaired_cost < best_cost:
                best_cost, best_schedule = repaired_cost, repaired

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