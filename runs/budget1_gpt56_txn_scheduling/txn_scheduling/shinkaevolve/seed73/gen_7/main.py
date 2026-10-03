import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction order using exact search for small inputs,
    beam search for larger inputs, and conflict-aware local refinement.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    from itertools import permutations

    n = workload.num_txns
    if n == 0:
        return 0, []

    cost_cache = {}

    def cost(sequence):
        """Memoized simulator cost for a transaction prefix or full schedule."""
        key = tuple(sequence)
        cached = cost_cache.get(key)
        if cached is None:
            cached = workload.get_opt_seq_cost(list(key))
            cost_cache[key] = cached
        return cached

    # For very small workloads, evaluate every possible serial transaction order.
    # This is exact and avoids relying on any heuristic when the search space is
    # still inexpensive.
    if n <= 8:
        best_schedule = None
        best_cost = float("inf")
        for schedule in permutations(range(n)):
            schedule_cost = cost(schedule)
            if schedule_cost < best_cost:
                best_cost = schedule_cost
                best_schedule = list(schedule)
        return best_cost, best_schedule

    # Keep a wider beam on moderate-sized workloads, where it is practical and
    # substantially reduces the chance of losing a good early ordering.
    requested_width = max(2, int(num_seqs) if num_seqs else 2)
    if n <= 12:
        beam_width = max(64, requested_width * 8)
    elif n <= 25:
        beam_width = max(24, requested_width * 5)
    else:
        beam_width = max(12, requested_width * 3)

    # For large workloads, evaluating every extension of every beam state is
    # costly.  A rotating deterministic candidate subset retains exploration
    # without making runtime quadratic in both beam width and transaction count.
    extension_limit = n if n <= 28 else min(n, max(18, requested_width * 3))

    # A state is (prefix_tuple, remaining_tuple, prefix_makespan).
    initial = tuple(range(n))
    beam = [((), initial, 0)]

    for depth in range(n):
        next_states = []
        seen_prefixes = set()

        for state_index, (prefix, remaining, _) in enumerate(beam):
            if len(remaining) <= extension_limit:
                candidates = remaining
            else:
                # Select evenly distributed candidates.  The offset changes with
                # depth and state index, preventing the search from repeatedly
                # favoring only low-numbered transaction identifiers.
                step = max(1, len(remaining) // extension_limit)
                start = (depth * 17 + state_index * 31 + len(prefix) * 7) % len(remaining)
                chosen = []
                used = set()
                index = start
                while len(chosen) < extension_limit:
                    txn = remaining[index]
                    if txn not in used:
                        chosen.append(txn)
                        used.add(txn)
                    index = (index + step) % len(remaining)
                    if len(used) == len(remaining):
                        break
                candidates = chosen

            for txn in candidates:
                new_prefix = prefix + (txn,)
                if new_prefix in seen_prefixes:
                    continue
                seen_prefixes.add(new_prefix)

                new_remaining = tuple(x for x in remaining if x != txn)
                next_states.append((new_prefix, new_remaining, cost(new_prefix)))

        # Prefix makespan is the true conflict-aware simulator result, not a
        # proxy based on operation count.  Lexicographic ordering makes ties
        # deterministic.
        next_states.sort(key=lambda item: (item[2], item[0]))
        beam = next_states[:beam_width]

        if not beam:
            # Defensive fallback; this should never happen for a valid workload.
            schedule = list(range(n))
            return cost(schedule), schedule

    completed = sorted(beam, key=lambda item: (item[2], item[0]))
    best_cost = completed[0][2]
    best_schedule = list(completed[0][0])

    # Refine several promising beam results.  Insertion moves are especially
    # effective for transactional conflicts because one transaction can be moved
    # before or after a group of conflicting operations in one operation.
    refinement_count = min(len(completed), max(2, requested_width))
    local_budget = 3500 if n <= 30 else 1800

    for candidate_prefix, _, candidate_cost in completed[:refinement_count]:
        current = list(candidate_prefix)
        current_cost = candidate_cost
        evaluations = 0
        improved = True

        while improved and evaluations < local_budget:
            improved = False

            # First-improvement insertion descent.  Restarting after an accepted
            # move makes every later test use the newly improved schedule.
            for source in range(n):
                if improved or evaluations >= local_budget:
                    break

                txn = current[source]
                reduced = current[:source] + current[source + 1:]

                for destination in range(n):
                    if destination == source:
                        continue

                    proposal = reduced[:destination] + [txn] + reduced[destination:]
                    proposal_cost = cost(proposal)
                    evaluations += 1

                    if proposal_cost < current_cost:
                        current = proposal
                        current_cost = proposal_cost
                        improved = True
                        break

                    if evaluations >= local_budget:
                        break

            # If no insertion helps, test exchanges.  Swaps can improve
            # schedules where two conflicting transactions need to cross.
            if not improved and evaluations < local_budget:
                for left in range(n - 1):
                    if improved or evaluations >= local_budget:
                        break
                    for right in range(left + 1, n):
                        proposal = current[:]
                        proposal[left], proposal[right] = proposal[right], proposal[left]
                        proposal_cost = cost(proposal)
                        evaluations += 1

                        if proposal_cost < current_cost:
                            current = proposal
                            current_cost = proposal_cost
                            improved = True
                            break

                        if evaluations >= local_budget:
                            break

        if current_cost < best_cost:
            best_cost = current_cost
            best_schedule = current

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
