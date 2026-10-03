import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Exact-cost beam search followed by variable-neighborhood and
    destroy-and-repair improvement.
    """
    n = workload.num_txns
    if n == 0:
        return 0, []

    cost_cache = {}

    def cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    # Keep several competing partial precedence decisions alive.  This avoids
    # the irreversible choices made by an append-only greedy construction.
    beam_width = max(5, min(14, max(1, num_seqs) + 4))
    beam = [(cost([txn]), [txn], frozenset([txn]))
            for txn in range(n)]

    # The initial beam should contain transactions with differing conflict
    # effects rather than merely arbitrary random starts.
    beam.sort(key=lambda entry: entry[0])
    beam = beam[:beam_width]

    all_txns = frozenset(range(n))

    while len(beam[0][1]) < n:
        expanded = []

        for _, sequence, used in beam:
            remaining = all_txns - used

            # Every legal extension is evaluated with the actual simulator
            # objective, not operation counts or conflict-count proxies.
            for txn in remaining:
                candidate = sequence + [txn]
                expanded.append((cost(candidate), candidate, used | {txn}))

        expanded.sort(key=lambda entry: entry[0])

        # Retain good prefixes, but initially require distinct final
        # transactions.  This keeps the beam from becoming many near-identical
        # copies of one short-sighted prefix.
        next_beam = []
        represented_last = set()
        seen = set()

        for entry in expanded:
            key = tuple(entry[1])
            last = entry[1][-1]
            if key in seen or last in represented_last:
                continue
            next_beam.append(entry)
            seen.add(key)
            represented_last.add(last)
            if len(next_beam) == beam_width:
                break

        # If there were fewer distinct last transactions than beam slots,
        # fill the remaining places strictly by exact prefix score.
        if len(next_beam) < beam_width:
            for entry in expanded:
                key = tuple(entry[1])
                if key in seen:
                    continue
                next_beam.append(entry)
                seen.add(key)
                if len(next_beam) == beam_width:
                    break

        beam = next_beam

    completed = sorted(
        [(entry[0], entry[1]) for entry in beam],
        key=lambda entry: entry[0]
    )

    best_cost, best_schedule = completed[0]

    def best_move_refine(schedule, schedule_cost):
        """
        Variable-neighborhood descent.  Each candidate is scored using the
        full makespan, so accepted moves always improve the true objective.
        """
        current = schedule[:]
        current_cost = schedule_cost
        size = len(current)

        for _ in range(4):
            move_cost = current_cost
            move_schedule = None

            # Transaction relocation repairs precedence inversions involving
            # one high-impact reader or writer.
            if size <= 42:
                insertion_moves = [
                    (source, destination)
                    for source in range(size)
                    for destination in range(size)
                    if source != destination
                ]
            else:
                insertion_moves = [
                    (random.randrange(size), random.randrange(size))
                    for _ in range(10 * size)
                ]

            for source, destination in insertion_moves:
                if source == destination:
                    continue
                candidate = current[:]
                txn = candidate.pop(source)
                candidate.insert(destination, txn)
                candidate_cost = cost(candidate)
                if candidate_cost < move_cost:
                    move_cost = candidate_cost
                    move_schedule = candidate

            # Distant exchanges can change two conflict directions at once.
            if size <= 42:
                swap_moves = [
                    (left, right)
                    for left in range(size - 1)
                    for right in range(left + 1, size)
                ]
            else:
                swap_moves = [
                    tuple(sorted(random.sample(range(size), 2)))
                    for _ in range(8 * size)
                ]

            for left, right in swap_moves:
                candidate = current[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_cost = cost(candidate)
                if candidate_cost < move_cost:
                    move_cost = candidate_cost
                    move_schedule = candidate

            # A reversal can repair an entire locally congested chain.
            if size <= 32:
                reverse_moves = [
                    (first, last)
                    for first in range(size - 2)
                    for last in range(first + 2, size)
                ]
            else:
                reverse_moves = [
                    tuple(sorted(random.sample(range(size), 2)))
                    for _ in range(5 * size)
                ]

            for first, last in reverse_moves:
                candidate = (current[:first] +
                             current[first:last + 1][::-1] +
                             current[last + 1:])
                candidate_cost = cost(candidate)
                if candidate_cost < move_cost:
                    move_cost = candidate_cost
                    move_schedule = candidate

            # Move short blocks jointly.  This captures mutually dependent
            # writer/reader groups that cannot improve when split apart.
            max_block = min(3, size - 1)
            for block_size in range(2, max_block + 1):
                for source in range(size - block_size + 1):
                    block = current[source:source + block_size]
                    remainder = current[:source] + current[source + block_size:]

                    if size <= 35:
                        destinations = range(len(remainder) + 1)
                    else:
                        destinations = random.sample(
                            range(len(remainder) + 1),
                            min(len(remainder) + 1, 6)
                        )

                    for destination in destinations:
                        candidate = (remainder[:destination] + block +
                                     remainder[destination:])
                        if candidate == current:
                            continue
                        candidate_cost = cost(candidate)
                        if candidate_cost < move_cost:
                            move_cost = candidate_cost
                            move_schedule = candidate

            if move_schedule is None:
                break

            current = move_schedule
            current_cost = move_cost

        return current_cost, current

    # Refine several beam survivors because a slightly worse final beam member
    # can have a much stronger nearby local optimum.
    elite_count = min(4, len(completed))
    for seed_cost, seed_schedule in completed[:elite_count]:
        refined_cost, refined_schedule = best_move_refine(
            seed_schedule, seed_cost
        )
        if refined_cost < best_cost:
            best_cost, best_schedule = refined_cost, refined_schedule

    # Destroy-and-repair changes multiple precedence decisions together.  The
    # repair itself remains simulator-guided: each insertion position and each
    # transaction choice is evaluated exactly.
    if n >= 3:
        attempts = max(8, min(20, num_seqs * 2))
        maximum_removed = min(6, n)

        for attempt in range(attempts):
            remove_count = random.randint(3, maximum_removed)

            if attempt % 2 == 0:
                # Reopen a congested local interval.
                start = random.randint(0, n - remove_count)
                removed = best_schedule[start:start + remove_count]
                repaired = (best_schedule[:start] +
                            best_schedule[start + remove_count:])
            else:
                # Reopen related but potentially distant hotspots.
                positions = set(random.sample(range(n), remove_count))
                removed = [
                    txn for index, txn in enumerate(best_schedule)
                    if index in positions
                ]
                repaired = [
                    txn for index, txn in enumerate(best_schedule)
                    if index not in positions
                ]

            while removed:
                insertion_cost = None
                choices = []

                for txn in removed:
                    for destination in range(len(repaired) + 1):
                        candidate = (repaired[:destination] + [txn] +
                                     repaired[destination:])
                        candidate_cost = cost(candidate)

                        if (insertion_cost is None or
                                candidate_cost < insertion_cost):
                            insertion_cost = candidate_cost
                            choices = [(txn, candidate)]
                        elif candidate_cost == insertion_cost:
                            choices.append((txn, candidate))

                txn, repaired = random.choice(choices)
                removed.remove(txn)

            repaired_cost = cost(repaired)
            if repaired_cost < best_cost:
                best_cost, best_schedule = best_move_refine(
                    repaired, repaired_cost
                )

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