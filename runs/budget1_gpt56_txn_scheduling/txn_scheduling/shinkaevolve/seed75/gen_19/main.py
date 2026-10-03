import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Construct several simulator-scored orderings with beam search, then improve
    the best beam candidates using relocation and swap variable-neighborhood
    descent.  All decisions are based on the actual simulated makespan.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    cost_cache = {}
    cache_misses = [0]

    beam_width = max(1, min(num_txns, num_seqs))

    # Beam expansion costs at most approximately beam_width * n^2 / 2
    # evaluations.  The remaining budget allows several complete relocation
    # and swap neighborhoods, with the amount scaling naturally by workload.
    evaluation_limit = max(
        3000,
        (beam_width + 20) * num_txns * num_txns
    )

    def schedule_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
            cache_misses[0] += 1
        return cost_cache[key]

    # Keep distinct low-cost prefixes.  Prefix costs come directly from the
    # simulator and therefore incorporate all read/write conflict delays.
    beam = [([], tuple(range(num_txns)), 0)]

    for _ in range(num_txns):
        expanded = []
        for prefix, remaining, _ in beam:
            for txn in remaining:
                candidate = prefix + [txn]
                candidate_remaining = tuple(
                    other for other in remaining if other != txn
                )
                expanded.append(
                    (
                        candidate,
                        candidate_remaining,
                        schedule_cost(candidate)
                    )
                )

        expanded.sort(key=lambda state: (state[2], tuple(state[0])))
        beam = expanded[:beam_width]

    terminal_states = sorted(
        beam,
        key=lambda state: (state[2], tuple(state[0]))
    )

    best_sequence = terminal_states[0][0]
    best_cost = terminal_states[0][2]

    relocation_neighborhood_size = num_txns * max(0, num_txns - 1)
    swap_neighborhood_size = num_txns * max(0, num_txns - 1) // 2

    def improve_seed(start_sequence, start_cost):
        """
        Best-improvement variable-neighborhood descent.

        Relocations are attempted first because they can repair long ordering
        mistakes.  If none helps, swaps provide a different neighborhood that
        can escape a relocation local optimum.  Any accepted swap restarts
        relocation descent, since it changes the usefulness of insertions.
        """
        sequence = start_sequence[:]
        cost = start_cost

        while True:
            # A complete neighborhood scan is required before declaring
            # stagnation.  Stop cleanly when the shared evaluation budget
            # cannot support one more such scan.
            if cache_misses[0] + relocation_neighborhood_size > evaluation_limit:
                break

            relocation_best_sequence = sequence
            relocation_best_cost = cost

            for source_index in range(num_txns):
                moved_txn = sequence[source_index]
                reduced = (
                    sequence[:source_index] + sequence[source_index + 1:]
                )

                for destination_index in range(num_txns):
                    # Re-inserting at the original position changes nothing.
                    if destination_index == source_index:
                        continue

                    candidate = (
                        reduced[:destination_index]
                        + [moved_txn]
                        + reduced[destination_index:]
                    )
                    candidate_cost = schedule_cost(candidate)

                    if candidate_cost < relocation_best_cost:
                        relocation_best_cost = candidate_cost
                        relocation_best_sequence = candidate

            if relocation_best_cost < cost:
                sequence = relocation_best_sequence
                cost = relocation_best_cost
                continue

            # Relocations have stagnated.  A swap is not equivalent to a
            # relocation because it preserves neither intervening order, so it
            # can expose schedules unreachable by improving insertions alone.
            if cache_misses[0] + swap_neighborhood_size > evaluation_limit:
                break

            swap_best_sequence = sequence
            swap_best_cost = cost

            for left in range(num_txns):
                for right in range(left + 1, num_txns):
                    candidate = sequence[:]
                    candidate[left], candidate[right] = (
                        candidate[right],
                        candidate[left]
                    )
                    candidate_cost = schedule_cost(candidate)

                    if candidate_cost < swap_best_cost:
                        swap_best_cost = candidate_cost
                        swap_best_sequence = candidate

            if swap_best_cost < cost:
                sequence = swap_best_sequence
                cost = swap_best_cost
                # A swap improvement may enable new relocation improvements.
                continue

            # True stagnation under both neighborhoods.
            break

        return cost, sequence

    def destroy_and_repair(sequence, removed_indices):
        """
        Preserve the residual order, then use exact simulator costs to insert
        the removed transactions one at a time at their best positions.
        """
        removed_set = set(removed_indices)
        removed_txns = [
            sequence[index] for index in removed_indices
        ]
        repaired = [
            txn for index, txn in enumerate(sequence)
            if index not in removed_set
        ]

        for txn in removed_txns:
            best_insert_cost = float("inf")
            best_insert_sequence = None
            for position in range(len(repaired) + 1):
                if cache_misses[0] >= evaluation_limit:
                    return None
                candidate = (
                    repaired[:position] + [txn] + repaired[position:]
                )
                candidate_cost = schedule_cost(candidate)
                if (
                    candidate_cost < best_insert_cost
                    or (
                        candidate_cost == best_insert_cost
                        and (
                            best_insert_sequence is None
                            or tuple(candidate) < tuple(best_insert_sequence)
                        )
                    )
                ):
                    best_insert_cost = candidate_cost
                    best_insert_sequence = candidate
            repaired = best_insert_sequence

        return best_insert_cost, repaired

    # Keep budget for several perturb-and-descent restarts.  This prevents all
    # available evaluations from being consumed by beam seeds that frequently
    # converge to the same relocation local optimum.
    restart_count = min(3, max(1, num_seqs // 3))
    repair_evaluations = 3 * num_txns
    restart_reserve = restart_count * (
        repair_evaluations
        + relocation_neighborhood_size
        + swap_neighborhood_size
    )

    # The beam maintains alternative prefixes specifically because a slightly
    # worse complete beam schedule can descend to a better local optimum.
    # Evaluate candidates in beam-cost order while the shared budget permits.
    seen_seeds = set()
    for seed_sequence, _, seed_cost in terminal_states:
        seed_key = tuple(seed_sequence)
        if seed_key in seen_seeds:
            continue
        seen_seeds.add(seed_key)

        improved_cost, improved_sequence = improve_seed(
            seed_sequence,
            seed_cost
        )

        if (
            improved_cost < best_cost
            or (
                improved_cost == best_cost
                and tuple(improved_sequence) < tuple(best_sequence)
            )
        ):
            best_cost = improved_cost
            best_sequence = improved_sequence

        # A further seed cannot receive a full relocation scan, so preserving
        # the incumbent is preferable to beginning an incomplete descent.
        if (
            cache_misses[0]
            + relocation_neighborhood_size
            + restart_reserve
            > evaluation_limit
        ):
            break

    # Relocation and swap descent only accepts improving moves and therefore
    # cannot cross a local-optimum barrier.  Remove well-spaced transactions
    # from the incumbent, greedily repair with exact costs, and descend again.
    # The different offsets produce complementary perturbations without
    # relying on random choices or workload-specific transaction features.
    if num_txns >= 4:
        for restart in range(restart_count):
            if (
                cache_misses[0]
                + repair_evaluations
                + relocation_neighborhood_size
                > evaluation_limit
            ):
                break

            offset = restart % num_txns
            removed_indices = sorted({
                offset,
                (offset + num_txns // 3) % num_txns,
                (offset + (2 * num_txns) // 3) % num_txns,
            })
            if len(removed_indices) < 2:
                continue

            repaired = destroy_and_repair(best_sequence, removed_indices)
            if repaired is None:
                break
            repaired_cost, repaired_sequence = repaired
            improved_cost, improved_sequence = improve_seed(
                repaired_sequence,
                repaired_cost
            )

            if (
                improved_cost < best_cost
                or (
                    improved_cost == best_cost
                    and tuple(improved_sequence) < tuple(best_sequence)
                )
            ):
                best_cost = improved_cost
                best_sequence = improved_sequence

    return best_cost, best_sequence

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