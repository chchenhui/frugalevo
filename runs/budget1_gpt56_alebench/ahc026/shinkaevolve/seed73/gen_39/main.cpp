# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    std::cin >> n >> m;

    const int height = n / m;
    std::vector<std::vector<int>> stacks(m);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(height);
        for (int j = 0; j < height; ++j) {
            std::cin >> stacks[i][j];
        }
    }

    std::vector<std::pair<int, int>> operations;
    operations.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int source = -1;
        int position = -1;

        for (int s = 0; s < m && source == -1; ++s) {
            for (int p = 0; p < (int)stacks[s].size(); ++p) {
                if (stacks[s][p] == target) {
                    source = s;
                    position = p;
                    break;
                }
            }
        }

        // The target is guaranteed to exist because each label is unique
        // and previous targets alone have been removed.
        int above = (int)stacks[source].size() - position - 1;

        if (above > 0) {
            int destination = -1;
            int best_rollout_cost = 1000000000;
            int best_imminent_penalty = 1000000000;
            const int rollout_length = 45;
            const int rollout_end = std::min(n, target + rollout_length - 1);

            // The current move has the same cost for every destination.
            // Compare destinations by simulating the following removals.
            for (int candidate = 0; candidate < m; ++candidate) {
                if (candidate == source) continue;

                std::vector<std::vector<int>> trial = stacks;
                trial[candidate].insert(
                    trial[candidate].end(),
                    trial[source].begin() + position + 1,
                    trial[source].end()
                );
                trial[source].resize(position + 1);

                int rollout_cost = 0;
                for (int next_target = target; next_target <= rollout_end; ++next_target) {
                    int trial_source = -1;
                    int trial_position = -1;
                    for (int s = 0; s < m && trial_source == -1; ++s) {
                        for (int p = 0; p < (int)trial[s].size(); ++p) {
                            if (trial[s][p] == next_target) {
                                trial_source = s;
                                trial_position = p;
                                break;
                            }
                        }
                    }

                    int trial_above =
                        (int)trial[trial_source].size() - trial_position - 1;
                    if (trial_above > 0) {
                        int trial_destination = -1;
                        for (int s = 0; s < m; ++s) {
                            if (s == trial_source) continue;
                            if (trial_destination == -1 ||
                                trial[s].size() < trial[trial_destination].size()) {
                                trial_destination = s;
                            }
                        }

                        rollout_cost += trial_above + 1;
                        trial[trial_destination].insert(
                            trial[trial_destination].end(),
                            trial[trial_source].begin() + trial_position + 1,
                            trial[trial_source].end()
                        );
                        trial[trial_source].resize(trial_position + 1);
                    }
                    trial[trial_source].pop_back();
                }

                // If a box needed soon is already in the candidate stack,
                // this move directly buries it under the whole moved suffix.
                int imminent_penalty = 0;
                for (int box : stacks[candidate]) {
                    if (box > target && box <= rollout_end) {
                        imminent_penalty += rollout_end - box + 1;
                    }
                }

                if (rollout_cost < best_rollout_cost ||
                    (rollout_cost == best_rollout_cost &&
                     (imminent_penalty < best_imminent_penalty ||
                      (imminent_penalty == best_imminent_penalty &&
                       (destination == -1 ||
                        stacks[candidate].size() < stacks[destination].size()))))) {
                    best_rollout_cost = rollout_cost;
                    best_imminent_penalty = imminent_penalty;
                    destination = candidate;
                }
            }

            // Under the official constraints m = 10, so a different
            // destination always exists.
            if (destination == -1) {
                return 0;
            }

            int first_moved_box = stacks[source][position + 1];
            operations.push_back({first_moved_box, destination + 1});

            stacks[destination].insert(
                stacks[destination].end(),
                stacks[source].begin() + position + 1,
                stacks[source].end()
            );
            stacks[source].resize(position + 1);
        }

        // target is now the top box of its stack.
        operations.push_back({target, 0});
        stacks[source].pop_back();
    }

    for (const auto& op : operations) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END