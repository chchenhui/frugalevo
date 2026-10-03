# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <utility>
#include <limits>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    const int h = n / m;
    vector<vector<int>> stacks(m);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(h);
        for (int j = 0; j < h; ++j) {
            cin >> stacks[i][j];
        }
    }

    vector<pair<int, int>> operations;
    operations.reserve(2 * n);

    auto stack_minimum = [&](const vector<int>& st) {
        if (st.empty()) return n + 1;
        int mn = n + 1;
        for (int x : st) mn = min(mn, x);
        return mn;
    };

    for (int target = 1; target <= n; ++target) {
        int source = -1;
        int pos = -1;

        for (int s = 0; s < m && source == -1; ++s) {
            for (int p = 0; p < (int)stacks[s].size(); ++p) {
                if (stacks[s][p] == target) {
                    source = s;
                    pos = p;
                    break;
                }
            }
        }

        if (source == -1) return 0;

        if (pos + 1 < (int)stacks[source].size()) {
            int destination = -1;
            const int horizon = 32;
            int best_energy = numeric_limits<int>::max();
            int best_boxes = numeric_limits<int>::max();
            int best_max_height = numeric_limits<int>::max();
            int best_buried_urgency = -1;

            for (int candidate = 0; candidate < m; ++candidate) {
                if (candidate == source) continue;

                // Evaluate this placement by replaying a short future interval.
                // During the rollout, retain the cheap urgency-aware policy used
                // previously, then compare the resulting relocation work.
                vector<vector<int>> trial = stacks;
                trial[candidate].insert(
                    trial[candidate].end(),
                    trial[source].begin() + pos + 1,
                    trial[source].end()
                );
                trial[source].resize(pos + 1);
                trial[source].pop_back(); // carry out current target

                int predicted_energy = 0;
                int predicted_boxes = 0;
                int max_height = 0;

                for (int future = target + 1;
                     future <= n && future <= target + horizon;
                     ++future) {
                    int src = -1, at = -1;
                    for (int s = 0; s < m && src == -1; ++s) {
                        for (int p = 0; p < (int)trial[s].size(); ++p) {
                            if (trial[s][p] == future) {
                                src = s;
                                at = p;
                                break;
                            }
                        }
                    }

                    if (at + 1 < (int)trial[src].size()) {
                        int dst = -1;
                        int latest_minimum = -1;
                        int lightest_height = numeric_limits<int>::max();

                        for (int s = 0; s < m; ++s) {
                            if (s == src) continue;
                            int mn = n + 1;
                            for (int x : trial[s]) mn = min(mn, x);
                            int sz = (int)trial[s].size();

                            if (mn > latest_minimum ||
                                (mn == latest_minimum && sz < lightest_height)) {
                                latest_minimum = mn;
                                lightest_height = sz;
                                dst = s;
                            }
                        }

                        int moved = (int)trial[src].size() - (at + 1);
                        predicted_energy += moved + 1;
                        predicted_boxes += moved;
                        trial[dst].insert(
                            trial[dst].end(),
                            trial[src].begin() + at + 1,
                            trial[src].end()
                        );
                        trial[src].resize(at + 1);
                    }
                    trial[src].pop_back();
                }

                int buried_urgency = n + 1;
                for (int s = 0; s < m; ++s) {
                    max_height = max(max_height, (int)trial[s].size());
                    for (int p = 0; p + 1 < (int)trial[s].size(); ++p) {
                        buried_urgency = min(buried_urgency, trial[s][p]);
                    }
                }

                if (destination == -1 ||
                    predicted_energy < best_energy ||
                    (predicted_energy == best_energy &&
                     predicted_boxes < best_boxes) ||
                    (predicted_energy == best_energy &&
                     predicted_boxes == best_boxes &&
                     max_height < best_max_height) ||
                    (predicted_energy == best_energy &&
                     predicted_boxes == best_boxes &&
                     max_height == best_max_height &&
                     buried_urgency > best_buried_urgency)) {
                    destination = candidate;
                    best_energy = predicted_energy;
                    best_boxes = predicted_boxes;
                    best_max_height = max_height;
                    best_buried_urgency = buried_urgency;
                }
            }

            int first_moved = stacks[source][pos + 1];
            operations.push_back({first_moved, destination + 1});

            stacks[destination].insert(
                stacks[destination].end(),
                stacks[source].begin() + pos + 1,
                stacks[source].end()
            );
            stacks[source].resize(pos + 1);
        }

        operations.push_back({target, 0});
        stacks[source].pop_back();
    }

    for (const auto& [v, dst] : operations) {
        cout << v << ' ' << dst << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END