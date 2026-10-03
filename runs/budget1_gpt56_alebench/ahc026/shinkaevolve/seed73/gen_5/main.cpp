# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    if (!(std::cin >> n >> m)) return 0;

    int height = n / m;
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

        // The input is a permutation, so target must always be present.
        if (source == -1) return 0;

        if (position + 1 < (int)stacks[source].size()) {
            int destination = -1;
            int best_height = 1 << 30;

            // Choose a different stack with minimum current height.
            for (int s = 0; s < m; ++s) {
                if (s == source) continue;
                if ((int)stacks[s].size() < best_height) {
                    best_height = (int)stacks[s].size();
                    destination = s;
                }
            }

            // Under the official constraints m = 10, destination always exists.
            if (destination == -1) return 0;

            int first_moved_box = stacks[source][position + 1];
            operations.push_back({first_moved_box, destination + 1});

            stacks[destination].insert(
                stacks[destination].end(),
                stacks[source].begin() + position + 1,
                stacks[source].end()
            );
            stacks[source].erase(
                stacks[source].begin() + position + 1,
                stacks[source].end()
            );
        }

        // target is now guaranteed to be the top box of its stack.
        operations.push_back({target, 0});
        stacks[source].pop_back();
    }

    for (const auto& op : operations) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END