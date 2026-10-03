# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    if (!(std::cin >> n >> m)) return 0;

    int h = n / m;
    std::vector<std::vector<int>> stacks(m);
    for (int i = 0; i < m; ++i) {
        stacks[i].resize(h);
        for (int j = 0; j < h; ++j) {
            std::cin >> stacks[i][j];
        }
    }

    std::vector<std::pair<int, int>> operations;
    operations.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int src = -1;
        int pos = -1;

        for (int i = 0; i < m && src == -1; ++i) {
            for (int j = 0; j < (int)stacks[i].size(); ++j) {
                if (stacks[i][j] == target) {
                    src = i;
                    pos = j;
                    break;
                }
            }
        }

        if (src == -1) return 0;

        if (pos + 1 < (int)stacks[src].size()) {
            int dst = -1;
            int best_minimum = -1;
            int best_height = n + 1;

            // Avoid putting boxes on a stack which contains a box needed soon.
            // Among equally safe stacks, use the shorter one.
            for (int i = 0; i < m; ++i) {
                if (i == src) continue;

                int minimum = n + 1;
                for (int box : stacks[i]) {
                    minimum = std::min(minimum, box);
                }

                int current_height = static_cast<int>(stacks[i].size());
                if (dst == -1 ||
                    minimum > best_minimum ||
                    (minimum == best_minimum && current_height < best_height)) {
                    dst = i;
                    best_minimum = minimum;
                    best_height = current_height;
                }
            }

            if (dst == -1) return 0;

            int first_moved_box = stacks[src][pos + 1];
            operations.push_back({first_moved_box, dst + 1});

            stacks[dst].insert(
                stacks[dst].end(),
                stacks[src].begin() + pos + 1,
                stacks[src].end()
            );
            stacks[src].erase(stacks[src].begin() + pos + 1, stacks[src].end());
        }

        // The target is now guaranteed to be at the top of its stack.
        if (stacks[src].empty() || stacks[src].back() != target) return 0;
        stacks[src].pop_back();
        operations.push_back({target, 0});
    }

    for (const auto& op : operations) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END