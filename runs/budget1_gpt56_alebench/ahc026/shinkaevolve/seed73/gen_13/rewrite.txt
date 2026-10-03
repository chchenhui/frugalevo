# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    if (!(std::cin >> n >> m)) return 0;

    const int height = n / m;
    std::vector<std::vector<int>> stacks(m);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(height);
        for (int j = 0; j < height; ++j) {
            std::cin >> stacks[i][j];
            --stacks[i][j];  // use 0-indexed box labels internally
        }
    }

    std::vector<std::pair<int, int>> operations;
    operations.reserve(2 * n);

    for (int target = 0; target < n; ++target) {
        int src = -1;
        int pos = -1;

        for (int i = 0; i < m && src == -1; ++i) {
            for (int j = 0; j < static_cast<int>(stacks[i].size()); ++j) {
                if (stacks[i][j] == target) {
                    src = i;
                    pos = j;
                    break;
                }
            }
        }

        // Input is a permutation, so every not-yet-carried box must be found.
        if (src == -1) return 0;

        // Move every box above target to another stack.
        if (pos + 1 < static_cast<int>(stacks[src].size())) {
            int dst = -1;
            for (int i = 0; i < m; ++i) {
                if (i == src) continue;
                if (dst == -1 || stacks[i].size() < stacks[dst].size()) {
                    dst = i;
                }
            }

            // Under the stated constraints m = 10, so a distinct destination exists.
            if (dst == -1) return 0;

            int first_moved_box = stacks[src][pos + 1];
            operations.push_back({first_moved_box + 1, dst + 1});

            stacks[dst].insert(
                stacks[dst].end(),
                stacks[src].begin() + pos + 1,
                stacks[src].end()
            );
            stacks[src].resize(pos + 1);
        }

        // Target is now guaranteed to be at the top of its stack.
        operations.push_back({target + 1, 0});
        stacks[src].pop_back();
    }

    for (const auto& op : operations) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END