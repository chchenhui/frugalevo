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
    std::vector<std::vector<int> > stacks(m);
    std::vector<int> position_stack(n + 1);
    std::vector<int> position_height(n + 1);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(height);
        for (int j = 0; j < height; ++j) {
            std::cin >> stacks[i][j];
            position_stack[stacks[i][j]] = i;
            position_height[stacks[i][j]] = j;
        }
    }

    std::vector<std::pair<int, int> > operations;
    operations.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int src = position_stack[target];
        int pos = position_height[target];

        if (pos + 1 < static_cast<int>(stacks[src].size())) {
            int dst = -1;
            for (int i = 0; i < m; ++i) {
                if (i == src) continue;
                if (dst == -1 || stacks[i].size() < stacks[dst].size()) {
                    dst = i;
                }
            }

            // The constraints guarantee m=10, so dst is always available.
            int first_moved_box = stacks[src][pos + 1];
            int old_dst_size = static_cast<int>(stacks[dst].size());

            for (int i = pos + 1; i < static_cast<int>(stacks[src].size()); ++i) {
                stacks[dst].push_back(stacks[src][i]);
            }
            stacks[src].resize(pos + 1);

            for (int i = old_dst_size; i < static_cast<int>(stacks[dst].size()); ++i) {
                int box = stacks[dst][i];
                position_stack[box] = dst;
                position_height[box] = i;
            }

            operations.push_back(std::make_pair(first_moved_box, dst + 1));
        }

        // target is now guaranteed to be the top box of its stack.
        src = position_stack[target];
        stacks[src].pop_back();
        operations.push_back(std::make_pair(target, 0));
    }

    for (const auto& op : operations) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END