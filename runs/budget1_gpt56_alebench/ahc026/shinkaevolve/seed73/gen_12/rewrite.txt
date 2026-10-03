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
    std::vector<std::pair<int, int> > position(n + 1);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(height);
        for (int j = 0; j < height; ++j) {
            std::cin >> stacks[i][j];
            position[stacks[i][j]] = std::make_pair(i, j);
        }
    }

    std::vector<std::pair<int, int> > operations;
    operations.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int src = position[target].first;
        int target_height = position[target].second;
        int above_count = static_cast<int>(stacks[src].size()) - 1 - target_height;

        if (above_count > 0) {
            int dst = -1;
            for (int i = 0; i < m; ++i) {
                if (i == src) continue;
                if (dst == -1 || stacks[i].size() < stacks[dst].size()) {
                    dst = i;
                }
            }

            // The official constraints have m = 10, so a distinct destination exists.
            if (dst == -1) return 0;

            int first_moved_box = stacks[src][target_height + 1];
            int old_dst_size = static_cast<int>(stacks[dst].size());

            for (int j = target_height + 1; j < static_cast<int>(stacks[src].size()); ++j) {
                stacks[dst].push_back(stacks[src][j]);
            }
            stacks[src].resize(target_height + 1);

            for (int j = 0; j < above_count; ++j) {
                position[stacks[dst][old_dst_size + j]] = std::make_pair(dst, old_dst_size + j);
            }

            operations.push_back(std::make_pair(first_moved_box, dst + 1));
        }

        // target is now at the top of its stack, so carrying it out is legal.
        stacks[src].pop_back();
        operations.push_back(std::make_pair(target, 0));
    }

    for (const auto& op : operations) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END