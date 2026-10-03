# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    if (!(std::cin >> n >> m)) return 0;

    const int height = n / m;
    std::vector<std::vector<int>> stacks(m);
    std::vector<int> stack_of(n + 1), pos_in_stack(n + 1);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(height);
        for (int j = 0; j < height; ++j) {
            int box;
            std::cin >> box;
            stacks[i][j] = box;
            stack_of[box] = i;
            pos_in_stack[box] = j;
        }
    }

    std::vector<std::pair<int, int>> operations;
    operations.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int src = stack_of[target];
        int target_pos = pos_in_stack[target];

        // Move every box above target as one contiguous block.
        if (target_pos + 1 < static_cast<int>(stacks[src].size())) {
            int dst = (src + 1) % m;
            int first_moved_box = stacks[src][target_pos + 1];

            int old_dst_size = static_cast<int>(stacks[dst].size());
            for (int p = target_pos + 1; p < static_cast<int>(stacks[src].size()); ++p) {
                stacks[dst].push_back(stacks[src][p]);
            }

            for (int p = old_dst_size; p < static_cast<int>(stacks[dst].size()); ++p) {
                int box = stacks[dst][p];
                stack_of[box] = dst;
                pos_in_stack[box] = p;
            }

            stacks[src].resize(target_pos + 1);
            operations.push_back({first_moved_box, dst + 1});
        }

        // Target is now at the top and is the globally smallest remaining box.
        stacks[src].pop_back();
        operations.push_back({target, 0});
    }

    for (const auto& op : operations) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END