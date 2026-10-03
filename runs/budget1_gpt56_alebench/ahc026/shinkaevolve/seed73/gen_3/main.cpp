# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    std::cin >> n >> m;

    const int h = n / m;
    std::vector<std::vector<int>> stacks(m);
    std::vector<int> position_stack(n + 1);
    std::vector<int> position_height(n + 1);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(h);
        for (int j = 0; j < h; ++j) {
            std::cin >> stacks[i][j];
            position_stack[stacks[i][j]] = i;
            position_height[stacks[i][j]] = j;
        }
    }

    std::vector<std::pair<int, int>> operations;
    operations.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int src = position_stack[target];
        int target_height = position_height[target];

        // Move the entire block above target to another stack if necessary.
        if (target_height + 1 < static_cast<int>(stacks[src].size())) {
            int dest = (src + 1) % m;

            // m is 10 in the official constraints, so a different stack exists.
            // This guard also makes the program safe for a degenerate m == 1 input.
            if (dest == src) {
                return 0;
            }

            int first_moved_box = stacks[src][target_height + 1];
            operations.push_back({first_moved_box, dest + 1});

            int old_dest_size = static_cast<int>(stacks[dest].size());
            for (int idx = target_height + 1; idx < static_cast<int>(stacks[src].size()); ++idx) {
                int box = stacks[src][idx];
                stacks[dest].push_back(box);
                position_stack[box] = dest;
                position_height[box] = old_dest_size + (idx - (target_height + 1));
            }

            stacks[src].resize(target_height + 1);
        }

        // target is now at the top of its stack and is the smallest remaining box.
        operations.push_back({target, 0});
        stacks[src].pop_back();
    }

    for (const auto& [box, destination] : operations) {
        std::cout << box << ' ' << destination << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END