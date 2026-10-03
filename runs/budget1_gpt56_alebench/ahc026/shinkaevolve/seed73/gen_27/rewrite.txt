# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <utility>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    std::cin >> n >> m;

    const int h = n / m;
    std::vector<std::vector<int>> stacks(m);
    std::vector<std::pair<int, int>> pos(n);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(h);
        for (int j = 0; j < h; ++j) {
            std::cin >> stacks[i][j];
            --stacks[i][j];
            pos[stacks[i][j]] = {i, j};
        }
    }

    std::vector<std::pair<int, int>> operations;
    operations.reserve(2 * n);

    for (int target = 0; target < n; ++target) {
        int src = pos[target].first;
        int target_height = pos[target].second;

        int above = static_cast<int>(stacks[src].size()) - target_height - 1;

        if (above > 0) {
            // Under the official constraints m = 10, so a distinct destination exists.
            int dest = (src + 1) % m;

            // Move the block beginning with the box directly above target.
            int first_box = stacks[src][target_height + 1];
            operations.push_back({first_box + 1, dest + 1});

            int old_dest_size = static_cast<int>(stacks[dest].size());
            for (int i = 0; i < above; ++i) {
                int box = stacks[src][target_height + 1 + i];
                stacks[dest].push_back(box);
                pos[box] = {dest, old_dest_size + i};
            }
            stacks[src].resize(target_height + 1);
        }

        // The target is now necessarily at the top of its stack.
        operations.push_back({target + 1, 0});
        stacks[src].pop_back();
    }

    for (const auto& [box, destination] : operations) {
        std::cout << box << ' ' << destination << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END