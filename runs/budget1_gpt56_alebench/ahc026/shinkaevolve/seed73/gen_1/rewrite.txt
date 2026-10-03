# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <utility>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    if (!(std::cin >> n >> m)) return 0;

    const int h = n / m;
    std::vector<std::vector<int>> stacks(m);
    std::vector<std::pair<int, int>> pos(n + 1);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(h);
        for (int j = 0; j < h; ++j) {
            std::cin >> stacks[i][j];
            pos[stacks[i][j]] = {i, j};
        }
    }

    std::vector<std::pair<int, int>> answer;
    answer.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int src = pos[target].first;
        int target_height = pos[target].second;

        // Move every box above target as one consecutive block.
        if (target_height + 1 < static_cast<int>(stacks[src].size())) {
            int dst = (src + 1) % m;

            // m is fixed to 10 by the problem, so dst is always different from src.
            int first_moved = stacks[src][target_height + 1];
            int old_dst_size = static_cast<int>(stacks[dst].size());

            answer.push_back({first_moved, dst + 1});

            for (int p = target_height + 1; p < static_cast<int>(stacks[src].size()); ++p) {
                int box = stacks[src][p];
                stacks[dst].push_back(box);
                pos[box] = {dst, old_dst_size + (p - target_height - 1)};
            }
            stacks[src].resize(target_height + 1);
        }

        // The target is now guaranteed to be at the top of its stack.
        answer.push_back({target, 0});
        stacks[src].pop_back();
    }

    for (const auto& [box, destination] : answer) {
        std::cout << box << ' ' << destination << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END