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

    const int h = n / m;
    std::vector<std::vector<int>> stacks(m);
    std::vector<std::pair<int, int>> pos(n + 1);

    for (int i = 0; i < m; ++i) {
        stacks[i].reserve(n);
        for (int j = 0; j < h; ++j) {
            int v;
            std::cin >> v;
            stacks[i].push_back(v);
            pos[v] = {i, j};
        }
    }

    std::vector<std::pair<int, int>> answer;
    answer.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int src = pos[target].first;
        int target_height = pos[target].second;

        // Move the complete block above target to another stack if necessary.
        if (target_height + 1 < static_cast<int>(stacks[src].size())) {
            int dest = -1;

            // Prefer an empty stack. Otherwise use the currently shortest stack.
            for (int i = 0; i < m; ++i) {
                if (i != src && stacks[i].empty()) {
                    dest = i;
                    break;
                }
            }
            if (dest == -1) {
                for (int i = 0; i < m; ++i) {
                    if (i == src) continue;
                    if (dest == -1 || stacks[i].size() < stacks[dest].size()) {
                        dest = i;
                    }
                }
            }

            // Under the official constraints m = 10, so a destination always exists.
            if (dest == -1) return 0;

            const int old_dest_size = static_cast<int>(stacks[dest].size());
            const int first_moved = stacks[src][target_height + 1];

            for (int idx = target_height + 1; idx < static_cast<int>(stacks[src].size()); ++idx) {
                stacks[dest].push_back(stacks[src][idx]);
            }
            stacks[src].resize(target_height + 1);

            for (int idx = old_dest_size; idx < static_cast<int>(stacks[dest].size()); ++idx) {
                pos[stacks[dest][idx]] = {dest, idx};
            }

            answer.push_back({first_moved, dest + 1});
        }

        // The target is now guaranteed to be at the top of its stack.
        src = pos[target].first;
        if (stacks[src].empty() || stacks[src].back() != target) {
            return 0;
        }

        stacks[src].pop_back();
        answer.push_back({target, 0});
    }

    for (const auto& [v, destination] : answer) {
        std::cout << v << ' ' << destination << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END