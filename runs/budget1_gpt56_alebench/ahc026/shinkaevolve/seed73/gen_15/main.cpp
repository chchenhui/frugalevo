# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <utility>
#include <limits>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    if (!(std::cin >> n >> m)) return 0;

    const int height = n / m;
    std::vector<std::vector<int>> stacks(m);
    std::vector<std::pair<int, int>> pos(n + 1);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(height);
        for (int j = 0; j < height; ++j) {
            std::cin >> stacks[i][j];
            pos[stacks[i][j]] = {i, j};
        }
    }

    std::vector<std::pair<int, int>> answer;
    answer.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int src = pos[target].first;
        int target_height = pos[target].second;

        // Move the suffix strictly above target if necessary.
        if (target_height + 1 < static_cast<int>(stacks[src].size())) {
            int dest = -1;
            int best_height = std::numeric_limits<int>::max();

            // A destination must differ from the source stack.
            for (int i = 0; i < m; ++i) {
                if (i == src) continue;
                if (static_cast<int>(stacks[i].size()) < best_height) {
                    best_height = static_cast<int>(stacks[i].size());
                    dest = i;
                }
            }

            // The official constraints have m = 10, hence dest always exists.
            const int first_moved_index = target_height + 1;
            const int moved_count =
                static_cast<int>(stacks[src].size()) - first_moved_index;
            const int first_moved_box = stacks[src][first_moved_index];

            answer.push_back({first_moved_box, dest + 1});

            const int old_dest_size = static_cast<int>(stacks[dest].size());
            for (int k = 0; k < moved_count; ++k) {
                int box = stacks[src][first_moved_index + k];
                stacks[dest].push_back(box);
                pos[box] = {dest, old_dest_size + k};
            }
            stacks[src].resize(first_moved_index);
        }

        // Target is now the top box of its stack and can be carried out.
        answer.push_back({target, 0});
        stacks[src].pop_back();
    }

    for (const auto& [v, destination] : answer) {
        std::cout << v << ' ' << destination << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END