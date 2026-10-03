# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    std::cin >> n >> m;

    std::vector<std::vector<int>> stacks(m);
    int per_stack = n / m;
    for (int i = 0; i < m; ++i) {
        stacks[i].resize(per_stack);
        for (int j = 0; j < per_stack; ++j) {
            std::cin >> stacks[i][j];
        }
    }

    std::vector<std::pair<int, int>> operations;
    operations.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
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

        // Under the problem constraints every remaining target must exist.
        if (src == -1) {
            return 0;
        }

        if (pos + 1 < static_cast<int>(stacks[src].size())) {
            int dest = -1;
            int best_height = n + 1;
            int best_earliest_box = -1;

            // Primarily use the shortest stack.  For equal heights, avoid
            // burying a stack containing a box needed soon: its minimum label
            // is the earliest box that will eventually require access there.
            for (int i = 0; i < m; ++i) {
                if (i == src) continue;

                int earliest_box = n + 1;
                for (int box : stacks[i]) {
                    earliest_box = std::min(earliest_box, box);
                }

                int current_height = static_cast<int>(stacks[i].size());
                if (dest == -1 ||
                    current_height < best_height ||
                    (current_height == best_height &&
                     earliest_box > best_earliest_box)) {
                    dest = i;
                    best_height = current_height;
                    best_earliest_box = earliest_box;
                }
            }

            // The official constraints always have m = 10. If m were 1,
            // moving a blocking suffix would be impossible.
            if (dest == -1) {
                return 0;
            }

            int first_moved_box = stacks[src][pos + 1];

            std::vector<int> moved(
                stacks[src].begin() + pos + 1,
                stacks[src].end()
            );
            stacks[src].erase(stacks[src].begin() + pos + 1, stacks[src].end());
            stacks[dest].insert(stacks[dest].end(), moved.begin(), moved.end());

            operations.push_back({first_moved_box, dest + 1});
        }

        // The target is now necessarily at the top of its stack.
        if (stacks[src].empty() || stacks[src].back() != target) {
            return 0;
        }

        stacks[src].pop_back();
        operations.push_back({target, 0});
    }

    for (const auto& op : operations) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END