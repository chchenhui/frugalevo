# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    std::cin >> n >> m;

    const int height = n / m;
    std::vector<std::vector<int>> stacks(m);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(height);
        for (int j = 0; j < height; ++j) {
            std::cin >> stacks[i][j];
        }
    }

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

        // The target must exist because boxes are removed only once.
        if (pos + 1 < static_cast<int>(stacks[src].size())) {
            int dest = (src + 1) % m;

            // Move the block immediately above target, including every box above it.
            int first_moved_box = stacks[src][pos + 1];
            std::cout << first_moved_box << ' ' << (dest + 1) << '\n';

            stacks[dest].insert(
                stacks[dest].end(),
                stacks[src].begin() + pos + 1,
                stacks[src].end()
            );
            stacks[src].erase(stacks[src].begin() + pos + 1, stacks[src].end());
        }

        // Now target is at the top of its stack and is the globally smallest remaining box.
        std::cout << target << " 0\n";
        stacks[src].pop_back();
    }

    return 0;
}
# EVOLVE-BLOCK-END