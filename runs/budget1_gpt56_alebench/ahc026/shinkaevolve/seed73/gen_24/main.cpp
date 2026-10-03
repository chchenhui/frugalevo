# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    if (!(std::cin >> n >> m)) return 0;

    const int height = n / m;
    std::vector<std::vector<int>> stacks(m, std::vector<int>(height));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < height; ++j) {
            std::cin >> stacks[i][j];
        }
    }

    for (int target = 1; target <= n; ++target) {
        int src = -1;
        int pos = -1;

        for (int i = 0; i < m && src == -1; ++i) {
            for (int j = 0; j < (int)stacks[i].size(); ++j) {
                if (stacks[i][j] == target) {
                    src = i;
                    pos = j;
                    break;
                }
            }
        }

        // Under the stated input conditions every target must exist.
        if (src == -1) return 0;

        // Move all boxes above target as one valid operation.
        if (pos + 1 < (int)stacks[src].size()) {
            int dest = (src + 1) % m;

            // The official constraints guarantee m = 10. This guard keeps
            // the program well-defined for a degenerate one-stack input.
            if (dest == src) return 0;

            int first_moved_box = stacks[src][pos + 1];
            std::cout << first_moved_box << ' ' << (dest + 1) << '\n';

            stacks[dest].insert(
                stacks[dest].end(),
                stacks[src].begin() + pos + 1,
                stacks[src].end()
            );
            stacks[src].erase(stacks[src].begin() + pos + 1, stacks[src].end());
        }

        // Target is now at the top of its stack and is the smallest
        // remaining box, so carrying it out is legal.
        std::cout << target << " 0\n";
        stacks[src].pop_back();
    }

    return 0;
}
# EVOLVE-BLOCK-END