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

    // The official constraints have m = 10.  For every box v, first move
    // all boxes above v to another stack, then v is exposed and can be removed.
    for (int v = 1; v <= n; ++v) {
        int src = -1;
        int pos = -1;

        for (int s = 0; s < m && src == -1; ++s) {
            for (int h = 0; h < static_cast<int>(stacks[s].size()); ++h) {
                if (stacks[s][h] == v) {
                    src = s;
                    pos = h;
                    break;
                }
            }
        }

        // Under valid input, every not-yet-removed box must be found.
        if (src == -1) return 0;

        if (pos + 1 < static_cast<int>(stacks[src].size())) {
            // m is 10 for this task, hence this destination is always distinct.
            int dst = (src + 1) % m;

            // If m were one, a blocked box could not be legally exposed.
            // This branch is unreachable under the stated constraints.
            if (dst == src) return 0;

            int first_moved_box = stacks[src][pos + 1];

            std::cout << first_moved_box << ' ' << (dst + 1) << '\n';

            stacks[dst].insert(
                stacks[dst].end(),
                stacks[src].begin() + pos + 1,
                stacks[src].end()
            );
            stacks[src].erase(stacks[src].begin() + pos + 1, stacks[src].end());
        }

        // v is now the top box of its stack and is the smallest remaining box.
        std::cout << v << " 0\n";
        stacks[src].pop_back();
    }

    return 0;
}
# EVOLVE-BLOCK-END