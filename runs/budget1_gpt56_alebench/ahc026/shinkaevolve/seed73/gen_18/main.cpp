# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    if (!(std::cin >> n >> m)) return 0;

    const int h = n / m;
    std::vector<std::vector<int>> stacks(m);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(h);
        for (int j = 0; j < h; ++j) {
            std::cin >> stacks[i][j];
        }
    }

    std::vector<std::pair<int, int>> operations;
    operations.reserve(2 * n);

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

        // Input is guaranteed to be a permutation, so target is always found.
        int above = (int)stacks[src].size() - pos - 1;

        if (above > 0) {
            int dest = -1;
            for (int i = 0; i < m; ++i) {
                if (i == src) continue;
                if (dest == -1 || stacks[i].size() < stacks[dest].size()) {
                    dest = i;
                }
            }

            // The operation is specified by the bottom box of the moved block.
            int first_moved_box = stacks[src][pos + 1];
            operations.push_back({first_moved_box, dest + 1});

            stacks[dest].insert(
                stacks[dest].end(),
                stacks[src].begin() + pos + 1,
                stacks[src].end()
            );
            stacks[src].erase(stacks[src].begin() + pos + 1, stacks[src].end());
        }

        // target is now at the top of its stack.
        operations.push_back({target, 0});
        stacks[src].pop_back();
    }

    for (const auto& op : operations) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END