# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <utility>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    if (!(std::cin >> n >> m)) return 0;

    int h = n / m;
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
        int index = pos[target].second;

        // Move every box above target as one legal block.
        if (index + 1 < static_cast<int>(stacks[src].size())) {
            int dst = (src + 1) % m;

            // m is 10 in this problem, but keep destination distinct generally.
            if (dst == src) {
                for (int i = 0; i < m; ++i) {
                    if (i != src) {
                        dst = i;
                        break;
                    }
                }
            }

            int first_moved_box = stacks[src][index + 1];
            int old_dst_size = static_cast<int>(stacks[dst].size());

            for (int p = index + 1; p < static_cast<int>(stacks[src].size()); ++p) {
                stacks[dst].push_back(stacks[src][p]);
                pos[stacks[src][p]] = {dst, old_dst_size + (p - index - 1)};
            }
            stacks[src].resize(index + 1);

            answer.push_back({first_moved_box, dst + 1});
        }

        // Target is now the top box of its stack and can be carried out.
        answer.push_back({target, 0});
        stacks[src].pop_back();
    }

    for (const auto& op : answer) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END