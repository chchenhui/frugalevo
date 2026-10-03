# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

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

    std::vector<std::pair<int, int>> operations;
    operations.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int source = -1;
        int position = -1;

        for (int s = 0; s < m && source == -1; ++s) {
            for (int p = 0; p < (int)stacks[s].size(); ++p) {
                if (stacks[s][p] == target) {
                    source = s;
                    position = p;
                    break;
                }
            }
        }

        // The target is guaranteed to exist because each label is unique
        // and previous targets alone have been removed.
        int above = (int)stacks[source].size() - position - 1;

        if (above > 0) {
            int destination = -1;

            // Pick the shortest stack other than the source. This is not
            // required for correctness, but generally avoids unnecessary
            // piling onto a tall stack.
            for (int s = 0; s < m; ++s) {
                if (s == source) continue;
                if (destination == -1 ||
                    stacks[s].size() < stacks[destination].size()) {
                    destination = s;
                }
            }

            // Under the official constraints m = 10, so a different
            // destination always exists.
            if (destination == -1) {
                return 0;
            }

            int first_moved_box = stacks[source][position + 1];
            operations.push_back({first_moved_box, destination + 1});

            stacks[destination].insert(
                stacks[destination].end(),
                stacks[source].begin() + position + 1,
                stacks[source].end()
            );
            stacks[source].resize(position + 1);
        }

        // target is now the top box of its stack.
        operations.push_back({target, 0});
        stacks[source].pop_back();
    }

    for (const auto& op : operations) {
        std::cout << op.first << ' ' << op.second << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END