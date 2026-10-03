# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <utility>
#include <limits>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    const int h = n / m;
    vector<vector<int>> stacks(m);

    for (int i = 0; i < m; ++i) {
        stacks[i].resize(h);
        for (int j = 0; j < h; ++j) {
            cin >> stacks[i][j];
        }
    }

    vector<pair<int, int>> operations;
    operations.reserve(2 * n);

    auto stack_minimum = [&](const vector<int>& st) {
        if (st.empty()) return n + 1;
        int mn = n + 1;
        for (int x : st) mn = min(mn, x);
        return mn;
    };

    for (int target = 1; target <= n; ++target) {
        int source = -1;
        int pos = -1;

        for (int s = 0; s < m && source == -1; ++s) {
            for (int p = 0; p < (int)stacks[s].size(); ++p) {
                if (stacks[s][p] == target) {
                    source = s;
                    pos = p;
                    break;
                }
            }
        }

        if (source == -1) return 0;

        if (pos + 1 < (int)stacks[source].size()) {
            int destination = -1;
            int best_minimum = -1;
            int best_height = numeric_limits<int>::max();

            for (int s = 0; s < m; ++s) {
                if (s == source) continue;

                int mn = stack_minimum(stacks[s]);
                int sz = (int)stacks[s].size();

                // Prefer the stack that will be required latest.
                // If equally safe, prefer the lighter stack.
                if (mn > best_minimum ||
                    (mn == best_minimum && sz < best_height)) {
                    best_minimum = mn;
                    best_height = sz;
                    destination = s;
                }
            }

            int first_moved = stacks[source][pos + 1];
            operations.push_back({first_moved, destination + 1});

            stacks[destination].insert(
                stacks[destination].end(),
                stacks[source].begin() + pos + 1,
                stacks[source].end()
            );
            stacks[source].resize(pos + 1);
        }

        operations.push_back({target, 0});
        stacks[source].pop_back();
    }

    for (const auto& [v, dst] : operations) {
        cout << v << ' ' << dst << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END