# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;

struct Policy {
    double height_penalty;
    double top_bonus;
    double noise;
    unsigned int seed;
};

struct Result {
    long long cost;
    vector<pair<int, int>> ops;
};

static unsigned int mix_hash(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static Result simulate_plan(
    const vector<vector<int>>& initial,
    int n,
    int m,
    const Policy& policy
) {
    vector<vector<int>> stacks = initial;
    vector<pair<int, int>> pos(n + 1);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < (int)stacks[i].size(); ++j) {
            pos[stacks[i][j]] = {i, j};
        }
    }

    Result result;
    result.cost = 0;
    result.ops.reserve(2 * n);

    for (int target = 1; target <= n; ++target) {
        int src = pos[target].first;
        int at = pos[target].second;

        if (at + 1 < (int)stacks[src].size()) {
            int dest = -1;
            double best_value = -1e100;

            for (int i = 0; i < m; ++i) {
                if (i == src) continue;

                int mn = n + 1;
                for (int x : stacks[i]) mn = min(mn, x);

                int h = (int)stacks[i].size();
                int top = stacks[i].empty() ? n + 1 : stacks[i].back();

                // The minimum label estimates how soon this stack will need
                // to be accessed.  A taller stack is mildly discouraged.
                double value = (double)mn
                    - policy.height_penalty * h
                    + policy.top_bonus * top;

                // Deterministic tiny perturbation gives diversified choices
                // without making the solver nondeterministic.
                unsigned int z = mix_hash(
                    policy.seed ^
                    (unsigned int)(target * 1009 + src * 97 + i * 7919)
                );
                double jitter = ((z & 65535U) / 65535.0 - 0.5) * policy.noise;
                value += jitter;

                if (dest == -1 || value > best_value) {
                    best_value = value;
                    dest = i;
                }
            }

            int first = at + 1;
            int moved_count = (int)stacks[src].size() - first;
            int moved_box = stacks[src][first];

            result.ops.push_back({moved_box, dest + 1});
            result.cost += moved_count + 1;

            int old_dest_size = (int)stacks[dest].size();
            for (int k = 0; k < moved_count; ++k) {
                int box = stacks[src][first + k];
                stacks[dest].push_back(box);
                pos[box] = {dest, old_dest_size + k};
            }
            stacks[src].resize(first);
        }

        result.ops.push_back({target, 0});
        stacks[src].pop_back();
    }

    return result;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> initial(m);
    int per_stack = n / m;
    for (int i = 0; i < m; ++i) {
        initial[i].resize(per_stack);
        for (int j = 0; j < per_stack; ++j) {
            cin >> initial[i][j];
        }
    }

    vector<Policy> policies;

    // Include a broad deterministic range from "protect urgent stacks"
    // to "keep stacks short".
    const double penalties[] = {
        0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25,
        0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        8.0, 12.0, 18.0, 25.0, 40.0, 70.0, 120.0, 250.0, 1000.0
    };

    for (int i = 0; i < (int)(sizeof(penalties) / sizeof(penalties[0])); ++i) {
        policies.push_back({penalties[i], 0.0, 0.0, (unsigned int)(12345 + i)});
        policies.push_back({penalties[i], 0.015, 0.0, (unsigned int)(54321 + i)});
        policies.push_back({penalties[i], 0.0, 0.18, (unsigned int)(99991 + i * 101)});
    }

    // Additional diversified policies around intermediate tradeoffs.
    unsigned int rng = 2463534242U;
    for (int t = 0; t < 180; ++t) {
        rng = rng * 1664525U + 1013904223U;
        double u = (rng & 0xffffU) / 65535.0;
        double penalty = exp(log(0.01) + u * (log(180.0) - log(0.01)));

        rng = rng * 1664525U + 1013904223U;
        double top_bonus = ((rng & 0xffffU) / 65535.0) * 0.04;

        rng = rng * 1664525U + 1013904223U;
        double noise = ((rng & 0xffffU) / 65535.0) * 0.55;

        policies.push_back({penalty, top_bonus, noise, rng});
    }

    Result best;
    best.cost = numeric_limits<long long>::max();

    for (const Policy& policy : policies) {
        Result cur = simulate_plan(initial, n, m, policy);
        if (cur.cost < best.cost) {
            best = std::move(cur);
        }
    }

    for (const auto& [v, destination] : best.ops) {
        cout << v << ' ' << destination << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END