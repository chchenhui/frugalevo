# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

using Stacks = vector<vector<int>>;

static pair<int, int> locate_box(const Stacks& stacks, int value) {
    for (int s = 0; s < (int)stacks.size(); ++s) {
        for (int p = 0; p < (int)stacks[s].size(); ++p) {
            if (stacks[s][p] == value) return {s, p};
        }
    }
    return {-1, -1};
}

/*
 * Estimate the damage caused by putting a suffix of length moved onto each
 * stack. Existing boxes in that stack become moved levels deeper. Small labels
 * are much more important because they will need to be extracted sooner.
 */
static int choose_destination(
    const Stacks& stacks,
    int source,
    int moved,
    int next_target,
    int n
) {
    int best = -1;
    long long best_score = (1LL << 60);

    for (int d = 0; d < (int)stacks.size(); ++d) {
        if (d == source) continue;

        long long score = 0;

        // A mild balancing term prevents pathological concentration.
        score += 7LL * (long long)stacks[d].size() * stacks[d].size();

        for (int x : stacks[d]) {
            if (x < next_target) continue;
            int dist = x - next_target;

            // The closest upcoming labels are particularly expensive to bury.
            long long weight = 2400LL / (dist + 6);
            score += weight * moved;

            // Still distinguish labels outside the main imminent range.
            if (dist <= 24) score += 18LL * moved;
        }

        if (score < best_score ||
            (score == best_score &&
             (best == -1 || stacks[d].size() < stacks[best].size())) ||
            (score == best_score && best != -1 &&
             stacks[d].size() == stacks[best].size() && d < best)) {
            best_score = score;
            best = d;
        }
    }
    return best;
}

static void move_suffix(Stacks& stacks, int source, int pos, int dest) {
    stacks[dest].insert(
        stacks[dest].end(),
        stacks[source].begin() + pos + 1,
        stacks[source].end()
    );
    stacks[source].erase(stacks[source].begin() + pos + 1, stacks[source].end());
}

/*
 * Priority-aware deterministic continuation.  Energy is multiplied by 100 so
 * that the terminal obstruction estimate only resolves strategically close
 * rollout outcomes rather than overwhelming actual movement costs.
 */
static long long rollout_score(
    Stacks state,
    int first_target,
    int n,
    int horizon
) {
    long long energy = 0;
    int last = min(n, first_target + horizon - 1);

    for (int target = first_target; target <= last; ++target) {
        auto [source, pos] = locate_box(state, target);
        if (source < 0) return (1LL << 60);

        int above = (int)state[source].size() - pos - 1;
        if (above > 0) {
            int dest = choose_destination(state, source, above, target + 1, n);
            energy += above + 1;
            move_suffix(state, source, pos, dest);
        }

        // The target is now at the top.
        state[source].pop_back();
    }

    // Estimate residual pain after the finite rollout. Near-future targets
    // buried under many boxes are especially undesirable.
    long long terminal = 0;
    int next = last + 1;
    if (next <= n) {
        for (const auto& st : state) {
            for (int p = 0; p < (int)st.size(); ++p) {
                int x = st[p];
                if (x < next) continue;
                int depth = (int)st.size() - p - 1;
                if (depth == 0) continue;

                int dist = x - next;
                if (dist <= 70) {
                    terminal += 95LL * depth / (dist + 4);
                } else {
                    terminal += depth / 8;
                }
            }
        }
    }

    return energy * 100 + terminal;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    int h = n / m;
    Stacks stacks(m, vector<int>(h));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < h; ++j) cin >> stacks[i][j];
    }

    vector<pair<int, int>> answer;
    answer.reserve(2 * n);

    constexpr int ROLLOUT_HORIZON = 68;

    for (int target = 1; target <= n; ++target) {
        auto [source, pos] = locate_box(stacks, target);
        if (source < 0) return 0;

        int above = (int)stacks[source].size() - pos - 1;

        if (above > 0) {
            int best_dest = -1;
            long long best_value = (1LL << 62);

            for (int dest = 0; dest < m; ++dest) {
                if (dest == source) continue;

                Stacks trial = stacks;
                move_suffix(trial, source, pos, dest);
                trial[source].pop_back();

                long long value = rollout_score(
                    trial,
                    target + 1,
                    n,
                    ROLLOUT_HORIZON
                );

                // Use the immediate label-aware potential as a stable
                // deterministic tie-breaker between equivalent rollouts.
                long long tie = 0;
                for (int x : stacks[dest]) {
                    if (x >= target + 1) {
                        tie += 1000LL * above / (x - target + 7);
                    }
                }
                value = value * 1000 + tie;

                if (value < best_value ||
                    (value == best_value &&
                     (best_dest == -1 ||
                      stacks[dest].size() < stacks[best_dest].size())) ||
                    (value == best_value && best_dest != -1 &&
                     stacks[dest].size() == stacks[best_dest].size() &&
                     dest < best_dest)) {
                    best_value = value;
                    best_dest = dest;
                }
            }

            int first_moved = stacks[source][pos + 1];
            answer.push_back({first_moved, best_dest + 1});
            move_suffix(stacks, source, pos, best_dest);
        }

        answer.push_back({target, 0});
        stacks[source].pop_back();
    }

    for (auto [v, dst] : answer) {
        cout << v << ' ' << dst << '\n';
    }
    return 0;
}
# EVOLVE-BLOCK-END