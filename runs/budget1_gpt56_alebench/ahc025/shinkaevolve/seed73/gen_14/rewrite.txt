# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

static int N, D, Q, used_queries = 0;
static mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());

static char ask(const vector<int>& L, const vector<int>& R) {
    ++used_queries;
    cout << L.size() << ' ' << R.size();
    for (int x : L) cout << ' ' << x;
    for (int x : R) cout << ' ' << x;
    cout << endl;
    char c;
    cin >> c;
    return c;
}

static int ceil_log2_int(int x) {
    int r = 0;
    while ((1 << r) < x) ++r;
    return r;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> N >> D >> Q;

    int ranking_budget = max(N / 2, Q * 42 / 100);
    ranking_budget = min(ranking_budget, max(0, Q - 2 * D));

    int K = 1;
    for (int k = 2; k <= min(N, 36); ++k) {
        int sorting = 0;
        for (int i = 1; i < k; ++i) sorting += ceil_log2_int(i + 1);
        int classification = (N - k) * ceil_log2_int(k + 1);
        if (sorting + classification <= ranking_budget) K = k;
    }

    vector<int> perm(N);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);

    vector<int> pivots;
    pivots.reserve(K);

    for (int z = 0; z < K; ++z) {
        int item = perm[z];
        int lo = 0, hi = (int)pivots.size();
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            char c = ask(vector<int>{item}, vector<int>{pivots[mid]});
            if (c == '<' || c == '=') hi = mid;
            else lo = mid + 1;
        }
        pivots.insert(pivots.begin() + lo, item);
    }

    vector<char> is_pivot(N, false);
    vector<int> pivot_pos(N, -1), bucket(N, 0);
    for (int i = 0; i < K; ++i) {
        is_pivot[pivots[i]] = true;
        pivot_pos[pivots[i]] = i;
    }

    vector<vector<int>> bin(K + 1);
    for (int item = 0; item < N; ++item) {
        if (is_pivot[item]) continue;
        int lo = 0, hi = K;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            char c = ask(vector<int>{item}, vector<int>{pivots[mid]});
            if (c == '<' || c == '=') hi = mid;
            else lo = mid + 1;
        }
        bucket[item] = lo;
        bin[lo].push_back(item);
    }

    // Infer expected exponential weights from observed ordinal-bin populations.
    vector<double> proxy(N, 1.0);
    int passed = 0;
    for (int b = 0; b <= K; ++b) {
        int cnt = (int)bin[b].size();
        if (cnt) {
            double rank = passed + (cnt - 1) * 0.5;
            double p = (rank + 1.0) / (N + 1.0);
            p = min(0.992, max(0.004, p));
            double val = -log(1.0 - p);
            for (int x : bin[b]) proxy[x] = val;
        }
        passed += cnt;
        if (b < K) {
            double p = (passed + 1.0) / (N + 1.0);
            p = min(0.992, max(0.004, p));
            proxy[pivots[b]] = -log(1.0 - p);
            ++passed;
        }
    }

    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), rng);
    stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return proxy[a] > proxy[b];
    });

    vector<vector<int>> groups(D);
    vector<int> belong(N);
    vector<double> estimated_sum(D, 0.0);

    for (int item : order) {
        int g = 0;
        for (int j = 1; j < D; ++j)
            if (estimated_sum[j] < estimated_sum[g]) g = j;
        groups[g].push_back(item);
        belong[item] = g;
        estimated_sum[g] += proxy[item];
    }

    auto erase_item = [&](int g, int item) {
        vector<int>& v = groups[g];
        for (int i = 0; i < (int)v.size(); ++i) {
            if (v[i] == item) {
                v[i] = v.back();
                v.pop_back();
                return;
            }
        }
    };

    auto commit_move = [&](int item, int from, int to) {
        erase_item(from, item);
        groups[to].push_back(item);
        belong[item] = to;
        estimated_sum[from] -= proxy[item];
        estimated_sum[to] += proxy[item];
    };

    int iteration = 0;
    while (used_queries < Q) {
        ++iteration;
        int a, b;

        if (iteration % 3 == 0) {
            a = 0;
            b = 0;
            for (int g = 1; g < D; ++g) {
                if (estimated_sum[g] > estimated_sum[a]) a = g;
                if (estimated_sum[g] < estimated_sum[b]) b = g;
            }
            if (a == b) {
                a = rng() % D;
                b = (a + 1 + rng() % (D - 1)) % D;
            }
        } else {
            a = rng() % D;
            b = rng() % D;
            while (a == b) b = rng() % D;
        }

        if (groups[a].empty() || groups[b].empty()) continue;

        char initial = ask(groups[a], groups[b]);
        if (initial == '=') {
            double mid = (estimated_sum[a] + estimated_sum[b]) * 0.5;
            estimated_sum[a] = estimated_sum[b] = mid;
            continue;
        }

        int heavy = initial == '>' ? a : b;
        int light = initial == '>' ? b : a;
        if ((int)groups[heavy].size() <= 1) continue;

        vector<int> cand = groups[heavy];
        sort(cand.begin(), cand.end(), [&](int x, int y) {
            return proxy[x] < proxy[y];
        });

        int lo = 0, hi = (int)cand.size() - 1;
        int best_safe = -1;
        bool equal_found = false;
        int equal_item = -1;

        while (lo <= hi && used_queries < Q) {
            int mid = (lo + hi) >> 1;
            int item = cand[mid];

            vector<int> left = groups[heavy];
            for (int i = 0; i < (int)left.size(); ++i) {
                if (left[i] == item) {
                    left[i] = left.back();
                    left.pop_back();
                    break;
                }
            }
            if (left.empty()) break;

            vector<int> right = groups[light];
            right.push_back(item);
            char after = ask(left, right);

            if (after == '=') {
                equal_found = true;
                equal_item = item;
                break;
            }
            if (after == '>') {
                best_safe = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        if (equal_found) {
            commit_move(equal_item, heavy, light);
        } else if (best_safe != -1) {
            commit_move(cand[best_safe], heavy, light);
        } else {
            // Estimated sums are adjusted slightly after an overshoot-only search.
            double mid = (estimated_sum[heavy] + estimated_sum[light]) * 0.5;
            estimated_sum[heavy] = mid + 0.05;
            estimated_sum[light] = mid - 0.05;
        }
    }

    while (used_queries < Q) ask(vector<int>{0}, vector<int>{1});

    for (int i = 0; i < N; ++i) {
        if (i) cout << ' ';
        cout << belong[i];
    }
    cout << endl;
    return 0;
}
# EVOLVE-BLOCK-END