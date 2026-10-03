# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

static int N, D, Q, used_queries = 0;
static mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());
static chrono::steady_clock::time_point start_time;
static const int TIME_LIMIT_MS = 1850;
static const long long BASE = 100000;

static char cmp_cache[105][105];
static char sum_cache[105][105][105];

static inline char revcmp(char c) {
    return c == '<' ? '>' : c == '>' ? '<' : '=';
}

static char query(const vector<int>& L, const vector<int>& R) {
    ++used_queries;
    cout << L.size() << ' ' << R.size();
    for (int x : L) cout << ' ' << x;
    for (int x : R) cout << ' ' << x;
    cout << endl;
    char r;
    cin >> r;
    return r;
}

static char compare_one(int a, int b) {
    if (a == b) return '=';
    if (cmp_cache[a][b]) return cmp_cache[a][b];
    if (used_queries >= Q) return '=';
    char r = query(vector<int>{a}, vector<int>{b});
    cmp_cache[a][b] = r;
    cmp_cache[b][a] = revcmp(r);
    return r;
}

static char compare_one_two(int a, int b, int c) {
    int x = min(b, c), y = max(b, c);
    if (sum_cache[a][x][y]) return sum_cache[a][x][y];
    if (used_queries >= Q) return '=';
    char r = query(vector<int>{a}, vector<int>{b, c});
    sum_cache[a][x][y] = r;
    return r;
}

static void merge_sort_items(vector<int>& v, vector<int>& tmp, int l, int r) {
    if (r - l <= 1) return;
    int m = (l + r) >> 1;
    merge_sort_items(v, tmp, l, m);
    merge_sort_items(v, tmp, m, r);
    int i = l, j = m, k = l;
    while (i < m && j < r) {
        char c = compare_one(v[i], v[j]);
        if (c == '<' || c == '=') tmp[k++] = v[i++];
        else tmp[k++] = v[j++];
    }
    while (i < m) tmp[k++] = v[i++];
    while (j < r) tmp[k++] = v[j++];
    for (i = l; i < r; ++i) v[i] = tmp[i];
}

static int estimate_cost(int n, int k) {
    if (k <= 1) return n - 1;
    double lgk = log2((double)k);
    double z = k * lgk + (n - k) * lgk;
    for (int i = 1; i < k - 1; ++i) z += log2((double)i);
    return (int)ceil(z);
}

static inline long double sq(long long x) {
    return (long double)x * x;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    start_time = chrono::steady_clock::now();

    cin >> N >> D >> Q;
    vector<long long> w(N, BASE);

    int K = 1;
    for (int k = N; k >= 1; --k) {
        if (estimate_cost(N, k) <= Q) {
            K = k;
            break;
        }
    }

    vector<int> ids(N);
    iota(ids.begin(), ids.end(), 0);
    shuffle(ids.begin(), ids.end(), rng);
    vector<int> piv(ids.begin(), ids.begin() + K), tmp(K);
    merge_sort_items(piv, tmp, 0, K);

    vector<char> is_pivot(N, false);
    for (int x : piv) is_pivot[x] = true;

    w[piv[0]] = BASE;
    if (K >= 2) {
        char c = compare_one(piv[1], piv[0]);
        w[piv[1]] = (c == '=') ? w[piv[0]] : max(1LL, w[piv[0]] * 2);
    }

    const long long max_delta = BASE * (N / max(1, D) + 10);

    for (int j = 2; j < K; ++j) {
        int cur = piv[j], prev = piv[j - 1];
        char adjacent = compare_one(cur, prev);
        if (adjacent == '=') {
            w[cur] = w[prev];
            continue;
        }

        long long lo_val = 1, hi_val = max_delta;
        bool got_lo = false, got_hi = false;
        int lo = 0, hi = j - 2;

        while (lo <= hi && used_queries < Q) {
            int m = (lo + hi) >> 1;
            char c = compare_one_two(cur, prev, piv[m]);
            if (c == '=') {
                lo_val = hi_val = w[piv[m]];
                got_lo = got_hi = true;
                break;
            } else if (c == '>') {
                lo_val = w[piv[m]];
                got_lo = true;
                lo = m + 1;
            } else {
                hi_val = w[piv[m]];
                got_hi = true;
                hi = m - 1;
            }
        }

        long long delta;
        if (got_lo && got_hi) delta = (lo_val + hi_val) / 2;
        else if (got_lo) delta = max(1LL, lo_val * 2);
        else if (got_hi) delta = max(1LL, hi_val / 2);
        else delta = max(1LL, w[prev]);
        w[cur] = w[prev] + delta;
    }

    for (int item = 0; item < N; ++item) {
        if (is_pivot[item]) continue;

        int lo = 0, hi = K - 1, equal_pos = -1;
        while (lo <= hi && used_queries < Q) {
            int m = (lo + hi) >> 1;
            char c = compare_one(item, piv[m]);
            if (c == '=') {
                equal_pos = m;
                break;
            }
            if (c == '<') hi = m - 1;
            else lo = m + 1;
        }

        if (equal_pos >= 0) {
            w[item] = w[piv[equal_pos]];
        } else if (lo == 0) {
            long long a = w[piv[0]];
            long long b = K >= 2 ? w[piv[1]] : a * 2;
            w[item] = max(1LL, b > a ? a * a / b : a / 2);
        } else if (lo == K) {
            long long a = w[piv[K - 1]];
            long long b = K >= 2 ? w[piv[K - 2]] : max(1LL, a / 2);
            w[item] = max(1LL, a > b ? a * a / b : a * 2);
        } else {
            long long a = w[piv[lo - 1]], b = w[piv[lo]];
            w[item] = max(a, min(b, (long long)sqrt((long double)a * b)));
        }
    }

    while (used_queries < Q) {
        int b = (used_queries % (N - 1)) + 1;
        query(vector<int>{0}, vector<int>{b});
    }

    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        return w[a] != w[b] ? w[a] > w[b] : a < b;
    });

    vector<int> group(N);
    vector<long long> sum(D, 0);
    for (int x : order) {
        int g = min_element(sum.begin(), sum.end()) - sum.begin();
        group[x] = g;
        sum[g] += w[x];
    }

    auto energy = [&]() {
        long double v = 0;
        for (long long x : sum) v += sq(x);
        return v;
    };

    long double current = energy();

    // Deterministic local improvement using estimated weights.
    for (int rounds = 0; rounds < 200; ++rounds) {
        long double best_delta = 0;
        int best_type = -1, bi = -1, bj = -1;

        for (int x = 0; x < N; ++x) {
            int a = group[x];
            for (int b = 0; b < D; ++b) if (a != b) {
                long long na = sum[a] - w[x], nb = sum[b] + w[x];
                long double delta = sq(na) + sq(nb) - sq(sum[a]) - sq(sum[b]);
                if (delta < best_delta) {
                    best_delta = delta;
                    best_type = 0;
                    bi = x;
                    bj = b;
                }
            }
        }

        for (int x = 0; x < N; ++x) for (int y = x + 1; y < N; ++y) {
            int a = group[x], b = group[y];
            if (a == b) continue;
            long long na = sum[a] - w[x] + w[y];
            long long nb = sum[b] - w[y] + w[x];
            long double delta = sq(na) + sq(nb) - sq(sum[a]) - sq(sum[b]);
            if (delta < best_delta) {
                best_delta = delta;
                best_type = 1;
                bi = x;
                bj = y;
            }
        }

        if (best_type < 0) break;
        if (best_type == 0) {
            int a = group[bi], b = bj;
            sum[a] -= w[bi];
            sum[b] += w[bi];
            group[bi] = b;
        } else {
            int a = group[bi], b = group[bj];
            sum[a] += w[bj] - w[bi];
            sum[b] += w[bi] - w[bj];
            swap(group[bi], group[bj]);
        }
        current += best_delta;
    }

    vector<int> best_group = group;
    long double best_energy = current;
    uniform_real_distribution<double> real01(0.0, 1.0);

    long double temp = max((long double)1.0, current / max(1, D) * 0.03);
    long long iter = 0;

    while (true) {
        if ((++iter & 1023) == 0) {
            auto now = chrono::steady_clock::now();
            if (chrono::duration_cast<chrono::milliseconds>(now - start_time).count() >= TIME_LIMIT_MS) break;
            temp *= 0.9995L;
            if (temp < 1e-6L) temp = 1e-6L;
        }

        bool do_swap = (rng() % 3 == 0);
        long double delta;
        if (!do_swap) {
            int x = rng() % N;
            int a = group[x], b = rng() % D;
            if (a == b) continue;
            long long na = sum[a] - w[x], nb = sum[b] + w[x];
            delta = sq(na) + sq(nb) - sq(sum[a]) - sq(sum[b]);
            bool accept = delta <= 0 || real01(rng) < exp((double)(-delta / temp));
            if (!accept) continue;
            sum[a] = na;
            sum[b] = nb;
            group[x] = b;
        } else {
            int x = rng() % N, y = rng() % N;
            if (x == y || group[x] == group[y]) continue;
            int a = group[x], b = group[y];
            long long na = sum[a] - w[x] + w[y];
            long long nb = sum[b] - w[y] + w[x];
            delta = sq(na) + sq(nb) - sq(sum[a]) - sq(sum[b]);
            bool accept = delta <= 0 || real01(rng) < exp((double)(-delta / temp));
            if (!accept) continue;
            sum[a] = na;
            sum[b] = nb;
            swap(group[x], group[y]);
        }

        current += delta;
        if (current < best_energy) {
            best_energy = current;
            best_group = group;
        }
    }

    for (int i = 0; i < N; ++i) {
        if (i) cout << ' ';
        cout << best_group[i];
    }
    cout << endl;
    return 0;
}
# EVOLVE-BLOCK-END