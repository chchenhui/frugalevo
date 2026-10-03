# EVOLVE-BLOCK-START
#pragma GCC optimize("O3,unroll-loops")

#include <bits/stdc++.h>
using namespace std;

static constexpr int N = 50;
static constexpr int M = 100;
static constexpr int SZ = N * N;

struct XorShift {
    uint32_t x = 123456789u, y = 362436069u, z = 521288629u, w = 88675123u;

    XorShift() {
        uint64_t t = chrono::high_resolution_clock::now().time_since_epoch().count();
        x ^= (uint32_t)t;
        y ^= (uint32_t)(t >> 16);
        z ^= (uint32_t)(t >> 32);
        w ^= (uint32_t)(t >> 48);
    }

    uint32_t next() {
        uint32_t t = x;
        t ^= t << 11;
        t ^= t >> 8;
        x = y; y = z; z = w;
        w ^= w >> 19;
        w ^= t;
        return w;
    }

    int operator()(int n) {
        return (int)(next() % (uint32_t)n);
    }
};

XorShift rng;

int g[SZ];
int bestg[SZ];

bool req[101][101];
int edges[101][101];
int cnt[101];

int seen[SZ];
int stamp = 1;
int qbuf[SZ];

vector<int> frontier;

chrono::steady_clock::time_point start_time;

inline int id(int r, int c) {
    return r * N + c;
}

inline bool inside(int r, int c) {
    return (unsigned)r < N && (unsigned)c < N;
}

inline void add_pair_delta(int a, int b, int delta, int keys[], int vals[], int &ks) {
    if (a == b) return;
    if (a > b) swap(a, b);
    int key = a * 101 + b;
    for (int i = 0; i < ks; ++i) {
        if (keys[i] == key) {
            vals[i] += delta;
            return;
        }
    }
    keys[ks] = key;
    vals[ks] = delta;
    ++ks;
}

inline bool is_frontier_cell(int p) {
    if (g[p] == 0) return false;
    int r = p / N, c = p % N;
    if (r == 0 || r == N - 1 || c == 0 || c == N - 1) return true;
    return g[p - N] == 0 || g[p + N] == 0 || g[p - 1] == 0 || g[p + 1] == 0;
}

// After changing p away from old_color, checks whether the remaining old_color
// cells stay connected. Since the state before the move was connected, it is
// enough to verify connectivity among old-color neighbors of p.
bool remains_connected_after_removal(int p, int old_color) {
    if (cnt[old_color] == 0) return false;

    int r = p / N, c = p % N;
    int targets[4];
    int nt = 0;

    auto add_target = [&](int v) {
        if (g[v] != old_color) return;
        for (int i = 0; i < nt; ++i) {
            if (targets[i] == v) return;
        }
        targets[nt++] = v;
    };

    if (r > 0) add_target(p - N);
    if (r + 1 < N) add_target(p + N);
    if (c > 0) add_target(p - 1);
    if (c + 1 < N) add_target(p + 1);

    if (nt <= 1) return true;

    ++stamp;
    if (stamp == INT_MAX) {
        memset(seen, 0, sizeof(seen));
        stamp = 1;
    }

    int head = 0, tail = 0;
    qbuf[tail++] = targets[0];
    seen[targets[0]] = stamp;

    int reached = 1;

    while (head < tail && reached < nt) {
        int v = qbuf[head++];
        int vr = v / N, vc = v % N;

        auto visit = [&](int u) {
            if (g[u] == old_color && seen[u] != stamp) {
                seen[u] = stamp;
                qbuf[tail++] = u;
                for (int i = 1; i < nt; ++i) {
                    if (u == targets[i]) ++reached;
                }
            }
        };

        if (vr > 0) visit(v - N);
        if (vr + 1 < N) visit(v + N);
        if (vc > 0) visit(v - 1);
        if (vc + 1 < N) visit(v + 1);
    }

    return reached == nt;
}

// Changes p from old color to nc. nc is either zero or a nonzero neighboring
// color. Returns true only if all adjacency and connectivity constraints remain valid.
bool attempt_move(int p, int nc) {
    int old = g[p];
    if (old == nc || old == 0) return false;
    if (cnt[old] <= 1) return false;

    if (nc == 0 && !is_frontier_cell(p)) return false;

    int r = p / N, c = p % N;
    int keys[8], vals[8], ks = 0;

    auto process_neighbor = [&](int other) {
        add_pair_delta(old, other, -1, keys, vals, ks);
        add_pair_delta(nc, other, +1, keys, vals, ks);
    };

    if (r == 0) process_neighbor(0);
    else process_neighbor(g[p - N]);

    if (r == N - 1) process_neighbor(0);
    else process_neighbor(g[p + N]);

    if (c == 0) process_neighbor(0);
    else process_neighbor(g[p - 1]);

    if (c == N - 1) process_neighbor(0);
    else process_neighbor(g[p + 1]);

    for (int i = 0; i < ks; ++i) {
        int a = keys[i] / 101;
        int b = keys[i] % 101;
        int after = edges[a][b] + vals[i];
        if ((after > 0) != req[a][b]) return false;
    }

    g[p] = nc;
    --cnt[old];
    ++cnt[nc];

    bool ok = remains_connected_after_removal(p, old);

    if (!ok) {
        g[p] = old;
        ++cnt[old];
        --cnt[nc];
        return false;
    }

    for (int i = 0; i < ks; ++i) {
        int a = keys[i] / 101;
        int b = keys[i] % 101;
        edges[a][b] += vals[i];
    }

    return true;
}

inline void add_neighbors_to_frontier(int p) {
    int r = p / N, c = p % N;
    if (r > 0 && g[p - N] != 0) frontier.push_back(p - N);
    if (r + 1 < N && g[p + N] != 0) frontier.push_back(p + N);
    if (c > 0 && g[p - 1] != 0) frontier.push_back(p - 1);
    if (c + 1 < N && g[p + 1] != 0) frontier.push_back(p + 1);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    memset(req, 0, sizeof(req));
    memset(edges, 0, sizeof(edges));
    memset(cnt, 0, sizeof(cnt));

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            cin >> g[id(r, c)];
            ++cnt[g[id(r, c)]];
        }
    }

    auto mark_req = [&](int a, int b) {
        if (a == b) return;
        req[a][b] = req[b][a] = true;
    };

    auto add_edge = [&](int a, int b) {
        if (a == b) return;
        ++edges[a][b];
        ++edges[b][a];
    };

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int p = id(r, c);
            int a = g[p];

            if (r == 0) {
                mark_req(a, 0);
                add_edge(a, 0);
            }
            if (r == N - 1) {
                mark_req(a, 0);
                add_edge(a, 0);
            }
            if (c == 0) {
                mark_req(a, 0);
                add_edge(a, 0);
            }
            if (c == N - 1) {
                mark_req(a, 0);
                add_edge(a, 0);
            }

            if (r + 1 < N) {
                int b = g[p + N];
                mark_req(a, b);
                add_edge(a, b);
            }
            if (c + 1 < N) {
                int b = g[p + 1];
                mark_req(a, b);
                add_edge(a, b);
            }
        }
    }

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (r == 0 || r == N - 1 || c == 0 || c == N - 1) {
                frontier.push_back(id(r, c));
            }
        }
    }

    memcpy(bestg, g, sizeof(g));
    int best_zero = 0;

    start_time = chrono::steady_clock::now();
    long long iterations = 0;

    while (true) {
        ++iterations;
        if ((iterations & 4095) == 0) {
            double elapsed = chrono::duration<double, milli>(
                chrono::steady_clock::now() - start_time
            ).count();
            if (elapsed > 1950.0) break;
        }

        if (frontier.empty()) {
            for (int p = 0; p < SZ; ++p) {
                if (is_frontier_cell(p)) frontier.push_back(p);
            }
            if (frontier.empty()) break;
        }

        int p = frontier[rng((int)frontier.size())];
        if (g[p] == 0 || !is_frontier_cell(p)) continue;

        // Mostly perform direct outside expansion.
        // Periodic nonzero recoloring reshapes narrow articulation bottlenecks.
        bool do_recolor = (rng(100) < 13);

        if (!do_recolor) {
            if (attempt_move(p, 0)) {
                ++best_zero;
                if (best_zero > 0) memcpy(bestg, g, sizeof(g));
                add_neighbors_to_frontier(p);
            }
        } else {
            int r = p / N, c = p % N;
            int cand[4];
            int ncand = 0;

            auto add_candidate = [&](int v) {
                int col = g[v];
                if (col == 0 || col == g[p]) return;
                for (int i = 0; i < ncand; ++i) {
                    if (cand[i] == col) return;
                }
                cand[ncand++] = col;
            };

            if (r > 0) add_candidate(p - N);
            if (r + 1 < N) add_candidate(p + N);
            if (c > 0) add_candidate(p - 1);
            if (c + 1 < N) add_candidate(p + 1);

            if (ncand) {
                int nc = cand[rng(ncand)];
                attempt_move(p, nc);
            }
        }
    }

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (c) cout << ' ';
            cout << bestg[id(r, c)];
        }
        cout << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END