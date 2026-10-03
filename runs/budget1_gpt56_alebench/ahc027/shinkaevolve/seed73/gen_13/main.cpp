# EVOLVE-BLOCK-START
#pragma GCC optimize("O3,unroll-loops")

#include <bits/stdc++.h>
using namespace std;

static constexpr int MAXN = 40;
static constexpr int MAXV = 1600;
static constexpr int MAXL = 100000;

int N, V;
int dirt[MAXV];
int nxtv[MAXV][4];
int dista[MAXV][MAXV];
int parentv[MAXV][MAXV];
char dch[4] = {'R','D','L','U'};
char invch[256];
int dirch[256];

mt19937 rng((uint32_t)chrono::high_resolution_clock::now().time_since_epoch().count());

struct PathData {
    vector<char> moves;
    vector<int> pos;
    double score = 1e100;
    double term[MAXV];
};

vector<int> visits[MAXV];
vector<int> tmp_indices;
vector<pair<double,int>> ranking;
vector<char> buf1, buf2;

inline int rndint(int l, int r) {
    return uniform_int_distribution<int>(l, r)(rng);
}
inline double rnd01() {
    return uniform_real_distribution<double>(0.0, 1.0)(rng);
}

void build_graph(const vector<string>& h, const vector<string>& vwall) {
    V = N * N;
    for (int id = 0; id < V; ++id)
        for (int d = 0; d < 4; ++d) nxtv[id][d] = -1;

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int x = r * N + c;
            if (c + 1 < N && vwall[r][c] == '0') {
                nxtv[x][0] = x + 1;
                nxtv[x + 1][2] = x;
            }
            if (r + 1 < N && h[r][c] == '0') {
                nxtv[x][1] = x + N;
                nxtv[x + N][3] = x;
            }
        }
    }
}

void compute_apsp() {
    static int q[MAXV];
    for (int s = 0; s < V; ++s) {
        fill(dista[s], dista[s] + V, -1);
        int head = 0, tail = 0;
        q[tail++] = s;
        dista[s][s] = 0;
        while (head < tail) {
            int x = q[head++];
            for (int d = 0; d < 4; ++d) {
                int y = nxtv[x][d];
                if (y >= 0 && dista[s][y] < 0) {
                    dista[s][y] = dista[s][x] + 1;
                    parentv[s][y] = x;
                    q[tail++] = y;
                }
            }
        }
    }
}

inline char move_between(int a, int b) {
    for (int d = 0; d < 4; ++d) if (nxtv[a][d] == b) return dch[d];
    return '?';
}

bool get_path(int from, int to, vector<char>& out) {
    out.clear();
    if (from == to) return true;
    if (dista[from][to] < 0) return false;
    int x = to;
    while (x != from) {
        int p = parentv[from][x];
        out.push_back(move_between(p, x));
        x = p;
    }
    reverse(out.begin(), out.end());
    return true;
}

bool evaluate(PathData& p) {
    const int L = (int)p.moves.size();
    if (L <= 0 || L > MAXL) return false;

    p.pos.clear();
    p.pos.reserve(L + 1);
    p.pos.push_back(0);

    static bool seen[MAXV];
    fill(seen, seen + V, false);
    seen[0] = true;

    int cur = 0;
    for (char ch : p.moves) {
        int d = dirch[(unsigned char)ch];
        if (d < 0) return false;
        int to = nxtv[cur][d];
        if (to < 0) return false;
        cur = to;
        p.pos.push_back(cur);
        seen[cur] = true;
    }
    if (cur != 0) return false;
    for (int i = 0; i < V; ++i) if (!seen[i]) return false;

    for (int i = 0; i < V; ++i) visits[i].clear();
    for (int t = 1; t <= L; ++t) visits[p.pos[t]].push_back(t);

    double sum = 0.0;
    for (int x = 0; x < V; ++x) {
        double val = 0.0;
        const auto& a = visits[x];
        int m = (int)a.size();
        for (int i = 0; i < m; ++i) {
            int prv = (i == 0 ? a.back() - L : a[i - 1]);
            double gap = (double)(a[i] - prv);
            val += gap * (gap - 1.0) * 0.5;
        }
        p.term[x] = val;
        sum += val * dirt[x];
    }
    p.score = sum / L;
    return true;
}

void dfs_initial(int x, vector<char>& ans, vector<char>& used) {
    used[x] = 1;
    for (int d = 0; d < 4; ++d) {
        int y = nxtv[x][d];
        if (y >= 0 && !used[y]) {
            ans.push_back(dch[d]);
            dfs_initial(y, ans, used);
            ans.push_back(dch[(d + 2) & 3]);
        }
    }
}

int select_target(const PathData& p, bool narrow) {
    ranking.clear();
    ranking.reserve(V);
    for (int x = 0; x < V; ++x)
        ranking.push_back({p.term[x] * dirt[x], x});
    int k = narrow ? max(1, N) : max(10, V / 10);
    k = min(k, V);
    nth_element(ranking.begin(), ranking.begin() + k, ranking.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    return ranking[rndint(0, k - 1)].second;
}

inline void insert_segment(vector<char>& a, int at, const vector<char>& b) {
    a.insert(a.begin() + at, b.begin(), b.end());
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    double time_limit = 1.93;
    if (argc > 1) time_limit = stod(argv[1]);
    auto start = chrono::high_resolution_clock::now();

    cin >> N;
    vector<string> h(N - 1), vw(N);
    for (int i = 0; i < N - 1; ++i) cin >> h[i];
    for (int i = 0; i < N; ++i) cin >> vw[i];
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            cin >> dirt[r * N + c];

    fill(dirch, dirch + 256, -1);
    dirch['R'] = 0; dirch['D'] = 1; dirch['L'] = 2; dirch['U'] = 3;
    invch['R'] = 'L'; invch['L'] = 'R';
    invch['D'] = 'U'; invch['U'] = 'D';

    build_graph(h, vw);
    compute_apsp();

    for (int i = 0; i < V; ++i) visits[i].reserve(128);
    buf1.reserve(128);
    buf2.reserve(128);

    PathData current;
    vector<char> used(V, 0);
    dfs_initial(0, current.moves, used);
    evaluate(current);
    PathData best = current;

    const double start_temp = 5000.0 * sqrt((double)N);
    const double end_temp = 0.1;
    int iter = 0;

    while (true) {
        if ((++iter & 63) == 0) {
            double elapsed = chrono::duration<double>(
                chrono::high_resolution_clock::now() - start).count();
            if (elapsed >= time_limit) break;
        }

        int L = (int)current.moves.size();
        PathData cand;
        bool changed = false;

        for (int tries = 0; tries < 8 && !changed; ++tries) {
            int roll = rndint(0, 99);
            int op = roll < 15 ? 0 : roll < 30 ? 1 : roll < 60 ? 2 :
                     roll < 70 ? 3 : roll < 85 ? 4 : 5;
            cand = current;

            if (op == 0) {
                if (L + 2 > MAXL) continue;
                int at = rndint(0, L);
                int x = current.pos[at];
                int ds[4], cnt = 0;
                for (int d = 0; d < 4; ++d) if (nxtv[x][d] >= 0) ds[cnt++] = d;
                if (!cnt) continue;
                int d = ds[rndint(0, cnt - 1)];
                cand.moves.insert(cand.moves.begin() + at, {dch[d], dch[(d + 2) & 3]});
                changed = true;
            } else if (op == 1) {
                if (L < 2) continue;
                tmp_indices.clear();
                for (int i = 0; i + 2 <= L; ++i)
                    if (current.pos[i] == current.pos[i + 2]) tmp_indices.push_back(i);
                if (tmp_indices.empty()) continue;
                int at = tmp_indices[rndint(0, (int)tmp_indices.size() - 1)];
                cand.moves.erase(cand.moves.begin() + at, cand.moves.begin() + at + 2);
                changed = true;
            } else if (op == 2) {
                if (L < 2) continue;
                int a = rndint(0, L - 1), b = rndint(a + 1, L);
                if (!get_path(current.pos[a], current.pos[b], buf1)) continue;
                if ((int)buf1.size() >= b - a) continue;
                cand.moves.erase(cand.moves.begin() + a, cand.moves.begin() + b);
                insert_segment(cand.moves, a, buf1);
                changed = true;
            } else if (op == 3) {
                if (L < 2) continue;
                int a = rndint(0, L - 1), b = rndint(a, L - 1);
                reverse(cand.moves.begin() + a, cand.moves.begin() + b + 1);
                for (int i = a; i <= b; ++i) cand.moves[i] = invch[(unsigned char)cand.moves[i]];
                changed = true;
            } else if (op == 4) {
                int target = select_target(current, false);

                int midpoint = rndint(0, L);
                const auto& vv = visits[target];
                if (!vv.empty()) {
                    int bestgap = -1, pa = 0, pb = 0;
                    for (int i = 0; i < (int)vv.size(); ++i) {
                        int x = (i == 0 ? vv.back() - L : vv[i - 1]);
                        int y = vv[i];
                        if (y - x > bestgap) bestgap = y - x, pa = x, pb = y;
                    }
                    midpoint = (pa + pb) / 2;
                    if (midpoint < 0) midpoint += L;
                    if (midpoint > L) midpoint -= L;
                }

                int at = midpoint;
                int bestcost = (current.pos[at] == target ? 2 : 2 * dista[current.pos[at]][target]);
                for (int z = 0; z < 20; ++z) {
                    int q = rndint(0, L);
                    int cost = current.pos[q] == target ? 2 : 2 * dista[current.pos[q]][target];
                    if (cost < bestcost) bestcost = cost, at = q;
                }
                if (bestcost < 0 || L + bestcost > MAXL) continue;

                int x = current.pos[at];
                if (x == target) {
                    int ds[4], cnt = 0;
                    for (int d = 0; d < 4; ++d) if (nxtv[x][d] >= 0) ds[cnt++] = d;
                    if (!cnt) continue;
                    int d = ds[rndint(0, cnt - 1)];
                    cand.moves.insert(cand.moves.begin() + at, {dch[d], dch[(d + 2) & 3]});
                } else {
                    get_path(x, target, buf1);
                    get_path(target, x, buf2);
                    insert_segment(cand.moves, at, buf2);
                    insert_segment(cand.moves, at, buf1);
                }
                changed = true;
            } else {
                int target = select_target(current, true);
                int a = rndint(0, L - 1);
                int b = rndint(a + 1, min(L, a + 20));
                if (!get_path(current.pos[a], target, buf1)) continue;
                if (!get_path(target, current.pos[b], buf2)) continue;
                int nl = L - (b - a) + (int)buf1.size() + (int)buf2.size();
                if (nl > MAXL) continue;
                cand.moves.erase(cand.moves.begin() + a, cand.moves.begin() + b);
                insert_segment(cand.moves, a, buf2);
                insert_segment(cand.moves, a, buf1);
                changed = true;
            }
        }

        if (!changed || !evaluate(cand)) continue;

        if (cand.score < current.score) {
            current = move(cand);
            if (current.score < best.score) best = current;
        } else {
            double elapsed = chrono::duration<double>(
                chrono::high_resolution_clock::now() - start).count();
            double p = min(1.0, max(0.0, elapsed / time_limit));
            double temp = start_temp * pow(end_temp / start_temp, p);
            double delta = current.score - cand.score;
            if (exp(min(0.0, delta / max(temp, 1e-9))) > rnd01())
                current = move(cand);
        }
    }

    for (char c : best.moves) cout << c;
    cout << '\n';
    return 0;
}
# EVOLVE-BLOCK-END