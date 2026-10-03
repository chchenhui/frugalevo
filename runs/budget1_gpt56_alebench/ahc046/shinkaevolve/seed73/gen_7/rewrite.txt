# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

static constexpr int N = 20;
static constexpr int SEG = 39;
static constexpr int INF = 1e9;
static constexpr int LIM = 1600;
static constexpr int BASE = 10;
static constexpr int POST = 17;
static constexpr int STRAT = BASE * POST;

struct Pos {
    int r, c;
    bool operator==(const Pos& o) const { return r == o.r && c == o.c; }
    bool operator!=(const Pos& o) const { return !(*this == o); }
};

struct Grid {
    array<unsigned, N> row{};
    bool blocked(int r, int c) const {
        return r < 0 || r >= N || c < 0 || c >= N || ((row[r] >> c) & 1U);
    }
    void toggle(int r, int c) { row[r] ^= (1U << c); }
};

static const int dr[4] = {-1, 1, 0, 0};
static const int dc[4] = {0, 0, -1, 1};
static const int revd[4] = {1, 0, 3, 2};
static const char dch[4] = {'U', 'D', 'L', 'R'};

mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count());
auto started = chrono::steady_clock::now();

double elapsed_ms() {
    return chrono::duration<double, milli>(chrono::steady_clock::now() - started).count();
}

struct Parent {
    short pr, pc;
    char act, dir;
};

struct SearchResult {
    int dist = INF;
    string path;
};

SearchResult bfs(Pos st, Pos goal, const Grid& g, Pos forbidden, bool avoid_forbidden, bool build) {
    static int seen[N][N], dist[N][N], gen = 0;
    static Parent par[N][N];
    ++gen;

    if (g.blocked(st.r, st.c)) return {};
    if (avoid_forbidden && st == forbidden && st != goal) return {};

    Pos q[400];
    int head = 0, tail = 0;
    q[tail++] = st;
    seen[st.r][st.c] = gen;
    dist[st.r][st.c] = 0;

    while (head < tail) {
        Pos v = q[head++];
        int nd = dist[v.r][v.c] + 1;
        if (v == goal) break;

        for (int d = 0; d < 4; ++d) {
            Pos u{v.r + dr[d], v.c + dc[d]};
            if (g.blocked(u.r, u.c)) continue;
            if (avoid_forbidden && u == forbidden && u != goal) continue;
            if (seen[u.r][u.c] == gen) continue;
            seen[u.r][u.c] = gen;
            dist[u.r][u.c] = nd;
            if (build) par[u.r][u.c] = {(short)v.r, (short)v.c, 'M', dch[d]};
            q[tail++] = u;
        }

        for (int d = 0; d < 4; ++d) {
            Pos u = v;
            while (!g.blocked(u.r + dr[d], u.c + dc[d])) {
                u.r += dr[d];
                u.c += dc[d];
            }
            if (u == v) continue;
            // Passing over the forbidden target is legal.  Landing on it is not.
            if (avoid_forbidden && u == forbidden && u != goal) continue;
            if (seen[u.r][u.c] == gen) continue;
            seen[u.r][u.c] = gen;
            dist[u.r][u.c] = nd;
            if (build) par[u.r][u.c] = {(short)v.r, (short)v.c, 'S', dch[d]};
            q[tail++] = u;
        }
    }

    if (seen[goal.r][goal.c] != gen) return {};
    SearchResult ret;
    ret.dist = dist[goal.r][goal.c];
    if (!build) return ret;

    string rev;
    Pos p = goal;
    while (p != st) {
        Parent x = par[p.r][p.c];
        rev += x.dir;
        rev += x.act;
        p = {x.pr, x.pc};
    }
    reverse(rev.begin(), rev.end());
    ret.path = move(rev);
    return ret;
}

struct SegResult {
    int turns = INF;
    string actions;
};

bool apply_base(int code, Pos cur, Pos tar, Grid& g, SegResult& out, bool build) {
    if (code == 0) {
        if (g.blocked(tar.r, tar.c)) return false;
        auto z = bfs(cur, tar, g, {-1, -1}, false, build);
        if (z.dist == INF) return false;
        out.turns = z.dist;
        if (build) out.actions = move(z.path);
        return true;
    }

    if (code == 1) {
        if (!g.blocked(tar.r, tar.c)) return false;
        int best = INF, bestd = -1;
        SearchResult bestpath;
        for (int d = 0; d < 4; ++d) {
            Pos a{tar.r + dr[d], tar.c + dc[d]};
            if (g.blocked(a.r, a.c)) continue;
            auto z = bfs(cur, a, g, tar, true, build);
            if (z.dist < best) best = z.dist, bestd = d, bestpath = move(z);
        }
        if (bestd < 0) return false;
        out.turns = best + 2;
        if (build) {
            out.actions = move(bestpath.path);
            char x = dch[revd[bestd]];
            out.actions += 'A'; out.actions += x;
            out.actions += 'M'; out.actions += x;
        }
        g.toggle(tar.r, tar.c);
        return true;
    }

    int type = (code >= 6);
    int d = code - (type ? 6 : 2);
    if (g.blocked(tar.r, tar.c)) return false;

    Pos start{tar.r - dr[d], tar.c - dc[d]};
    Pos stopper{tar.r + dr[d], tar.c + dc[d]};
    if (start.r < 0 || start.r >= N || start.c < 0 || start.c >= N) return false;

    if (!type) {
        if (!g.blocked(stopper.r, stopper.c)) return false;
        auto z = bfs(cur, start, g, tar, true, build);
        if (z.dist == INF) return false;
        out.turns = z.dist + 1;
        if (build) {
            out.actions = move(z.path);
            out.actions += 'S'; out.actions += dch[d];
        }
        return true;
    }

    if (stopper.r < 0 || stopper.r >= N || stopper.c < 0 || stopper.c >= N || g.blocked(stopper.r, stopper.c))
        return false;

    auto p1 = bfs(cur, tar, g, {-1, -1}, false, build);
    if (p1.dist == INF) return false;
    Grid ng = g;
    ng.toggle(stopper.r, stopper.c);

    // We are already at the current target after p1, so it may be crossed later.
    auto p2 = bfs(tar, start, ng, {-1, -1}, false, build);
    if (p2.dist == INF) return false;

    out.turns = p1.dist + p2.dist + 2;
    if (build) {
        out.actions = move(p1.path);
        out.actions += 'A'; out.actions += dch[d];
        out.actions += p2.path;
        out.actions += 'S'; out.actions += dch[d];
    }
    g = ng;
    return true;
}

bool apply_strategy(int code, Pos& player, Pos target, Grid& g, SegResult& out, bool build) {
    out = {};
    out.turns = 0;
    Grid old = g;
    int base = code % BASE;
    int post = code / BASE;

    SegResult b;
    if (!apply_base(base, player, target, g, b, build)) {
        g = old;
        return false;
    }
    out = move(b);
    Pos p = target;

    if (post == 0) {
    } else if (post <= 4) {
        int d = post - 1;
        Pos x{p.r + dr[d], p.c + dc[d]};
        if (x.r < 0 || x.r >= N || x.c < 0 || x.c >= N) {
            g = old;
            return false;
        }
        ++out.turns;
        if (build) { out.actions += 'A'; out.actions += dch[d]; }
        g.toggle(x.r, x.c);
    } else {
        int z = post - 5;
        int dm = z / 3, kth = z % 3;
        int da = -1, cnt = 0;
        for (int d = 0; d < 4; ++d) if (d != revd[dm]) {
            if (cnt++ == kth) { da = d; break; }
        }

        Pos np{p.r + dr[dm], p.c + dc[dm]};
        if (g.blocked(np.r, np.c)) {
            g = old;
            return false;
        }
        Pos x{np.r + dr[da], np.c + dc[da]};
        if (x.r < 0 || x.r >= N || x.c < 0 || x.c >= N) {
            g = old;
            return false;
        }

        out.turns += 2;
        if (build) {
            out.actions += 'M'; out.actions += dch[dm];
            out.actions += 'A'; out.actions += dch[da];
        }
        g.toggle(x.r, x.c);
        p = np;
    }

    player = p;
    return true;
}

struct Cache {
    Grid grid;
    Pos player;
    int turns = INF;
};

struct Eval {
    int turns = INF;
    bool ok = false;
    string actions;
};

Pos initial_pos;
array<Pos, SEG> target;

Eval evaluate(const array<int, SEG>& choice, bool build, int from,
              const array<Cache, SEG>* reference, array<Cache, SEG>* output) {
    Grid g;
    Pos p;
    int total;

    if (from > 0 && reference && (*reference)[from].turns < INF) {
        g = (*reference)[from].grid;
        p = (*reference)[from].player;
        total = (*reference)[from].turns;
        if (output) for (int i = 0; i < from; ++i) (*output)[i] = (*reference)[i];
    } else {
        from = 0;
        g = Grid{};
        p = initial_pos;
        total = 0;
    }

    string acts;
    for (int i = from; i < SEG; ++i) {
        if (output) (*output)[i] = {g, p, total};
        SegResult r;
        if (!apply_strategy(choice[i], p, target[i], g, r, build) || total + r.turns > LIM) {
            if (output) for (int j = i; j < SEG; ++j) (*output)[j].turns = INF;
            return {};
        }
        total += r.turns;
        if (build) acts += r.actions;
    }
    return {total, true, move(acts)};
}

int rnd(int l, int r) {
    return uniform_int_distribution<int>(l, r)(rng);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int ni, mi;
    cin >> ni >> mi;
    cin >> initial_pos.r >> initial_pos.c;
    for (auto& x : target) cin >> x.r >> x.c;

    array<int, SEG> cur{}, best{}, nxt{};
    array<Cache, SEG> cache{}, next_cache{};

    Grid g;
    Pos p = initial_pos;
    int greedy_total = 0;

    for (int k = 0; k < SEG; ++k) {
        int bestcode = -1, bestcost = INF;
        vector<pair<int,int>> near;

        for (int code = 0; code < STRAT; ++code) {
            Grid tg = g;
            Pos tp = p;
            SegResult r;
            if (apply_strategy(code, tp, target[k], tg, r, false)) {
                if (r.turns < bestcost) bestcost = r.turns, bestcode = code;
                if (r.turns <= bestcost + 1) near.push_back({code, r.turns});
            }
        }

        if (k + 1 < SEG && bestcode >= 0) {
            shuffle(near.begin(), near.end(), rng);
            if ((int)near.size() > 12) near.resize(12);
            int two_best = INF;
            for (auto [code, cost] : near) {
                Grid tg = g;
                Pos tp = p;
                SegResult r;
                if (!apply_strategy(code, tp, target[k], tg, r, false)) continue;
                int nc = INF;
                for (int b = 0; b < BASE; ++b) {
                    Grid gg = tg;
                    Pos pp = tp;
                    SegResult rr;
                    if (apply_strategy(b, pp, target[k + 1], gg, rr, false))
                        nc = min(nc, rr.turns);
                }
                if (nc < INF && cost + nc < two_best) {
                    two_best = cost + nc;
                    bestcode = code;
                    bestcost = cost;
                }
            }
        }

        cur[k] = bestcode < 0 ? 0 : bestcode;
        SegResult r;
        if (!apply_strategy(cur[k], p, target[k], g, r, false)) {
            cur[k] = 0;
            apply_strategy(0, p, target[k], g, r, false);
        }
        greedy_total += r.turns;
    }

    auto e = evaluate(cur, false, 0, nullptr, &cache);
    if (!e.ok) return 0;

    int curcost = e.turns;
    int bestcost = curcost;
    best = cur;

    while (elapsed_ms() < 1940.0) {
        nxt = cur;
        int first = SEG;
        double roll = uniform_real_distribution<double>(0, 1)(rng);

        if (roll < 0.55) {
            int changes = (roll < 0.35 ? 1 : (roll < 0.48 ? 2 : 3));
            array<int, SEG> ids;
            iota(ids.begin(), ids.end(), 0);
            shuffle(ids.begin(), ids.end(), rng);
            for (int z = 0; z < changes; ++z) {
                int k = ids[z];
                first = min(first, k);
                int x;
                do x = rnd(0, STRAT - 1); while (x == nxt[k]);
                nxt[k] = x;
            }
        } else if (roll < 0.85) {
            int k = rnd(0, SEG - 1);
            first = k;
            int base = nxt[k] % BASE, post = nxt[k] / BASE;
            if (rnd(0, 1)) {
                int np;
                do np = rnd(0, POST - 1); while (np == post);
                nxt[k] = np * BASE + base;
            } else {
                int nb;
                do nb = rnd(0, BASE - 1); while (nb == base);
                nxt[k] = post * BASE + nb;
            }
        } else {
            int k = rnd(0, SEG - 1);
            first = k;
            int chosen = nxt[k], localbest = INF;
            for (int z = 0; z < 45; ++z) {
                int code = (z < BASE ? z : rnd(0, STRAT - 1));
                Grid tg = cache[k].grid;
                Pos tp = cache[k].player;
                SegResult r;
                if (apply_strategy(code, tp, target[k], tg, r, false) && r.turns < localbest) {
                    localbest = r.turns;
                    chosen = code;
                }
            }
            nxt[k] = chosen;
            if (nxt[k] == cur[k]) continue;
        }

        auto ne = evaluate(nxt, false, first, &cache, &next_cache);
        if (!ne.ok) continue;

        double progress = min(1.0, elapsed_ms() / 1940.0);
        double temp = 18.0 * pow(0.01 / 18.0, progress);
        bool accept = ne.turns <= curcost ||
            uniform_real_distribution<double>(0, 1)(rng) < exp((curcost - ne.turns) / max(0.01, temp));

        if (accept) {
            cur = nxt;
            curcost = ne.turns;
            cache = next_cache;
            if (curcost < bestcost) {
                bestcost = curcost;
                best = cur;
            }
        }
    }

    auto ans = evaluate(best, true, 0, nullptr, nullptr);
    if (!ans.ok) return 0;
    for (int i = 0; i < (int)ans.actions.size(); i += 2)
        cout << ans.actions[i] << ' ' << ans.actions[i + 1] << '\n';
    return 0;
}
# EVOLVE-BLOCK-END