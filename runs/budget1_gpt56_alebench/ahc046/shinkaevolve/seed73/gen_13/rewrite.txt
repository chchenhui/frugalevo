# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

static constexpr int N = 20;
static constexpr int M = 40;
static constexpr int SEG = 39;
static constexpr int LIM = 1600;
static constexpr int INF = 1e9;

struct Pos {
    int r, c;
    bool operator==(const Pos& o) const { return r == o.r && c == o.c; }
    bool operator!=(const Pos& o) const { return !(*this == o); }
};

using Grid = array<array<unsigned char, N>, N>;

static const int dr[4] = {-1, 1, 0, 0};
static const int dc[4] = {0, 0, -1, 1};
static const int revd[4] = {1, 0, 3, 2};
static const char dch[4] = {'U', 'D', 'L', 'R'};

static mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count());
static auto start_clock = chrono::steady_clock::now();

static inline bool inside(int r, int c) {
    return (unsigned)r < N && (unsigned)c < N;
}
static inline bool blocked(int r, int c, const Grid& g) {
    return !inside(r, c) || g[r][c];
}
static inline int id(Pos p) { return p.r * N + p.c; }
static inline Pos pos(int x) { return {x / N, x % N}; }

struct Pred {
    short from;
    char act, dir;
};

static int dista[N * N];
static int seen[N * N];
static int stamp = 0;
static Pred pred[N * N];
static int que[N * N];

static string restore(int s, int t) {
    string z;
    while (t != s) {
        Pred p = pred[t];
        z.push_back(p.dir);
        z.push_back(p.act);
        t = p.from;
    }
    reverse(z.begin(), z.end());
    return z;
}

struct BRes {
    int cost;
    string actions;
};

static BRes bfs(Pos st, Pos goal, const Grid& g, Pos avoid, bool use_avoid, bool build) {
    if (!inside(st.r, st.c) || blocked(goal.r, goal.c, g)) return {INF, ""};
    if (use_avoid && st == avoid && st != goal) return {INF, ""};

    ++stamp;
    int s = id(st), target = id(goal);
    int head = 0, tail = 0;
    que[tail++] = s;
    seen[s] = stamp;
    dista[s] = 0;

    while (head < tail) {
        int v = que[head++];
        if (v == target) break;
        Pos cur = pos(v);
        int nd = dista[v] + 1;

        for (int d = 0; d < 4; ++d) {
            int nr = cur.r + dr[d], nc = cur.c + dc[d];
            if (blocked(nr, nc, g)) continue;
            if (use_avoid && nr == avoid.r && nc == avoid.c && id({nr,nc}) != target) continue;
            int u = nr * N + nc;
            if (seen[u] != stamp) {
                seen[u] = stamp;
                dista[u] = nd;
                if (build) pred[u] = {(short)v, 'M', dch[d]};
                que[tail++] = u;
            }
        }

        for (int d = 0; d < 4; ++d) {
            int nr = cur.r, nc = cur.c;
            bool crossed_avoid = false;
            while (true) {
                int xr = nr + dr[d], xc = nc + dc[d];
                if (blocked(xr, xc, g)) break;
                if (use_avoid && xr == avoid.r && xc == avoid.c && xr * N + xc != target) {
                    crossed_avoid = true;
                    break;
                }
                nr = xr; nc = xc;
            }
            if (crossed_avoid || (nr == cur.r && nc == cur.c)) continue;
            int u = nr * N + nc;
            if (seen[u] != stamp) {
                seen[u] = stamp;
                dista[u] = nd;
                if (build) pred[u] = {(short)v, 'S', dch[d]};
                que[tail++] = u;
            }
        }
    }

    if (seen[target] != stamp) return {INF, ""};
    return {dista[target], build ? restore(s, target) : ""};
}

struct SegRes {
    int turns = INF;
    string actions;
};

static constexpr int BASE = 10;
static constexpr int POST = 17;
static constexpr int STRAT = BASE * POST;

static bool base_strategy(int code, Pos cur, Pos tar, Grid& g, SegRes& out, bool build) {
    if (code == 0) {
        if (blocked(tar.r, tar.c, g)) return false;
        BRes b = bfs(cur, tar, g, {-1,-1}, false, build);
        if (b.cost == INF) return false;
        out.turns = b.cost;
        out.actions = move(b.actions);
        return true;
    }

    if (code == 1) {
        if (!blocked(tar.r, tar.c, g)) return false;
        int best = INF, bestdir = -1;
        Pos bestp{-1,-1};

        for (int d = 0; d < 4; ++d) {
            Pos q{tar.r + dr[d], tar.c + dc[d]};
            if (!inside(q.r,q.c) || blocked(q.r,q.c,g)) continue;
            BRes b = bfs(cur, q, g, tar, true, build);
            if (b.cost < best) {
                best = b.cost;
                bestdir = d;
                bestp = q;
                if (build) out.actions = move(b.actions);
            }
        }
        if (best == INF) return false;
        int toward = revd[bestdir];
        out.turns = best + 2;
        if (build) {
            out.actions += 'A'; out.actions += dch[toward];
            out.actions += 'M'; out.actions += dch[toward];
        }
        g[tar.r][tar.c] ^= 1;
        return true;
    }

    int type = code < 6 ? 0 : 1;
    int d = code < 6 ? code - 2 : code - 6;
    if (blocked(tar.r, tar.c, g)) return false;

    Pos pre{tar.r - dr[d], tar.c - dc[d]};
    Pos wall{tar.r + dr[d], tar.c + dc[d]};
    if (!inside(pre.r, pre.c)) return false;

    if (type == 0) {
        if (inside(wall.r, wall.c) && !g[wall.r][wall.c]) return false;
        BRes b = bfs(cur, pre, g, tar, true, build);
        if (b.cost == INF) return false;
        out.turns = b.cost + 1;
        out.actions = move(b.actions);
        if (build) { out.actions += 'S'; out.actions += dch[d]; }
        return true;
    }

    if (!inside(wall.r, wall.c) || g[wall.r][wall.c]) return false;
    BRes a = bfs(cur, tar, g, {-1,-1}, false, build);
    if (a.cost == INF) return false;
    Grid h = g;
    h[wall.r][wall.c] = 1;
    BRes b = bfs(tar, pre, h, tar, true, build);
    if (b.cost == INF) return false;

    out.turns = a.cost + b.cost + 2;
    if (build) {
        out.actions = move(a.actions);
        out.actions += 'A'; out.actions += dch[d];
        out.actions += b.actions;
        out.actions += 'S'; out.actions += dch[d];
    }
    g = h;
    return true;
}

static bool apply_strategy(int code, Pos& player, Pos target, Grid& g, SegRes& out, bool build) {
    out.turns = 0;
    out.actions.clear();
    Grid backup = g;

    int bcode = code % BASE;
    int post = code / BASE;
    if (!base_strategy(bcode, player, target, g, out, build)) {
        g = backup;
        return false;
    }

    Pos p = target;
    if (post == 0) {
    } else if (post <= 4) {
        int d = post - 1;
        int nr = p.r + dr[d], nc = p.c + dc[d];
        if (!inside(nr,nc)) { g = backup; return false; }
        g[nr][nc] ^= 1;
        ++out.turns;
        if (build) { out.actions += 'A'; out.actions += dch[d]; }
    } else {
        int x = post - 5;
        int md = x / 3, k = x % 3;
        int ad = -1, cnt = 0;
        for (int d = 0; d < 4; ++d) {
            if (d == revd[md]) continue;
            if (cnt++ == k) { ad = d; break; }
        }
        int nr = p.r + dr[md], nc = p.c + dc[md];
        if (!inside(nr,nc) || g[nr][nc]) { g = backup; return false; }
        int ar = nr + dr[ad], ac = nc + dc[ad];
        if (!inside(ar,ac)) { g = backup; return false; }
        g[ar][ac] ^= 1;
        out.turns += 2;
        if (build) {
            out.actions += 'M'; out.actions += dch[md];
            out.actions += 'A'; out.actions += dch[ad];
        }
        p = {nr,nc};
    }

    player = p;
    return true;
}

struct Cache {
    Pos player;
    Grid grid;
    int turns;
};

static Pos initial_pos;
static array<Pos, SEG> targets;

static int evaluate(const array<int,SEG>& choice, int begin,
                    const vector<Cache>* oldc, vector<Cache>* newc,
                    bool build, string* log) {
    Grid g{};
    Pos player = initial_pos;
    int total = 0;

    if (begin > 0 && oldc && begin < (int)oldc->size() && (*oldc)[begin].turns < INF) {
        g = (*oldc)[begin].grid;
        player = (*oldc)[begin].player;
        total = (*oldc)[begin].turns;
        if (newc) copy(oldc->begin(), oldc->begin() + begin, newc->begin());
    } else begin = 0;

    if (build && log) log->clear();

    for (int i = begin; i < SEG; ++i) {
        if (newc) (*newc)[i] = {player, g, total};
        SegRes r;
        if (!apply_strategy(choice[i], player, targets[i], g, r, build) ||
            total + r.turns > LIM) {
            if (newc) for (int j = i; j < SEG; ++j) (*newc)[j].turns = INF;
            return INF;
        }
        total += r.turns;
        if (build && log) *log += r.actions;
    }
    return total;
}

static inline double elapsed() {
    return chrono::duration<double, milli>(chrono::steady_clock::now() - start_clock).count();
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int ni, mi;
    cin >> ni >> mi;
    cin >> initial_pos.r >> initial_pos.c;
    for (int i = 0; i < SEG; ++i) cin >> targets[i].r >> targets[i].c;

    array<int,SEG> cur{}, best{}, nxt{};
    vector<Cache> cache(SEG), ncache(SEG);
    Grid g{};
    Pos player = initial_pos;
    int current = 0;

    for (int i = 0; i < SEG; ++i) {
        int bc = -1, bt = INF;
        for (int s = 0; s < STRAT; ++s) {
            Grid tg = g;
            Pos tp = player;
            SegRes z;
            if (apply_strategy(s, tp, targets[i], tg, z, false) && z.turns < bt) {
                bt = z.turns;
                bc = s;
            }
        }
        if (bc < 0) return 0;
        cur[i] = bc;
        cache[i] = {player, g, current};
        SegRes z;
        apply_strategy(bc, player, targets[i], g, z, false);
        current += z.turns;
    }

    best = cur;
    int bestcost = current;
    uniform_real_distribution<double> real01(0.0, 1.0);

    while (elapsed() < 1940.0) {
        nxt = cur;
        int changed = 0;
        int begin = SEG;

        double roll = real01(rng);
        if (roll < 0.62) {
            int count = real01(rng) < 0.68 ? 1 : (real01(rng) < 0.86 ? 2 : 3);
            array<int,SEG> ord;
            iota(ord.begin(), ord.end(), 0);
            shuffle(ord.begin(), ord.end(), rng);
            for (int q = 0; q < count; ++q) {
                int k = ord[q];
                int v;
                do v = (int)(rng() % STRAT); while (v == nxt[k]);
                nxt[k] = v;
                begin = min(begin, k);
                ++changed;
            }
        } else if (roll < 0.82) {
            int k = (int)(rng() % SEG);
            int old = nxt[k], base = old % BASE, post = old / BASE;
            if (real01(rng) < 0.5) {
                int np;
                do np = (int)(rng() % POST); while (np == post);
                nxt[k] = np * BASE + base;
            } else {
                int nb;
                do nb = (int)(rng() % BASE); while (nb == base);
                nxt[k] = post * BASE + nb;
            }
            begin = k;
            changed = 1;
        } else {
            int k = (int)(rng() % SEG);
            if (cache[k].turns >= INF) continue;
            int chosen = cur[k], score = INF;
            for (int s = 0; s < STRAT; ++s) {
                Grid tg = cache[k].grid;
                Pos tp = cache[k].player;
                SegRes z;
                if (apply_strategy(s, tp, targets[k], tg, z, false) && z.turns < score) {
                    score = z.turns;
                    chosen = s;
                }
            }
            if (chosen == cur[k]) continue;
            nxt[k] = chosen;
            begin = k;
            changed = 1;
        }

        if (!changed) continue;
        int value = evaluate(nxt, begin, &cache, &ncache, false, nullptr);
        if (value == INF) continue;

        double progress = min(1.0, elapsed() / 1940.0);
        double temp = 18.0 * pow(0.01 / 18.0, progress);
        bool accept = value <= current ||
            real01(rng) < exp((double)(current - value) / max(0.01, temp));

        if (accept) {
            cur = nxt;
            current = value;
            cache.swap(ncache);
            if (current < bestcost) {
                bestcost = current;
                best = cur;
            }
        }
    }

    string answer;
    int final_cost = evaluate(best, 0, nullptr, nullptr, true, &answer);
    if (final_cost == INF) return 0;
    for (int i = 0; i + 1 < (int)answer.size(); i += 2)
        cout << answer[i] << ' ' << answer[i + 1] << '\n';
    return 0;
}
# EVOLVE-BLOCK-END