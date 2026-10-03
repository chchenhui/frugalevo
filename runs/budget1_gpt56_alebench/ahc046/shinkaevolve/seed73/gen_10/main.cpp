# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

static constexpr int N = 20;
static constexpr int M = 40;
static constexpr int SEG = M - 1;
static constexpr int LIM = 2 * N * M;
static constexpr int INF = 1e9;
static constexpr int WORDS = 7;
static constexpr int BEAM = 34;

struct Pos {
    int r, c;
    bool operator==(const Pos& o) const { return r == o.r && c == o.c; }
};
static const int dr[4] = {-1, 1, 0, 0};
static const int dc[4] = {0, 0, -1, 1};
static const char ds[4] = {'U', 'D', 'L', 'R'};
static const int revd[4] = {1, 0, 3, 2};

struct Board {
    array<unsigned long long, WORDS> b{};
    bool get(int id) const { return (b[id >> 6] >> (id & 63)) & 1ULL; }
    void flip(int id) { b[id >> 6] ^= 1ULL << (id & 63); }
    bool operator==(const Board& o) const { return b == o.b; }
};

static inline bool inside(int r, int c) {
    return 0 <= r && r < N && 0 <= c && c < N;
}
static inline int idof(int r, int c) { return r * N + c; }
static inline bool blocked(const Board& b, int r, int c) {
    return !inside(r, c) || b.get(idof(r, c));
}

struct Route {
    int cost = INF;
    string act;
};

static Route shortest_path(Pos st, Pos gl, const Board& b, bool build) {
    if (blocked(b, st.r, st.c) || blocked(b, gl.r, gl.c)) return {};

    static int dist[400], par[400], pa[400], pd[400];
    for (int i = 0; i < 400; ++i) dist[i] = -1;

    int s = idof(st.r, st.c), g = idof(gl.r, gl.c);
    int q[400], head = 0, tail = 0;
    dist[s] = 0;
    q[tail++] = s;

    while (head < tail) {
        int v = q[head++];
        if (v == g) break;
        int r = v / N, c = v % N;

        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d], nc = c + dc[d];
            if (!blocked(b, nr, nc)) {
                int to = idof(nr, nc);
                if (dist[to] == -1) {
                    dist[to] = dist[v] + 1;
                    par[to] = v; pa[to] = 0; pd[to] = d;
                    q[tail++] = to;
                }
            }

            nr = r; nc = c;
            while (!blocked(b, nr + dr[d], nc + dc[d])) {
                nr += dr[d];
                nc += dc[d];
            }
            int to = idof(nr, nc);
            if (to != v && dist[to] == -1) {
                dist[to] = dist[v] + 1;
                par[to] = v; pa[to] = 1; pd[to] = d;
                q[tail++] = to;
            }
        }
    }

    if (dist[g] < 0) return {};
    Route ret;
    ret.cost = dist[g];
    if (!build) return ret;

    vector<pair<char,char>> tmp;
    for (int v = g; v != s; v = par[v]) {
        tmp.push_back({pa[v] ? 'S' : 'M', ds[pd[v]]});
    }
    reverse(tmp.begin(), tmp.end());
    ret.act.reserve(tmp.size() * 2);
    for (auto [a, d] : tmp) {
        ret.act.push_back(a);
        ret.act.push_back(d);
    }
    return ret;
}

// Route to target; if it currently contains a block, reach an adjacent cell,
// remove the block, and move onto the target.
static Route route_to_target(Pos st, Pos target, Board& b, bool build) {
    if (!b.get(idof(target.r, target.c))) {
        return shortest_path(st, target, b, build);
    }

    Route best;
    int bestd = -1;
    for (int d = 0; d < 4; ++d) {
        int ar = target.r + dr[d], ac = target.c + dc[d];
        if (blocked(b, ar, ac)) continue;
        Route x = shortest_path(st, {ar, ac}, b, build);
        if (x.cost < best.cost) {
            best = x;
            bestd = d;
        }
    }
    if (bestd < 0 || best.cost >= INF) return {};

    // From adjacent target+dir, direction toward target is reverse(dir).
    int toward = revd[bestd];
    b.flip(idof(target.r, target.c));
    best.cost += 2;
    if (build) {
        best.act.push_back('A'); best.act.push_back(ds[toward]);
        best.act.push_back('M'); best.act.push_back(ds[toward]);
    }
    return best;
}

struct State {
    Board board;
    Pos pos;
    int cost;
    string act;
    double key;
};

static int empty_grid_estimate(Pos a, Pos z) {
    // A cheap look-ahead estimate. It rewards ending nearer to the next target
    // while retaining the accumulated turn count as the dominant criterion.
    int man = abs(a.r - z.r) + abs(a.c - z.c);
    int slideish = 2 + min(min(a.r + z.r, 38 - a.r - z.r),
                           min(a.c + z.c, 38 - a.c - z.c));
    return min(man, max(1, slideish));
}

static bool same_state(const State& a, const State& b) {
    return a.pos == b.pos && a.board == b.board;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int nin, minput;
    if (!(cin >> nin >> minput)) return 0;

    vector<Pos> p(M);
    for (int i = 0; i < M; ++i) cin >> p[i].r >> p[i].c;

    vector<State> beam;
    beam.push_back({Board(), p[0], 0, "", 0.0});

    for (int step = 1; step < M; ++step) {
        vector<State> cand;
        Pos target = p[step];

        for (const State& base : beam) {
            if (base.cost >= LIM) continue;

            // Fundamental transition 1: shortest route, including automatic
            // removal when a previously created wall covers this target.
            Board reachedBoard = base.board;
            Route arrival = route_to_target(base.pos, target, reachedBoard, true);
            if (arrival.cost >= INF || base.cost + arrival.cost > LIM) continue;

            State reached{reachedBoard, target, base.cost + arrival.cost,
                          base.act + arrival.act, 0.0};

            // Leave the target unchanged.
            cand.push_back(reached);

            // Fundamental transition 2: local wall construction after visiting.
            // These choices keep diverse reusable stopper layouts in the beam.
            for (int d = 0; d < 4; ++d) {
                int nr = target.r + dr[d], nc = target.c + dc[d];
                if (!inside(nr, nc)) continue;
                State x = reached;
                x.board.flip(idof(nr, nc));
                x.cost++;
                x.act.push_back('A'); x.act.push_back(ds[d]);
                if (x.cost <= LIM) cand.push_back(move(x));
            }

            // Move one square away from target and alter any adjacent square.
            // Altering the departed target itself is intentionally included.
            for (int md = 0; md < 4; ++md) {
                int rr = target.r + dr[md], cc = target.c + dc[md];
                if (blocked(reached.board, rr, cc)) continue;
                for (int ad = 0; ad < 4; ++ad) {
                    int br = rr + dr[ad], bc = cc + dc[ad];
                    if (!inside(br, bc)) continue;
                    State x = reached;
                    x.pos = {rr, cc};
                    x.board.flip(idof(br, bc));
                    x.cost += 2;
                    x.act.push_back('M'); x.act.push_back(ds[md]);
                    x.act.push_back('A'); x.act.push_back(ds[ad]);
                    if (x.cost <= LIM) cand.push_back(move(x));
                }
            }

            // Fundamental transition 3: construct an explicit stopper beyond
            // target, return to the opposite side, and slide into the target.
            // This often creates a durable rail useful for many later targets.
            if (!reached.board.get(idof(target.r, target.c))) {
                for (int d = 0; d < 4; ++d) {
                    int fr = target.r + dr[d], fc = target.c + dc[d];
                    int sr = target.r - dr[d], sc = target.c - dc[d];
                    if (!inside(fr, fc) || !inside(sr, sc)) continue;
                    if (reached.board.get(idof(fr, fc))) continue;

                    Board bb = reached.board;
                    bb.flip(idof(fr, fc));
                    Route back = shortest_path(target, {sr, sc}, bb, true);
                    if (back.cost >= INF) continue;

                    State x;
                    x.board = bb;
                    x.pos = target;
                    x.cost = reached.cost + 1 + back.cost + 1;
                    if (x.cost > LIM) continue;
                    x.act = reached.act;
                    x.act.push_back('A'); x.act.push_back(ds[d]);
                    x.act += back.act;
                    x.act.push_back('S'); x.act.push_back(ds[d]);
                    cand.push_back(move(x));
                }
            }
        }

        if (cand.empty()) break;

        Pos nxt = (step + 1 < M ? p[step + 1] : target);
        for (State& s : cand) {
            // One-step lookahead is deliberately modest: it maintains a broad
            // set of block configurations rather than greedily minimizing only
            // the current segment.
            s.key = s.cost + (step + 1 < M ? 0.72 * empty_grid_estimate(s.pos, nxt) : 0.0);
        }

        sort(cand.begin(), cand.end(), [](const State& a, const State& b) {
            if (a.key != b.key) return a.key < b.key;
            return a.cost < b.cost;
        });

        beam.clear();
        for (const State& x : cand) {
            bool duplicate = false;
            for (const State& y : beam) {
                if (same_state(x, y)) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) beam.push_back(x);
            if ((int)beam.size() >= BEAM) break;
        }
    }

    if (beam.empty()) return 0;
    const State* ans = &beam[0];
    for (const State& x : beam) {
        if (x.cost < ans->cost) ans = &x;
    }

    for (size_t i = 0; i + 1 < ans->act.size(); i += 2) {
        cout << ans->act[i] << ' ' << ans->act[i + 1] << '\n';
    }
    return 0;
}
# EVOLVE-BLOCK-END