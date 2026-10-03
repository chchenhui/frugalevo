# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

static constexpr int N = 20;
static constexpr int M = 40;
static constexpr int K = M - 1;
static constexpr int LIMIT = 2 * N * M;
static constexpr int INF = 1e9;

static const int DR[4] = {-1, 1, 0, 0};
static const int DC[4] = {0, 0, -1, 1};
static const int REV[4] = {1, 0, 3, 2};
static const char DCH[4] = {'U', 'D', 'L', 'R'};

struct Pos {
    int r, c;
    int id() const { return r * N + c; }
    bool operator==(const Pos& o) const { return r == o.r && c == o.c; }
};

struct Board {
    array<uint64_t, 7> bits{};

    bool inside(int r, int c) const {
        return 0 <= r && r < N && 0 <= c && c < N;
    }
    bool block(int r, int c) const {
        if (!inside(r, c)) return true;
        int x = r * N + c;
        return (bits[x >> 6] >> (x & 63)) & 1ULL;
    }
    void toggle(int r, int c) {
        int x = r * N + c;
        bits[x >> 6] ^= 1ULL << (x & 63);
    }
};

struct State {
    Board board;
    Pos pos;
};

struct Route {
    int cost = INF;
    vector<pair<char, char>> act;
};

class Router {
    array<int, N * N> dist;
    array<int, N * N> parent;
    array<char, N * N> pa, pd;
    array<int, N * N> q;

public:
    Route find(const State& st, Pos goal, int forbidden = -1, bool record = false) {
        Route out;
        if (st.board.block(goal.r, goal.c)) return out;

        dist.fill(-1);
        int head = 0, tail = 0;
        int s = st.pos.id(), g = goal.id();
        dist[s] = 0;
        q[tail++] = s;

        while (head < tail) {
            int v = q[head++];
            if (v == g) break;
            int r = v / N, c = v % N;

            for (int d = 0; d < 4; ++d) {
                int nr = r + DR[d], nc = c + DC[d];
                if (st.board.block(nr, nc)) continue;
                int u = nr * N + nc;
                if (u == forbidden && u != g) continue;
                if (dist[u] != -1) continue;
                dist[u] = dist[v] + 1;
                parent[u] = v; pa[u] = 'M'; pd[u] = DCH[d];
                q[tail++] = u;
            }

            for (int d = 0; d < 4; ++d) {
                int nr = r, nc = c;
                bool crossed_forbidden = false;
                while (!st.board.block(nr + DR[d], nc + DC[d])) {
                    nr += DR[d];
                    nc += DC[d];
                    if (nr * N + nc == forbidden && forbidden != g) {
                        crossed_forbidden = true;
                        break;
                    }
                }
                if (crossed_forbidden) continue;
                int u = nr * N + nc;
                if (u == v || dist[u] != -1) continue;
                dist[u] = dist[v] + 1;
                parent[u] = v; pa[u] = 'S'; pd[u] = DCH[d];
                q[tail++] = u;
            }
        }

        if (dist[g] < 0) return out;
        out.cost = dist[g];
        if (!record) return out;

        vector<pair<char, char>> rev;
        for (int v = g; v != s; v = parent[v]) rev.push_back({pa[v], pd[v]});
        reverse(rev.begin(), rev.end());
        out.act = move(rev);
        return out;
    }
};

struct Segment {
    bool ok = false;
    int cost = INF;
    State after;
    vector<pair<char, char>> actions;
};

class SegmentFactory {
    Router& router;

    static void append(vector<pair<char, char>>& a, const vector<pair<char, char>>& b) {
        a.insert(a.end(), b.begin(), b.end());
    }

    bool add_route(State& s, Pos to, int forbid, bool record,
                   vector<pair<char, char>>& a, int& cost) {
        Route r = router.find(s, to, forbid, record);
        if (r.cost == INF) return false;
        cost += r.cost;
        if (record) append(a, r.act);
        s.pos = to;
        return true;
    }

    bool apply_base(int base, State& s, Pos target, bool record,
                    vector<pair<char, char>>& a, int& cost) {
        if (base == 0) {
            return add_route(s, target, -1, record, a, cost);
        }

        if (base == 1) {
            if (!s.board.block(target.r, target.c)) return false;
            int best = INF, bestd = -1;
            Pos bestp{-1, -1};
            for (int d = 0; d < 4; ++d) {
                Pos p{target.r + DR[d], target.c + DC[d]};
                if (s.board.block(p.r, p.c)) continue;
                Route r = router.find(s, p, target.id(), false);
                if (r.cost < best) best = r.cost, bestd = d, bestp = p;
            }
            if (bestd < 0) return false;
            if (!add_route(s, bestp, target.id(), record, a, cost)) return false;
            int toward = REV[bestd];
            s.board.toggle(target.r, target.c);
            ++cost;
            if (record) a.push_back({'A', DCH[toward]});
            s.pos = target;
            ++cost;
            if (record) a.push_back({'M', DCH[toward]});
            return true;
        }

        int d = (base - 2) & 3;
        bool create = base >= 6;
        Pos before{target.r - DR[d], target.c - DC[d]};
        Pos stopper{target.r + DR[d], target.c + DC[d]};
        if (!s.board.inside(before.r, before.c)) return false;

        if (!create) {
            if (!s.board.block(stopper.r, stopper.c)) return false;
            if (!add_route(s, before, target.id(), record, a, cost)) return false;
            ++cost;
            if (record) a.push_back({'S', DCH[d]});
            s.pos = target;
            return true;
        }

        if (!s.board.inside(stopper.r, stopper.c) || s.board.block(stopper.r, stopper.c)) return false;
        if (!add_route(s, target, -1, record, a, cost)) return false;
        s.board.toggle(stopper.r, stopper.c);
        ++cost;
        if (record) a.push_back({'A', DCH[d]});

        // Starting at the current target is allowed; only re-entering it is forbidden.
        if (!add_route(s, before, target.id(), record, a, cost)) return false;
        ++cost;
        if (record) a.push_back({'S', DCH[d]});
        s.pos = target;
        return true;
    }

    bool apply_post(int post, State& s, bool record,
                    vector<pair<char, char>>& a, int& cost) {
        if (post == 0) return true;

        if (1 <= post && post <= 4) {
            int d = post - 1;
            int nr = s.pos.r + DR[d], nc = s.pos.c + DC[d];
            if (!s.board.inside(nr, nc)) return false;
            s.board.toggle(nr, nc);
            ++cost;
            if (record) a.push_back({'A', DCH[d]});
            return true;
        }

        int x = post - 5;
        int md = x / 4, ad = x % 4;
        int nr = s.pos.r + DR[md], nc = s.pos.c + DC[md];
        if (s.board.block(nr, nc)) return false;
        int tr = nr + DR[ad], tc = nc + DC[ad];
        if (!s.board.inside(tr, tc)) return false;

        ++cost;
        if (record) a.push_back({'M', DCH[md]});
        s.pos = {nr, nc};
        s.board.toggle(tr, tc);
        ++cost;
        if (record) a.push_back({'A', DCH[ad]});
        return true;
    }

public:
    explicit SegmentFactory(Router& r) : router(r) {}

    Segment run(int code, const State& in, Pos target, bool record) {
        Segment ret;
        State s = in;
        vector<pair<char, char>> actions;
        int cost = 0;
        int base = code % 10;
        int post = code / 10;

        if (!apply_base(base, s, target, record, actions, cost)) return ret;
        if (!apply_post(post, s, record, actions, cost)) return ret;

        ret.ok = true;
        ret.cost = cost;
        ret.after = s;
        if (record) ret.actions = move(actions);
        return ret;
    }
};

struct Evaluation {
    bool ok = false;
    int total = INF;
    vector<State> checkpoint;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int ni, mi;
    cin >> ni >> mi;
    Pos start;
    cin >> start.r >> start.c;
    vector<Pos> target(K);
    for (auto& p : target) cin >> p.r >> p.c;

    auto started = chrono::steady_clock::now();
    auto elapsed = [&]() {
        return chrono::duration<double, milli>(chrono::steady_clock::now() - started).count();
    };

    mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count());
    Router router;
    SegmentFactory factory(router);

    constexpr int POSTS = 21;
    constexpr int CODES = 10 * POSTS;

    auto evaluate = [&](const vector<int>& choice, int from,
                        const vector<State>* oldcp) -> Evaluation {
        Evaluation e;
        e.checkpoint.resize(K);
        State s;
        int total = 0;

        if (from > 0 && oldcp != nullptr) {
            for (int i = 0; i < from; ++i) e.checkpoint[i] = (*oldcp)[i];
            s = (*oldcp)[from];
            for (int i = 0; i < from; ++i) {
                Segment z = factory.run(choice[i], e.checkpoint[i], target[i], false);
                if (!z.ok) return e;
                total += z.cost;
            }
        } else {
            from = 0;
            s.pos = start;
        }

        for (int i = from; i < K; ++i) {
            e.checkpoint[i] = s;
            Segment z = factory.run(choice[i], s, target[i], false);
            if (!z.ok || total + z.cost > LIMIT) return e;
            total += z.cost;
            s = z.after;
        }
        e.ok = true;
        e.total = total;
        return e;
    };

    vector<int> current(K), best(K);
    State greedy;
    greedy.pos = start;
    int greedy_cost = 0;

    for (int i = 0; i < K; ++i) {
        int chosen = -1, best_local = INF;
        for (int code = 0; code < CODES; ++code) {
            Segment z = factory.run(code, greedy, target[i], false);
            if (z.ok && z.cost < best_local) {
                best_local = z.cost;
                chosen = code;
            }
        }
        if (chosen < 0) chosen = 0;
        Segment z = factory.run(chosen, greedy, target[i], false);
        if (!z.ok) {
            chosen = 0;
            z = factory.run(chosen, greedy, target[i], false);
        }
        current[i] = chosen;
        greedy_cost += z.cost;
        greedy = z.after;
    }

    Evaluation cur = evaluate(current, 0, nullptr);
    if (!cur.ok) return 0;
    int cur_score = cur.total;
    best = current;
    int best_score = cur_score;

    vector<int> candidate(K);
    uniform_real_distribution<double> real01(0.0, 1.0);

    while (elapsed() < 1900.0) {
        candidate = current;
        int changes;
        double r = real01(rng);
        if (r < 0.60) changes = 1;
        else if (r < 0.86) changes = 2;
        else if (r < 0.96) changes = 3;
        else changes = 4 + (rng() % 6);
        changes = min(changes, K);

        int first = K;
        if (real01(rng) < 0.72) {
            vector<int> order(K);
            iota(order.begin(), order.end(), 0);
            shuffle(order.begin(), order.end(), rng);
            for (int t = 0; t < changes; ++t) {
                int i = order[t];
                first = min(first, i);
                int old = candidate[i], nw = old;
                do nw = rng() % CODES; while (nw == old);
                candidate[i] = nw;
            }
        } else {
            int l = rng() % (K - changes + 1);
            first = l;
            for (int i = l; i < l + changes; ++i) {
                int old = candidate[i], nw = old;
                do nw = rng() % CODES; while (nw == old);
                candidate[i] = nw;
            }
        }

        // Periodically replace a selected segment by its best local plan.
        if (real01(rng) < 0.18) {
            int i = rng() % K;
            first = min(first, i);
            State s = cur.checkpoint[i];
            int local_best = candidate[i], val = INF;
            for (int trial = 0; trial < 45; ++trial) {
                int code = (trial < 10 ? trial : int(rng() % CODES));
                Segment z = factory.run(code, s, target[i], false);
                if (z.ok && z.cost < val) val = z.cost, local_best = code;
            }
            candidate[i] = local_best;
        }

        Evaluation nxt = evaluate(candidate, first, &cur.checkpoint);
        if (!nxt.ok) continue;

        double progress = min(1.0, elapsed() / 1900.0);
        double temp = 16.0 * pow(0.015 / 16.0, progress);
        bool accept = nxt.total <= cur_score;
        if (!accept) {
            double p = exp(double(cur_score - nxt.total) / max(0.01, temp));
            accept = real01(rng) < p;
        }

        if (accept) {
            current.swap(candidate);
            cur = move(nxt);
            cur_score = cur.total;
            if (cur_score < best_score) {
                best_score = cur_score;
                best = current;
            }
        }
    }

    State replay;
    replay.pos = start;
    vector<pair<char, char>> output;
    for (int i = 0; i < K; ++i) {
        Segment z = factory.run(best[i], replay, target[i], true);
        if (!z.ok) return 0;
        output.insert(output.end(), z.actions.begin(), z.actions.end());
        replay = z.after;
    }

    for (auto [a, d] : output) cout << a << ' ' << d << '\n';
    return 0;
}
# EVOLVE-BLOCK-END