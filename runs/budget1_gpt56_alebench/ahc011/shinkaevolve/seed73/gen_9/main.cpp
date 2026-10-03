# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

static constexpr int MAXN = 10;
static constexpr int MAXV = MAXN * MAXN;
static constexpr int DR[4] = {-1, 1, 0, 0};
static constexpr int DC[4] = {0, 0, -1, 1};
static constexpr char MC[4] = {'U', 'D', 'L', 'R'};

int N, T, V;
uint64_t zob[MAXV][16];

struct RNG {
    uint64_t x;
    explicit RNG(uint64_t seed) : x(seed) {}
    uint64_t next() {
        x ^= x << 7;
        x ^= x >> 9;
        return x;
    }
    int next_int(int n) {
        return (int)(next() % n);
    }
};

struct Board {
    array<unsigned char, MAXV> a{};
    int empty = 0;
    uint64_t hash = 0;

    bool can_move(int d) const {
        int r = empty / N, c = empty % N;
        int nr = r + DR[d], nc = c + DC[d];
        return 0 <= nr && nr < N && 0 <= nc && nc < N;
    }

    void move(int d) {
        int to = (empty / N + DR[d]) * N + (empty % N + DC[d]);
        int tile = a[to];

        hash ^= zob[empty][0];
        hash ^= zob[to][tile];
        hash ^= zob[empty][tile];
        hash ^= zob[to][0];

        a[empty] = (unsigned char)tile;
        a[to] = 0;
        empty = to;
    }
};

struct Eval {
    unsigned char largest_tree;
    unsigned char components;
};

struct DSU {
    int p[MAXV];
    unsigned char sz[MAXV];
    unsigned char edges[MAXV];

    void init(const Board& b) {
        for (int i = 0; i < V; ++i) {
            p[i] = i;
            sz[i] = (b.a[i] != 0);
            edges[i] = 0;
        }
    }

    int find(int x) {
        while (p[x] != x) {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }

    void edge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) {
            ++edges[x];
            return;
        }
        if (sz[x] < sz[y]) swap(x, y);
        p[y] = x;
        sz[x] += sz[y];
        edges[x] += edges[y] + 1;
    }
};

unordered_map<uint64_t, Eval> eval_cache;

Eval evaluate(const Board& b) {
    auto it = eval_cache.find(b.hash);
    if (it != eval_cache.end()) return it->second;

    DSU uf;
    uf.init(b);

    for (int r = 0; r < N; ++r) {
        int base = r * N;
        for (int c = 0; c + 1 < N; ++c) {
            int x = base + c, y = x + 1;
            if ((b.a[x] & 4) && (b.a[y] & 1)) uf.edge(x, y);
        }
    }
    for (int r = 0; r + 1 < N; ++r) {
        int base = r * N;
        for (int c = 0; c < N; ++c) {
            int x = base + c, y = x + N;
            if ((b.a[x] & 8) && (b.a[y] & 2)) uf.edge(x, y);
        }
    }

    int best = 0, components = 0;
    for (int i = 0; i < V; ++i) {
        if (uf.p[i] == i && uf.sz[i]) {
            ++components;
            if (uf.edges[i] + 1 == uf.sz[i]) {
                best = max(best, (int)uf.sz[i]);
            }
        }
    }

    Eval result{(unsigned char)best, (unsigned char)components};
    if (eval_cache.size() < 1800000) eval_cache.emplace(b.hash, result);
    return result;
}

struct Trace {
    int parent;
    char move;
};

vector<Trace> traces;

string trace_path(int trace_id) {
    string result;
    while (trace_id > 0) {
        result.push_back(traces[trace_id].move);
        trace_id = traces[trace_id].parent;
    }
    reverse(result.begin(), result.end());
    return result;
}

string candidate_path(int parent_trace, char last_move) {
    string result = trace_path(parent_trace);
    result.push_back(last_move);
    return result;
}

struct State {
    Board board;
    int trace;
    signed char previous_move;
    int value;
};

struct Candidate {
    Board board;
    int parent_trace;
    signed char move;
    int value;
};

static inline int state_value(const Eval& e, const Board& b, int total_moves) {
    const int target = V - 1;
    if (e.largest_tree == target) {
        return 1000000000 - total_moves;
    }

    // The primary term is the official objective.  Component count is used
    // as a strong structural guide toward joining tree fragments.
    int score = (int)e.largest_tree * 1000000;
    score -= max(0, (int)e.components - 1) * 700000;

    int er = b.empty / N, ec = b.empty % N;
    int corner_dist = (N - 1 - er) + (N - 1 - ec);
    score -= corner_dist * 3;
    score -= total_moves / 8;
    return score;
}

string route_empty_to_corner(Board& b) {
    string route;
    int r = b.empty / N, c = b.empty % N;
    while (c < N - 1) {
        b.move(3);
        route.push_back('R');
        ++c;
    }
    while (r < N - 1) {
        b.move(1);
        route.push_back('D');
        ++r;
    }
    return route;
}

int hex_value(char ch) {
    return ('0' <= ch && ch <= '9') ? ch - '0' : ch - 'a' + 10;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> N >> T;
    V = N * N;

    RNG zob_rng(0x9e3779b97f4a7c15ULL);
    for (int i = 0; i < V; ++i) {
        for (int x = 0; x < 16; ++x) zob[i][x] = zob_rng.next();
    }

    Board initial;
    for (int r = 0; r < N; ++r) {
        string s;
        cin >> s;
        for (int c = 0; c < N; ++c) {
            int pos = r * N + c;
            initial.a[pos] = (unsigned char)hex_value(s[c]);
            initial.hash ^= zob[pos][initial.a[pos]];
            if (initial.a[pos] == 0) initial.empty = pos;
        }
    }

    // Inputs are produced from a solution whose blank is at the lower-right
    // corner.  Moving the blank there first gives the beam a stable target
    // geometry and costs only Manhattan distance.
    string prefix;
    if (initial.empty != V - 1) prefix = route_empty_to_corner(initial);
    int prefix_len = (int)prefix.size();

    eval_cache.reserve(1900000);
    eval_cache.max_load_factor(0.72f);
    traces.reserve(700000);
    traces.push_back({-1, '?'});

    RNG rng((uint64_t)chrono::steady_clock::now().time_since_epoch().count());

    int beam_width;
    if (N == 6) beam_width = 1350;
    else if (N == 7) beam_width = 1150;
    else if (N == 8) beam_width = 850;
    else if (N == 9) beam_width = 580;
    else beam_width = 400;

    Eval init_eval = evaluate(initial);
    int best_size = init_eval.largest_tree;
    int best_total_moves = prefix_len;
    string best_path = prefix;

    vector<State> frontier;
    frontier.reserve(beam_width);
    frontier.push_back({initial, 0, -1, state_value(init_eval, initial, prefix_len)});

    unordered_map<uint64_t, int> reached_depth;
    reached_depth.reserve(1800000);
    reached_depth.max_load_factor(0.72f);
    reached_depth[initial.hash] = prefix_len;

    vector<Candidate> candidates;
    candidates.reserve(beam_width * 4);
    vector<int> order;
    order.reserve(beam_width * 3);

    const auto start = chrono::steady_clock::now();
    const int time_limit_ms = 2670;

    for (int depth = 0; prefix_len + depth < T && !frontier.empty(); ++depth) {
        if ((depth & 7) == 0) {
            int elapsed = (int)chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - start).count();
            if (elapsed >= time_limit_ms) break;
        }

        candidates.clear();
        int next_total_moves = prefix_len + depth + 1;

        for (const State& state : frontier) {
            for (int d = 0; d < 4; ++d) {
                if (state.previous_move >= 0 && (state.previous_move ^ 1) == d) continue;
                if (!state.board.can_move(d)) continue;

                Board next = state.board;
                next.move(d);

                auto hit = reached_depth.find(next.hash);
                if (hit != reached_depth.end()) {
                    if (hit->second <= next_total_moves) continue;
                    hit->second = next_total_moves;
                } else if (reached_depth.size() < 1750000) {
                    reached_depth.emplace(next.hash, next_total_moves);
                }

                Eval e = evaluate(next);

                bool improve = false;
                if (e.largest_tree > best_size) {
                    improve = true;
                } else if (e.largest_tree == V - 1 && best_size == V - 1 &&
                           next_total_moves < best_total_moves) {
                    improve = true;
                }

                if (improve) {
                    best_size = e.largest_tree;
                    best_total_moves = next_total_moves;
                    best_path = prefix + candidate_path(state.trace, MC[d]);
                }

                candidates.push_back({
                    next,
                    state.trace,
                    (signed char)d,
                    state_value(e, next, next_total_moves)
                });
            }
        }

        if (candidates.empty()) break;

        int selection_pool = min((int)candidates.size(), beam_width * 3);
        auto better = [](const Candidate& x, const Candidate& y) {
            return x.value > y.value;
        };

        if ((int)candidates.size() > selection_pool) {
            nth_element(candidates.begin(), candidates.begin() + selection_pool,
                        candidates.end(), better);
            candidates.resize(selection_pool);
        }
        sort(candidates.begin(), candidates.end(), better);

        frontier.clear();
        int elite = min((int)candidates.size(), max(1, beam_width / 4));

        for (int i = 0; i < elite; ++i) {
            traces.push_back({candidates[i].parent_trace, MC[candidates[i].move]});
            frontier.push_back({
                candidates[i].board,
                (int)traces.size() - 1,
                candidates[i].move,
                candidates[i].value
            });
        }

        order.clear();
        for (int i = elite; i < (int)candidates.size(); ++i) order.push_back(i);

        for (int i = (int)order.size() - 1; i > 0; --i) {
            swap(order[i], order[rng.next_int(i + 1)]);
        }

        for (int idx : order) {
            if ((int)frontier.size() >= beam_width) break;
            const Candidate& c = candidates[idx];
            traces.push_back({c.parent_trace, MC[c.move]});
            frontier.push_back({c.board, (int)traces.size() - 1, c.move, c.value});
        }
    }

    cout << best_path << '\n';
    return 0;
}
# EVOLVE-BLOCK-END