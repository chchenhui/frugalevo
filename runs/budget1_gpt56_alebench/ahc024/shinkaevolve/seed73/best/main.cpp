# EVOLVE-BLOCK-START
#pragma GCC optimize("O3,unroll-loops")

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <climits>

using namespace std;

static constexpr int N = 50;
static constexpr int M = 100;
static constexpr int S = N * N;

struct XorShift {
    unsigned int x, y, z, w;
    XorShift() {
        random_device rd;
        x = rd(); y = rd(); z = rd(); w = rd();
        if ((x | y | z | w) == 0) w = 1;
    }
    inline unsigned int next_uint() {
        unsigned int t = x;
        t ^= t << 11;
        t ^= t >> 8;
        x = y; y = z; z = w;
        w ^= w >> 19;
        w ^= t;
        return w;
    }
    inline int next_int(int n) { return (int)(next_uint() % (unsigned)n); }
    inline double next_double() {
        return (double)next_uint() * (1.0 / 4294967296.0);
    }
    using result_type = unsigned int;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT_MAX; }
    inline result_type operator()() { return next_uint(); }
};

static XorShift rng;
static chrono::high_resolution_clock::time_point start_time;

static inline double elapsed_ms() {
    return chrono::duration<double, milli>(
        chrono::high_resolution_clock::now() - start_time
    ).count();
}

static int grid[S];
static int best_grid[S];
static int pos_in_list[S];
static vector<int> cells[M + 1];

static bool required_adj[M + 1][M + 1];
static bool has_required_adj[M + 1];
static int edge_count[M + 1][M + 1];

static unsigned int seen[S];
static unsigned int seen_token = 1;
static int bfs_queue[S];

struct Delta {
    int a, b, v;
};
static Delta deltas[8];
static int delta_count;

static inline int id(int r, int c) {
    return r * N + c;
}

static inline void norm_pair(int &a, int &b) {
    if (a > b) swap(a, b);
}

static inline void set_required(int a, int b) {
    if (a == b) return;
    norm_pair(a, b);
    required_adj[a][b] = true;
}

static inline bool is_required(int a, int b) {
    if (a == b) return false;
    norm_pair(a, b);
    return required_adj[a][b];
}

static inline void add_edge(int a, int b, int v) {
    if (a == b) return;
    norm_pair(a, b);
    edge_count[a][b] += v;
}

static inline void add_cell(int p, int color) {
    pos_in_list[p] = (int)cells[color].size();
    cells[color].push_back(p);
}

static inline void remove_cell(int p, int color) {
    int idx = pos_in_list[p];
    int last = cells[color].back();
    cells[color][idx] = last;
    pos_in_list[last] = idx;
    cells[color].pop_back();
}

static inline unsigned int next_token() {
    ++seen_token;
    if (seen_token == 0) {
        memset(seen, 0, sizeof(seen));
        seen_token = 1;
    }
    return seen_token;
}

static bool connected_color(int color) {
    const vector<int> &v = cells[color];
    if (v.empty()) return true;

    unsigned int mark = next_token();
    int head = 0, tail = 0;
    bfs_queue[tail++] = v[0];
    seen[v[0]] = mark;
    int cnt = 0;

    while (head < tail) {
        int p = bfs_queue[head++];
        ++cnt;
        int r = p / N, c = p % N;

        if (r > 0) {
            int q = p - N;
            if (grid[q] == color && seen[q] != mark) {
                seen[q] = mark; bfs_queue[tail++] = q;
            }
        }
        if (r + 1 < N) {
            int q = p + N;
            if (grid[q] == color && seen[q] != mark) {
                seen[q] = mark; bfs_queue[tail++] = q;
            }
        }
        if (c > 0) {
            int q = p - 1;
            if (grid[q] == color && seen[q] != mark) {
                seen[q] = mark; bfs_queue[tail++] = q;
            }
        }
        if (c + 1 < N) {
            int q = p + 1;
            if (grid[q] == color && seen[q] != mark) {
                seen[q] = mark; bfs_queue[tail++] = q;
            }
        }
    }
    return cnt == (int)v.size();
}

static bool connected_zero_to_outside() {
    const vector<int> &v = cells[0];
    if (v.empty()) return !has_required_adj[0];

    unsigned int mark = next_token();
    int head = 0, tail = 0;

    for (int p : v) {
        int r = p / N, c = p % N;
        if (r == 0 || r == N - 1 || c == 0 || c == N - 1) {
            if (seen[p] != mark) {
                seen[p] = mark;
                bfs_queue[tail++] = p;
            }
        }
    }

    if (tail == 0) return false;

    while (head < tail) {
        int p = bfs_queue[head++];
        int r = p / N, c = p % N;

        if (r > 0) {
            int q = p - N;
            if (grid[q] == 0 && seen[q] != mark) {
                seen[q] = mark; bfs_queue[tail++] = q;
            }
        }
        if (r + 1 < N) {
            int q = p + N;
            if (grid[q] == 0 && seen[q] != mark) {
                seen[q] = mark; bfs_queue[tail++] = q;
            }
        }
        if (c > 0) {
            int q = p - 1;
            if (grid[q] == 0 && seen[q] != mark) {
                seen[q] = mark; bfs_queue[tail++] = q;
            }
        }
        if (c + 1 < N) {
            int q = p + 1;
            if (grid[q] == 0 && seen[q] != mark) {
                seen[q] = mark; bfs_queue[tail++] = q;
            }
        }
    }

    for (int p : v) {
        if (seen[p] != mark) return false;
    }
    return true;
}

static inline void add_delta(int a, int b, int value) {
    if (a == b) return;
    norm_pair(a, b);
    for (int i = 0; i < delta_count; ++i) {
        if (deltas[i].a == a && deltas[i].b == b) {
            deltas[i].v += value;
            return;
        }
    }
    deltas[delta_count++] = {a, b, value};
}

static inline void revert_move(int p, int old_color, int new_color) {
    grid[p] = old_color;
    remove_cell(p, new_color);
    add_cell(p, old_color);
    for (int i = 0; i < delta_count; ++i) {
        edge_count[deltas[i].a][deltas[i].b] -= deltas[i].v;
    }
}

static bool attempt_change(int p, int old_color, int new_color) {
    grid[p] = new_color;
    remove_cell(p, old_color);
    add_cell(p, new_color);

    delta_count = 0;
    int r = p / N, c = p % N;

    auto process_neighbor = [&](int neighbor_color) {
        add_delta(old_color, neighbor_color, -1);
        add_delta(new_color, neighbor_color, +1);
    };

    if (r == 0) process_neighbor(0);
    else process_neighbor(grid[p - N]);

    if (r == N - 1) process_neighbor(0);
    else process_neighbor(grid[p + N]);

    if (c == 0) process_neighbor(0);
    else process_neighbor(grid[p - 1]);

    if (c == N - 1) process_neighbor(0);
    else process_neighbor(grid[p + 1]);

    for (int i = 0; i < delta_count; ++i) {
        edge_count[deltas[i].a][deltas[i].b] += deltas[i].v;
    }

    bool ok = true;

    for (int i = 0; i < delta_count; ++i) {
        int a = deltas[i].a, b = deltas[i].b;
        bool now_exists = edge_count[a][b] > 0;
        if (now_exists != required_adj[a][b]) {
            ok = false;
            break;
        }
    }

    if (ok && old_color != 0) {
        if (cells[old_color].empty()) {
            if (has_required_adj[old_color]) ok = false;
        } else if (!connected_color(old_color)) {
            ok = false;
        }
    }

    // Every nonzero proposed target is an adjacent cell's color, hence it is
    // connected immediately after the recoloring. Only zero needs a full
    // outside-connectivity validation.
    if (ok && (old_color == 0 || new_color == 0)) {
        if (!connected_zero_to_outside()) ok = false;
    }

    if (!ok) {
        revert_move(p, old_color, new_color);
        return false;
    }
    return true;
}

static void initialize(const int input_grid[S]) {
    memset(required_adj, 0, sizeof(required_adj));
    memset(edge_count, 0, sizeof(edge_count));
    for (int c = 0; c <= M; ++c) cells[c].clear();

    for (int p = 0; p < S; ++p) {
        grid[p] = input_grid[p];
        add_cell(p, grid[p]);
    }

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int p = id(r, c);
            int x = input_grid[p];

            if (r == 0 || r == N - 1 || c == 0 || c == N - 1) {
                set_required(0, x);
            }
            if (r + 1 < N) set_required(x, input_grid[p + N]);
            if (c + 1 < N) set_required(x, input_grid[p + 1]);

            if (r == 0) add_edge(0, x, 1);
            if (r == N - 1) add_edge(0, x, 1);
            if (c == 0) add_edge(0, x, 1);
            if (c == N - 1) add_edge(0, x, 1);
            if (r + 1 < N) add_edge(x, input_grid[p + N], 1);
            if (c + 1 < N) add_edge(x, input_grid[p + 1], 1);
        }
    }

    for (int a = 0; a <= M; ++a) {
        has_required_adj[a] = false;
        for (int b = 0; b <= M; ++b) {
            if (a != b && is_required(a, b)) {
                has_required_adj[a] = true;
                break;
            }
        }
    }

    memcpy(best_grid, grid, sizeof(grid));
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    start_time = chrono::high_resolution_clock::now();

    int n_input, m_input;
    cin >> n_input >> m_input;

    int input_grid[S];
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            cin >> input_grid[id(r, c)];
        }
    }

    initialize(input_grid);

    int best_score = 0;
    vector<int> order(S);
    for (int i = 0; i < S; ++i) order[i] = i;

    // Revisit cells after successful erosion: removals made later in a pass
    // often unlock cells that failed when they were tested earlier.
    // Stop quickly once randomized blocks no longer produce any deletions,
    // retaining most of the time budget for non-monotone SA moves.
    const double greedy_limit = 500.0;
    const int greedy_block_size = 256;
    const int max_empty_greedy_blocks = 3;
    int empty_greedy_blocks = 0;
    bool stop_greedy = false;

    while (!stop_greedy && elapsed_ms() < greedy_limit) {
        shuffle(order.begin(), order.end(), rng);
        int block_attempts = 0;
        int block_successes = 0;

        for (int p : order) {
            if (elapsed_ms() >= greedy_limit) {
                stop_greedy = true;
                break;
            }

            int old_color = grid[p];
            if (old_color != 0 && attempt_change(p, old_color, 0)) {
                ++block_successes;
                int score = (int)cells[0].size();
                if (score > best_score) {
                    best_score = score;
                    memcpy(best_grid, grid, sizeof(grid));
                }
            }

            if (++block_attempts == greedy_block_size) {
                if (block_successes == 0) {
                    if (++empty_greedy_blocks >= max_empty_greedy_blocks) {
                        stop_greedy = true;
                        break;
                    }
                } else {
                    empty_greedy_blocks = 0;
                }
                block_attempts = 0;
                block_successes = 0;
            }
        }

        if (!stop_greedy && block_attempts > 0) {
            if (block_successes == 0) {
                if (++empty_greedy_blocks >= max_empty_greedy_blocks) stop_greedy = true;
            } else {
                empty_greedy_blocks = 0;
            }
        }
    }

    const double total_limit = 1950.0;
    const double sa_start = elapsed_ms();
    double sa_duration = total_limit - sa_start;
    if (sa_duration < 1.0) sa_duration = 1.0;

    double temperature = 2.0;
    int iteration = 0;

    while (true) {
        ++iteration;
        if ((iteration & 255) == 0) {
            double now = elapsed_ms();
            if (now >= total_limit) break;

            double progress = (now - sa_start) / sa_duration;
            if (progress < 0.0) progress = 0.0;
            if (progress > 1.0) progress = 1.0;
            temperature = 2.0 * pow(0.005, progress);
            if (temperature < 0.01) temperature = 0.01;
        }

        int p = rng.next_int(S);
        int old_color = grid[p];
        int r = p / N, c = p % N;

        int candidates[5];
        int cnt = 0;
        candidates[cnt++] = 0;
        candidates[cnt++] = (r > 0 ? grid[p - N] : 0);
        candidates[cnt++] = (r + 1 < N ? grid[p + N] : 0);
        candidates[cnt++] = (c > 0 ? grid[p - 1] : 0);
        candidates[cnt++] = (c + 1 < N ? grid[p + 1] : 0);

        int new_color = candidates[rng.next_int(cnt)];
        if (new_color == old_color) continue;

        int delta_score = (new_color == 0 ? 1 : 0) - (old_color == 0 ? 1 : 0);

        if (!attempt_change(p, old_color, new_color)) continue;

        bool accept = false;
        if (delta_score >= 0) {
            accept = true;
        } else if (rng.next_double() < exp((double)delta_score / temperature)) {
            accept = true;
        }

        if (!accept) {
            revert_move(p, old_color, new_color);
        } else if (delta_score > 0) {
            int score = (int)cells[0].size();
            if (score > best_score) {
                best_score = score;
                memcpy(best_grid, grid, sizeof(grid));
            }
        }
    }

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (c) cout << ' ';
            cout << best_grid[id(r, c)];
        }
        cout << '\n';
    }

    return 0;
}
# EVOLVE-BLOCK-END