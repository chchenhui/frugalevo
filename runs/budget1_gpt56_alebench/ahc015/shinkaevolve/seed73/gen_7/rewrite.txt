# EVOLVE-BLOCK-START
#include <iostream>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <cstdint>

constexpr int GRID_SIZE = 10;
constexpr int CELLS = 100;
constexpr int NUM_TURNS = 100;
constexpr int NUM_FLAVORS = 3;
constexpr int NUM_DIRECTIONS = 4;

constexpr char DIR_CHARS[NUM_DIRECTIONS] = {'F', 'B', 'L', 'R'};
constexpr std::array<int, 2> NUM_SAMPLES_CONFIG = {22, 9};

std::array<int, NUM_TURNS> G_FLAVOR_SEQUENCE;
std::array<int, NUM_FLAVORS + 1> G_flavor_total_counts;
std::array<std::pair<int, int>, NUM_FLAVORS + 1> G_target_col_ranges;

struct XorshiftRNG {
    uint64_t x;

    XorshiftRNG()
        : x(static_cast<uint64_t>(
              std::chrono::steady_clock::now().time_since_epoch().count()) ^
            0x9e3779b97f4a7c15ULL) {}

    uint64_t next() {
        x ^= x << 7;
        x ^= x >> 9;
        return x;
    }

    int uniform_int(int lo, int hi) {
        if (lo >= hi) return lo;
        return lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1));
    }
};

XorshiftRNG rng;

struct GameState {
    std::array<uint8_t, CELLS> board{};
    int turn = 0;
    int candy_count = 0;

    int find_pth_empty_cell(int p) const {
        for (int i = 0; i < CELLS; ++i) {
            if (board[i] == 0 && --p == 0) return i;
        }
        return -1;
    }

    void place_candy(int pos, int flavor) {
        board[pos] = static_cast<uint8_t>(flavor);
        ++candy_count;
    }

    void apply_tilt(int dir) {
        if (dir == 0 || dir == 1) {
            const bool reverse = (dir == 1);
            for (int c = 0; c < GRID_SIZE; ++c) {
                uint8_t line[GRID_SIZE];
                int cnt = 0;

                if (!reverse) {
                    for (int r = 0; r < GRID_SIZE; ++r) {
                        uint8_t v = board[r * GRID_SIZE + c];
                        if (v) line[cnt++] = v;
                    }
                    for (int r = 0; r < GRID_SIZE; ++r) {
                        board[r * GRID_SIZE + c] = (r < cnt ? line[r] : 0);
                    }
                } else {
                    for (int r = GRID_SIZE - 1; r >= 0; --r) {
                        uint8_t v = board[r * GRID_SIZE + c];
                        if (v) line[cnt++] = v;
                    }
                    for (int r = GRID_SIZE - 1, k = 0; r >= 0; --r, ++k) {
                        board[r * GRID_SIZE + c] = (k < cnt ? line[k] : 0);
                    }
                }
            }
        } else {
            const bool reverse = (dir == 3);
            for (int r = 0; r < GRID_SIZE; ++r) {
                uint8_t line[GRID_SIZE];
                int cnt = 0;
                int base = r * GRID_SIZE;

                if (!reverse) {
                    for (int c = 0; c < GRID_SIZE; ++c) {
                        uint8_t v = board[base + c];
                        if (v) line[cnt++] = v;
                    }
                    for (int c = 0; c < GRID_SIZE; ++c) {
                        board[base + c] = (c < cnt ? line[c] : 0);
                    }
                } else {
                    for (int c = GRID_SIZE - 1; c >= 0; --c) {
                        uint8_t v = board[base + c];
                        if (v) line[cnt++] = v;
                    }
                    for (int c = GRID_SIZE - 1, k = 0; c >= 0; --c, ++k) {
                        board[base + c] = (k < cnt ? line[k] : 0);
                    }
                }
            }
        }
    }

    double evaluate() const {
        std::array<uint8_t, CELLS> visited{};
        std::array<int, CELLS> queue{};
        long long component_sq_sum = 0;

        for (int start = 0; start < CELLS; ++start) {
            const uint8_t flavor = board[start];
            if (flavor == 0 || visited[start]) continue;

            int head = 0, tail = 0;
            queue[tail++] = start;
            visited[start] = 1;
            int size = 0;

            while (head < tail) {
                int v = queue[head++];
                ++size;
                int r = v / GRID_SIZE;
                int c = v % GRID_SIZE;

                if (r > 0) {
                    int to = v - GRID_SIZE;
                    if (!visited[to] && board[to] == flavor) {
                        visited[to] = 1;
                        queue[tail++] = to;
                    }
                }
                if (r + 1 < GRID_SIZE) {
                    int to = v + GRID_SIZE;
                    if (!visited[to] && board[to] == flavor) {
                        visited[to] = 1;
                        queue[tail++] = to;
                    }
                }
                if (c > 0) {
                    int to = v - 1;
                    if (!visited[to] && board[to] == flavor) {
                        visited[to] = 1;
                        queue[tail++] = to;
                    }
                }
                if (c + 1 < GRID_SIZE) {
                    int to = v + 1;
                    if (!visited[to] && board[to] == flavor) {
                        visited[to] = 1;
                        queue[tail++] = to;
                    }
                }
            }
            component_sq_sum += 1LL * size * size;
        }

        std::array<int, NUM_FLAVORS + 1> count{};
        std::array<int, NUM_FLAVORS + 1> sum_r{};
        std::array<int, NUM_FLAVORS + 1> sum_c{};

        for (int pos = 0; pos < CELLS; ++pos) {
            int f = board[pos];
            if (!f) continue;
            ++count[f];
            sum_r[f] += pos / GRID_SIZE;
            sum_c[f] += pos % GRID_SIZE;
        }

        double com_penalty = 0.0;
        double region_penalty = 0.0;
        double edge_bonus = 0.0;

        for (int pos = 0; pos < CELLS; ++pos) {
            int f = board[pos];
            if (!f) continue;

            int r = pos / GRID_SIZE;
            int c = pos % GRID_SIZE;

            if (count[f] > 1) {
                double cr = static_cast<double>(sum_r[f]) / count[f];
                double cc = static_cast<double>(sum_c[f]) / count[f];
                com_penalty += std::abs(r - cr) + std::abs(c - cc);
            }

            const auto [left, right] = G_target_col_ranges[f];
            if (c < left) {
                region_penalty += left - c;
            } else if (c > right) {
                region_penalty += c - right;
            } else {
                if (r == 0 || r == GRID_SIZE - 1) edge_bonus += 0.5;
                if ((c == 0 && left == 0) ||
                    (c == GRID_SIZE - 1 && right == GRID_SIZE - 1)) {
                    edge_bonus += 0.5;
                }
            }
        }

        const double t = static_cast<double>(turn);
        const double conn_coeff = 15.0 + 1.1 * t;
        const double com_coeff = std::max(0.0, 170.0 - 1.7 * t);
        const double region_coeff = std::max(2.0, 27.0 - 0.17 * t);
        const double edge_coeff = 5.0 + 0.2 * t;

        return conn_coeff * component_sq_sum
             - com_coeff * com_penalty
             - region_coeff * region_penalty
             + edge_coeff * edge_bonus;
    }
};

double eval_lookahead(const GameState& state, int depth) {
    if (depth == 0 || state.turn == NUM_TURNS) {
        return state.evaluate();
    }

    const int empty_count = CELLS - state.candy_count;
    if (empty_count == 0) return state.evaluate();

    const int level = 2 - depth;
    const int samples = std::min(NUM_SAMPLES_CONFIG[level], empty_count);
    std::array<int, CELLS> ranks;

    for (int i = 0; i < empty_count; ++i) ranks[i] = i + 1;

    double total = 0.0;
    const int next_flavor = G_FLAVOR_SEQUENCE[state.turn];

    for (int s = 0; s < samples; ++s) {
        int chosen = rng.uniform_int(s, empty_count - 1);
        std::swap(ranks[s], ranks[chosen]);

        GameState after_place = state;
        int pos = after_place.find_pth_empty_cell(ranks[s]);
        after_place.place_candy(pos, next_flavor);
        ++after_place.turn;

        double best = -std::numeric_limits<double>::infinity();
        for (int dir = 0; dir < NUM_DIRECTIONS; ++dir) {
            GameState next = after_place;
            next.apply_tilt(dir);
            best = std::max(best, eval_lookahead(next, depth - 1));
        }
        total += best;
    }

    return total / samples;
}

int decide_direction(const GameState& state_after_placement) {
    double best_value = -std::numeric_limits<double>::infinity();
    int best_dir = 0;

    for (int dir = 0; dir < NUM_DIRECTIONS; ++dir) {
        GameState next = state_after_placement;
        next.apply_tilt(dir);
        double value = eval_lookahead(next, 2);
        if (value > best_value) {
            best_value = value;
            best_dir = dir;
        }
    }
    return best_dir;
}

void initialize_global_data() {
    G_flavor_total_counts.fill(0);

    for (int i = 0; i < NUM_TURNS; ++i) {
        std::cin >> G_FLAVOR_SEQUENCE[i];
        ++G_flavor_total_counts[G_FLAVOR_SEQUENCE[i]];
    }

    std::array<int, NUM_FLAVORS> order = {1, 2, 3};
    std::sort(order.begin(), order.end(), [](int a, int b) {
        if (G_flavor_total_counts[a] != G_flavor_total_counts[b]) {
            return G_flavor_total_counts[a] > G_flavor_total_counts[b];
        }
        return a < b;
    });

    std::array<int, NUM_FLAVORS + 1> width{};
    int used = 0;

    for (int f = 1; f <= NUM_FLAVORS; ++f) {
        width[f] = GRID_SIZE * G_flavor_total_counts[f] / NUM_TURNS;
        used += width[f];
    }

    for (int i = 0; used < GRID_SIZE; ++i, ++used) {
        ++width[order[i % NUM_FLAVORS]];
    }

    int col = 0;
    for (int f : order) {
        G_target_col_ranges[f] = {col, col + width[f] - 1};
        col += width[f];
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    initialize_global_data();

    GameState current;

    for (int t = 0; t < NUM_TURNS; ++t) {
        int p;
        std::cin >> p;

        int pos = current.find_pth_empty_cell(p);
        current.place_candy(pos, G_FLAVOR_SEQUENCE[t]);
        current.turn = t + 1;

        int dir = decide_direction(current);
        std::cout << DIR_CHARS[dir] << std::endl;

        current.apply_tilt(dir);
    }

    return 0;
}
# EVOLVE-BLOCK-END