# EVOLVE-BLOCK-START
#include <iostream>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <chrono>

constexpr int N = 10;
constexpr int CELLS = 100;
constexpr int TURNS = 100;
constexpr int FLAVORS = 3;
constexpr char DIRS[4] = {'F', 'B', 'L', 'R'};
constexpr int LOOKAHEAD = 2;
constexpr std::array<int, LOOKAHEAD> SAMPLES = {20, 8};

std::array<int, TURNS> flavor_seq;
std::array<int, FLAVORS + 1> flavor_total{};
std::array<int, FLAVORS + 1> strip_lo{};
std::array<int, FLAVORS + 1> strip_hi{};

struct XorShift {
    uint64_t x;
    XorShift() : x(static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) ^ 0x9e3779b97f4a7c15ULL) {}

    uint64_t next() {
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        return x;
    }

    int uniform(int upper) {
        return static_cast<int>(next() % static_cast<uint64_t>(upper));
    }
};

XorShift rng;

struct State {
    std::array<uint8_t, CELLS> b{};
    int occupied = 0;
    int turn = 0;

    int find_empty(int rank1) const {
        int seen = 0;
        for (int i = 0; i < CELLS; ++i) {
            if (b[i] == 0 && ++seen == rank1) return i;
        }
        return -1;
    }

    int only_empty() const {
        for (int i = 0; i < CELLS; ++i) {
            if (b[i] == 0) return i;
        }
        return -1;
    }

    void place(int pos, int flavor) {
        b[pos] = static_cast<uint8_t>(flavor);
        ++occupied;
    }

    void tilt(int dir) {
        if (dir == 0) { // F
            for (int c = 0; c < N; ++c) {
                int w = c;
                for (int r = 0; r < N; ++r) {
                    int p = r * N + c;
                    if (b[p]) {
                        uint8_t v = b[p];
                        b[p] = 0;
                        b[w] = v;
                        w += N;
                    }
                }
            }
        } else if (dir == 1) { // B
            for (int c = 0; c < N; ++c) {
                int w = (N - 1) * N + c;
                for (int r = N - 1; r >= 0; --r) {
                    int p = r * N + c;
                    if (b[p]) {
                        uint8_t v = b[p];
                        b[p] = 0;
                        b[w] = v;
                        w -= N;
                    }
                }
            }
        } else if (dir == 2) { // L
            for (int r = 0; r < N; ++r) {
                int base = r * N;
                int w = base;
                for (int c = 0; c < N; ++c) {
                    int p = base + c;
                    if (b[p]) {
                        uint8_t v = b[p];
                        b[p] = 0;
                        b[w++] = v;
                    }
                }
            }
        } else { // R
            for (int r = 0; r < N; ++r) {
                int base = r * N;
                int w = base + N - 1;
                for (int c = N - 1; c >= 0; --c) {
                    int p = base + c;
                    if (b[p]) {
                        uint8_t v = b[p];
                        b[p] = 0;
                        b[w--] = v;
                    }
                }
            }
        }
    }

    double evaluate() const {
        std::array<int, FLAVORS + 1> cnt{};
        std::array<int, FLAVORS + 1> sr{};
        std::array<int, FLAVORS + 1> sc{};
        int region_penalty = 0;
        int edge_units = 0;

        for (int p = 0; p < CELLS; ++p) {
            int f = b[p];
            if (!f) continue;
            int r = p / N;
            int c = p % N;
            ++cnt[f];
            sr[f] += r;
            sc[f] += c;

            if (c < strip_lo[f]) region_penalty += strip_lo[f] - c;
            else if (c > strip_hi[f]) region_penalty += c - strip_hi[f];
            else {
                if (r == 0 || r == N - 1) ++edge_units;
                if ((c == 0 && strip_lo[f] == 0) ||
                    (c == N - 1 && strip_hi[f] == N - 1)) {
                    ++edge_units;
                }
            }
        }

        double com_penalty = 0.0;
        for (int p = 0; p < CELLS; ++p) {
            int f = b[p];
            if (!f || cnt[f] <= 1) continue;
            int r = p / N;
            int c = p % N;
            com_penalty += std::abs(static_cast<double>(r) - static_cast<double>(sr[f]) / cnt[f]);
            com_penalty += std::abs(static_cast<double>(c) - static_cast<double>(sc[f]) / cnt[f]);
        }

        std::array<uint8_t, CELLS> used{};
        std::array<int, CELLS> queue{};
        int sum_sq = 0;

        for (int start = 0; start < CELLS; ++start) {
            int f = b[start];
            if (!f || used[start]) continue;

            int head = 0, tail = 0;
            queue[tail++] = start;
            used[start] = 1;

            while (head < tail) {
                int p = queue[head++];
                int r = p / N;
                int c = p % N;

                if (r > 0) {
                    int q = p - N;
                    if (!used[q] && b[q] == f) used[q] = 1, queue[tail++] = q;
                }
                if (r + 1 < N) {
                    int q = p + N;
                    if (!used[q] && b[q] == f) used[q] = 1, queue[tail++] = q;
                }
                if (c > 0) {
                    int q = p - 1;
                    if (!used[q] && b[q] == f) used[q] = 1, queue[tail++] = q;
                }
                if (c + 1 < N) {
                    int q = p + 1;
                    if (!used[q] && b[q] == f) used[q] = 1, queue[tail++] = q;
                }
            }
            sum_sq += tail * tail;
        }

        double t = static_cast<double>(turn);
        double a = 15.0 + 1.1 * t;
        double bp = std::max(0.0, 170.0 - 1.7 * t);
        double cp = std::max(2.0, 27.0 - 0.17 * t);
        double d = 5.0 + 0.2 * t;

        return a * sum_sq - bp * com_penalty - cp * region_penalty + d * (0.5 * edge_units);
    }
};

double search_value(const State& state, int depth) {
    if (depth == 0 || state.turn == TURNS || state.occupied == CELLS) {
        return state.evaluate();
    }

    const int empty = CELLS - state.occupied;
    const int next_flavor = flavor_seq[state.turn];

    if (empty == 1) {
        State next = state;
        next.place(next.only_empty(), next_flavor);
        next.turn = state.turn + 1;
        return next.evaluate();
    }

    int sample_count = std::min(SAMPLES[LOOKAHEAD - depth], empty);
    double total = 0.0;

    for (int s = 0; s < sample_count; ++s) {
        int rank;
        if (sample_count == empty) {
            rank = s + 1;
        } else {
            rank = rng.uniform(empty) + 1;
        }

        State placed = state;
        placed.place(placed.find_empty(rank), next_flavor);
        placed.turn = state.turn + 1;

        double best = -std::numeric_limits<double>::infinity();
        for (int dir = 0; dir < 4; ++dir) {
            State moved = placed;
            moved.tilt(dir);
            double value = search_value(moved, depth - 1);
            if (value > best) best = value;
        }
        total += best;
    }

    return total / sample_count;
}

int choose_direction(const State& state) {
    double best_value = -std::numeric_limits<double>::infinity();
    int best_dir = 0;

    for (int dir = 0; dir < 4; ++dir) {
        State next = state;
        next.tilt(dir);
        double value = search_value(next, LOOKAHEAD);
        if (value > best_value) {
            best_value = value;
            best_dir = dir;
        }
    }
    return best_dir;
}

void initialize() {
    for (int i = 0; i < TURNS; ++i) {
        std::cin >> flavor_seq[i];
        ++flavor_total[flavor_seq[i]];
    }

    std::array<int, FLAVORS> order = {1, 2, 3};
    std::sort(order.begin(), order.end(), [](int a, int b) {
        if (flavor_total[a] != flavor_total[b]) return flavor_total[a] > flavor_total[b];
        return a < b;
    });

    std::array<int, FLAVORS + 1> width{};
    int assigned = 0;
    for (int f = 1; f <= FLAVORS; ++f) {
        width[f] = N * flavor_total[f] / TURNS;
        assigned += width[f];
    }

    for (int i = 0; assigned < N; ++i, ++assigned) {
        ++width[order[i % FLAVORS]];
    }

    int col = 0;
    for (int f : order) {
        strip_lo[f] = col;
        strip_hi[f] = col + width[f] - 1;
        col += width[f];
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    initialize();

    State current;
    for (int t = 0; t < TURNS; ++t) {
        int p;
        std::cin >> p;

        current.turn = t + 1;
        current.place(current.find_empty(p), flavor_seq[t]);

        int dir;
        if (t == TURNS - 1) {
            dir = 0;
        } else {
            dir = choose_direction(current);
        }

        std::cout << DIRS[dir] << std::endl;
        current.tilt(dir);
    }

    return 0;
}
# EVOLVE-BLOCK-END
