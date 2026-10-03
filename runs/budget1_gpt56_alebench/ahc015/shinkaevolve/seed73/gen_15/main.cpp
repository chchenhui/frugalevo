# EVOLVE-BLOCK-START
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace std;

namespace {

constexpr int N = 10;
constexpr int CELLS = 100;
constexpr int FLAVORS = 3;
constexpr int DIRS = 4;
constexpr array<char, DIRS> DIR_CHAR = {'F', 'B', 'L', 'R'};
constexpr array<int, 2> SAMPLE_COUNT = {20, 7};

struct Random {
    uint64_t state;

    Random()
        : state(static_cast<uint64_t>(
              chrono::steady_clock::now().time_since_epoch().count()) | 1ULL) {}

    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    int range(int lo, int hi) {
        if (lo >= hi) return lo;
        return lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1));
    }
};

Random rng;

struct LayoutPlan {
    array<int, FLAVORS + 1> total{};
    array<int, FLAVORS + 1> left{};
    array<int, FLAVORS + 1> right{};

    void build(const array<int, CELLS>& flavor) {
        total.fill(0);
        left.fill(0);
        right.fill(-1);

        for (int x : flavor) ++total[x];

        array<int, FLAVORS> order = {1, 2, 3};
        sort(order.begin(), order.end(), [&](int a, int b) {
            if (total[a] != total[b]) return total[a] > total[b];
            return a < b;
        });

        array<int, FLAVORS + 1> width{};
        int used = 0;
        for (int f = 1; f <= FLAVORS; ++f) {
            width[f] = N * total[f] / CELLS;
            used += width[f];
        }
        for (int i = 0; used < N; ++i, ++used) {
            ++width[order[i % FLAVORS]];
        }

        int cursor = 0;
        for (int f : order) {
            left[f] = cursor;
            right[f] = cursor + width[f] - 1;
            cursor += width[f];
        }
    }
};

struct Board {
    array<uint8_t, CELLS> cell{};
    int count = 0;

    int emptyCount() const {
        return CELLS - count;
    }

    int nthEmpty(int rank) const {
        int found = 0;
        for (int p = 0; p < CELLS; ++p) {
            if (cell[p] == 0 && ++found == rank) return p;
        }
        return -1;
    }

    void placeAt(int p, int flavor) {
        cell[p] = static_cast<uint8_t>(flavor);
        ++count;
    }

    void tilt(int dir) {
        array<uint8_t, CELLS> next{};

        if (dir == 0 || dir == 1) {
            for (int c = 0; c < N; ++c) {
                int write = (dir == 0 ? 0 : N - 1);
                int step = (dir == 0 ? 1 : -1);
                for (int k = 0; k < N; ++k) {
                    int r = (dir == 0 ? k : N - 1 - k);
                    uint8_t x = cell[r * N + c];
                    if (x != 0) {
                        next[write * N + c] = x;
                        write += step;
                    }
                }
            }
        } else {
            for (int r = 0; r < N; ++r) {
                int write = (dir == 2 ? 0 : N - 1);
                int step = (dir == 2 ? 1 : -1);
                for (int k = 0; k < N; ++k) {
                    int c = (dir == 2 ? k : N - 1 - k);
                    uint8_t x = cell[r * N + c];
                    if (x != 0) {
                        next[r * N + write] = x;
                        write += step;
                    }
                }
            }
        }

        cell = next;
    }
};

struct Evaluator {
    const LayoutPlan& plan;

    long long componentSquares(const Board& board) const {
        array<uint8_t, CELLS> visited{};
        array<int, CELLS> queue{};
        long long result = 0;

        for (int start = 0; start < CELLS; ++start) {
            uint8_t flavor = board.cell[start];
            if (flavor == 0 || visited[start]) continue;

            int head = 0;
            int tail = 0;
            int size = 0;
            queue[tail++] = start;
            visited[start] = 1;

            while (head < tail) {
                int v = queue[head++];
                ++size;
                int r = v / N;
                int c = v % N;

                if (r > 0) {
                    int to = v - N;
                    if (!visited[to] && board.cell[to] == flavor) {
                        visited[to] = 1;
                        queue[tail++] = to;
                    }
                }
                if (r + 1 < N) {
                    int to = v + N;
                    if (!visited[to] && board.cell[to] == flavor) {
                        visited[to] = 1;
                        queue[tail++] = to;
                    }
                }
                if (c > 0) {
                    int to = v - 1;
                    if (!visited[to] && board.cell[to] == flavor) {
                        visited[to] = 1;
                        queue[tail++] = to;
                    }
                }
                if (c + 1 < N) {
                    int to = v + 1;
                    if (!visited[to] && board.cell[to] == flavor) {
                        visited[to] = 1;
                        queue[tail++] = to;
                    }
                }
            }

            result += 1LL * size * size;
        }

        return result;
    }

    double score(const Board& board) const {
        const long long components = componentSquares(board);

        if (board.count == CELLS) {
            return static_cast<double>(components) * 10000.0;
        }

        array<int, FLAVORS + 1> cnt{};
        array<int, FLAVORS + 1> sumR{};
        array<int, FLAVORS + 1> sumC{};

        double stripPenalty = 0.0;
        double edgeBonus = 0.0;

        for (int p = 0; p < CELLS; ++p) {
            int f = board.cell[p];
            if (f == 0) continue;

            int r = p / N;
            int c = p % N;
            ++cnt[f];
            sumR[f] += r;
            sumC[f] += c;

            if (c < plan.left[f]) {
                stripPenalty += plan.left[f] - c;
            } else if (c > plan.right[f]) {
                stripPenalty += c - plan.right[f];
            } else {
                if (r == 0 || r == N - 1) edgeBonus += 0.5;
                if ((c == 0 && plan.left[f] == 0) ||
                    (c == N - 1 && plan.right[f] == N - 1)) {
                    edgeBonus += 0.5;
                }
            }
        }

        double centerPenalty = 0.0;
        for (int p = 0; p < CELLS; ++p) {
            int f = board.cell[p];
            if (f == 0 || cnt[f] <= 1) continue;

            int r = p / N;
            int c = p % N;
            centerPenalty += abs(r - static_cast<double>(sumR[f]) / cnt[f]);
            centerPenalty += abs(c - static_cast<double>(sumC[f]) / cnt[f]);
        }

        const double turn = static_cast<double>(board.count);
        const double connectivityWeight = 15.0 + 1.1 * turn;
        const double centerWeight = max(0.0, 170.0 - 1.7 * turn);
        const double stripWeight = max(2.0, 27.0 - 0.17 * turn);
        const double edgeWeight = 5.0 + 0.2 * turn;

        return connectivityWeight * components
             - centerWeight * centerPenalty
             - stripWeight * stripPenalty
             + edgeWeight * edgeBonus;
    }
};

struct ExpectimaxPolicy {
    const array<int, CELLS>& flavor;
    const Evaluator& evaluator;

    double forecast(const Board& state, int depth) const {
        if (depth == 0 || state.count >= CELLS) {
            return evaluator.score(state);
        }

        const int empty = state.emptyCount();
        const int level = 2 - depth;
        const int samples = min(empty, SAMPLE_COUNT[level]);
        const int nextFlavor = flavor[state.count];

        array<int, CELLS> ranks{};
        for (int i = 0; i < empty; ++i) ranks[i] = i + 1;

        double sum = 0.0;

        for (int s = 0; s < samples; ++s) {
            int chosen = rng.range(s, empty - 1);
            swap(ranks[s], ranks[chosen]);

            Board afterPlacement = state;
            afterPlacement.placeAt(afterPlacement.nthEmpty(ranks[s]), nextFlavor);

            double best = -numeric_limits<double>::infinity();
            for (int dir = 0; dir < DIRS; ++dir) {
                Board child = afterPlacement;
                child.tilt(dir);
                best = max(best, forecast(child, depth - 1));
            }
            sum += best;
        }

        return sum / samples;
    }

    int choose(const Board& afterPlacement) const {
        int bestDir = 0;
        double bestValue = -numeric_limits<double>::infinity();

        for (int dir = 0; dir < DIRS; ++dir) {
            Board child = afterPlacement;
            child.tilt(dir);
            double value = forecast(child, 2);

            if (value > bestValue) {
                bestValue = value;
                bestDir = dir;
            }
        }

        return bestDir;
    }
};

}  // namespace

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    array<int, CELLS> flavor{};
    for (int i = 0; i < CELLS; ++i) cin >> flavor[i];

    LayoutPlan plan;
    plan.build(flavor);

    Evaluator evaluator{plan};
    ExpectimaxPolicy policy{flavor, evaluator};
    Board current;

    for (int turn = 0; turn < CELLS; ++turn) {
        int placementRank;
        cin >> placementRank;

        current.placeAt(current.nthEmpty(placementRank), flavor[turn]);

        int direction = policy.choose(current);
        cout << DIR_CHAR[direction] << '\n' << flush;

        current.tilt(direction);
    }

    return 0;
}
# EVOLVE-BLOCK-END
