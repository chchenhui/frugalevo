# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

static int M, N, L;
static double eps;
static bool use_canonical = false;

static vector<uint16_t> codes;
static vector<int> edge_counts;
static vector<double> expected_counts;
static unordered_map<uint16_t, int> owner;
static vector<array<int, 6>> permutations;

static const uint16_t CANONICAL_N6[] = {
    0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 27, 29, 31,
    37, 39, 43, 45, 47, 53, 55, 61, 63, 73, 75, 77, 79, 91, 93,
    95, 111, 117, 119, 125, 127, 141, 143, 157, 159, 173, 175,
    181, 183, 189, 191, 205, 207, 221, 223, 237, 239, 253, 255,
    285, 287, 315, 317, 319, 349, 351, 379, 381, 383, 413, 415,
    445, 447, 477, 479, 509, 511, 565, 567, 573, 575, 589, 591,
    605, 607, 637, 639, 701, 703, 717, 719, 733, 735, 749, 751,
    765, 767, 797, 799, 829, 831, 861, 863, 893, 895, 957, 959,
    989, 991, 1021, 1023, 1149, 1151, 1213, 1215, 1245, 1247,
    1277, 1279, 1533, 1535, 1661, 1663, 1789, 1791, 1917, 1919,
    2045, 2047, 2109, 2111, 2141, 2143, 2173, 2175, 2205, 2207,
    2237, 2239, 2269, 2271, 2301, 2303, 2685, 2687, 2813, 2815,
    2941, 2943, 3069, 3071, 3277, 3279, 3285, 3287, 3293, 3295,
    3309, 3311, 3325, 3327, 3357, 3359, 3389, 3391, 3421, 3423,
    3453, 3455, 3517, 3519, 3549, 3551, 3581, 3583, 3613, 3615,
    3645, 3647, 3709, 3711, 3773, 3775, 3837, 3839, 4095, 8191,
    16383, 32767
};

static inline int edge_index(int a, int b) {
    if (a > b) swap(a, b);
    return a * (2 * N - a - 1) / 2 + (b - a - 1);
}

static uint16_t permute_mask(uint16_t mask, const array<int, 6>& p) {
    uint16_t result = 0;
    int bit = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j, ++bit) {
            int source_bit = edge_index(p[i], p[j]);
            if ((mask >> source_bit) & 1U) result |= uint16_t(1U << bit);
        }
    }
    return result;
}

static uint16_t canonical_mask(uint16_t mask) {
    uint16_t best = numeric_limits<uint16_t>::max();
    for (const auto& p : permutations) {
        best = min(best, permute_mask(mask, p));
    }
    return best;
}

static uint16_t string_to_mask(const string& s) {
    uint16_t mask = 0;
    for (int i = 0; i < L; ++i) {
        if (s[i] == '1') mask |= uint16_t(1U << i);
    }
    return mask;
}

static string mask_to_string(uint16_t mask) {
    string result(L, '0');
    for (int i = 0; i < L; ++i) {
        if ((mask >> i) & 1U) result[i] = '1';
    }
    return result;
}

static void build_permutations() {
    permutations.clear();
    array<int, 6> p{};
    for (int i = 0; i < N; ++i) p[i] = i;
    do {
        permutations.push_back(p);
    } while (next_permutation(p.begin(), p.begin() + N));
}

static void build_canonical_codes() {
    build_permutations();

    vector<uint16_t> representatives;
    if (N == 6) {
        representatives.assign(
            begin(CANONICAL_N6),
            end(CANONICAL_N6)
        );
    } else {
        vector<char> seen(1 << L, false);
        for (int mask = 0; mask < (1 << L); ++mask) {
            uint16_t c = canonical_mask(uint16_t(mask));
            if (!seen[c]) {
                seen[c] = true;
                representatives.push_back(c);
            }
        }
        sort(representatives.begin(), representatives.end());
    }

    codes.clear();
    owner.clear();
    for (int i = 0; i < M; ++i) {
        uint16_t code = representatives[i];
        codes.push_back(code);
        owner[code] = i;
    }
}

static void build_edge_codec() {
    edge_counts.resize(M);
    expected_counts.resize(M);

    for (int i = 0; i < M; ++i) {
        edge_counts[i] = int(llround(double(i) * L / double(M - 1)));
    }

    for (int i = 1; i < M; ++i) {
        edge_counts[i] = max(edge_counts[i], edge_counts[i - 1] + 1);
    }

    if (edge_counts.back() > L) {
        int excess = edge_counts.back() - L;
        for (int& x : edge_counts) x -= excess;
    }

    const double slope = 1.0 - 2.0 * eps;
    for (int i = 0; i < M; ++i) {
        edge_counts[i] = max(0, min(L, edge_counts[i]));
        expected_counts[i] = edge_counts[i] * slope + L * eps;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> M >> eps)) return 0;

    int n_ged;
    if (M <= 11) n_ged = 4;
    else if (M <= 34) n_ged = 5;
    else n_ged = 6;

    constexpr double K_SEP = 2.5;
    const double denom = (0.5 - eps) * (0.5 - eps);

    double ideal_l;
    if (denom > 1e-15) {
        ideal_l = K_SEP * K_SEP * (M - 1.0) * (M - 1.0)
                * eps * (1.0 - eps) / denom;
    } else {
        ideal_l = 4950.0;
    }

    int n_ec = 4;
    if (ideal_l > 1e-12) {
        n_ec = int(ceil((1.0 + sqrt(1.0 + 8.0 * ideal_l)) * 0.5));
    }
    n_ec = max(4, min(100, n_ec));

    // Input epsilon is a multiple of 0.01, so this branch is exactly eps=0.
    if (eps < 0.005) {
        use_canonical = true;
        N = n_ged;
    } else {
        use_canonical = false;
        N = max(n_ec, n_ged + 1);
        N = min(N, 100);
    }

    L = N * (N - 1) / 2;

    cout << N << '\n';

    if (use_canonical) {
        build_canonical_codes();
        for (uint16_t code : codes) {
            cout << mask_to_string(code) << '\n';
        }
    } else {
        build_edge_codec();
        for (int i = 0; i < M; ++i) {
            string graph(L, '0');
            for (int j = 0; j < edge_counts[i]; ++j) graph[j] = '1';
            cout << graph << '\n';
        }
    }
    cout.flush();

    for (int q = 0; q < 100; ++q) {
        string received;
        cin >> received;

        int answer = 0;

        if (use_canonical) {
            uint16_t key = canonical_mask(string_to_mask(received));
            auto it = owner.find(key);
            if (it != owner.end()) answer = it->second;
        } else {
            int observed = int(count(received.begin(), received.end(), '1'));

            auto it = lower_bound(
                expected_counts.begin(),
                expected_counts.end(),
                double(observed)
            );

            if (it == expected_counts.begin()) {
                answer = 0;
            } else if (it == expected_counts.end()) {
                answer = M - 1;
            } else {
                int right = int(it - expected_counts.begin());
                int left = right - 1;
                answer = (fabs(observed - expected_counts[left]) <=
                          fabs(observed - expected_counts[right]))
                    ? left : right;
            }
        }

        cout << answer << '\n';
        cout.flush();
    }

    return 0;
}
# EVOLVE-BLOCK-END