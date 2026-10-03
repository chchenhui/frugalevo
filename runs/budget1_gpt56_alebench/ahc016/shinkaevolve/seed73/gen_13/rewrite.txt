# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

static int M, N, L;
static double eps;

static vector<vector<int>> permutations_list;
static vector<unsigned short> canonical_masks;
static vector<unsigned short> selected_masks;
static vector<int> edge_counts;

static int bit_index[6][6];

static unsigned short permute_mask(unsigned short mask, const vector<int>& p) {
    unsigned short result = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            int old_bit = bit_index[p[i]][p[j]];
            if (mask & (1u << old_bit)) result |= (1u << bit_index[i][j]);
        }
    }
    return result;
}

static unsigned short canonical(unsigned short mask) {
    unsigned short best = numeric_limits<unsigned short>::max();
    for (const auto& p : permutations_list) {
        best = min(best, permute_mask(mask, p));
    }
    return best;
}

static int graph_distance(unsigned short a, unsigned short b) {
    int best = L + 1;
    for (const auto& p : permutations_list) {
        int d = __builtin_popcount((unsigned)(a ^ permute_mask(b, p)));
        if (d < best) best = d;
        if (best == 0) break;
    }
    return best;
}

static unsigned short string_to_mask(const string& s) {
    unsigned short mask = 0;
    for (int i = 0; i < L; ++i) {
        if (s[i] == '1') mask |= (1u << i);
    }
    return mask;
}

static string mask_to_string(unsigned short mask) {
    string s(L, '0');
    for (int i = 0; i < L; ++i) {
        if (mask & (1u << i)) s[i] = '1';
    }
    return s;
}

static vector<int> make_counts(int n, int m) {
    int l = n * (n - 1) / 2;
    vector<int> c(m);
    for (int i = 0; i < m; ++i) {
        c[i] = (int)llround((double)i * l / (m - 1));
    }
    for (int i = 1; i < m; ++i) c[i] = max(c[i], c[i - 1] + 1);
    if (c.back() > l) {
        int excess = c.back() - l;
        for (int& x : c) x -= excess;
    }
    for (int i = 0; i < m; ++i) c[i] = max(0, min(l, c[i]));
    return c;
}

static int choose_density_n() {
    const double slope = 1.0 - 2.0 * eps;
    const double var_factor = eps * (1.0 - eps);
    const double sq2 = sqrt(2.0);

    int best_n = 4;
    double best_score = -1.0;

    for (int n = 4; n <= 100; ++n) {
        int l = n * (n - 1) / 2;
        if (l < M - 1) continue;

        vector<int> counts = make_counts(n, M);
        double estimated_error = 0.0;

        if (var_factor > 1e-15) {
            double sigma = sqrt(l * var_factor);
            for (int i = 0; i < M; ++i) {
                double mu = eps * l + slope * counts[i];
                double lo = -numeric_limits<double>::infinity();
                double hi = numeric_limits<double>::infinity();

                if (i > 0) {
                    double prev = eps * l + slope * counts[i - 1];
                    lo = ((prev + mu) * 0.5 - mu) / sigma;
                }
                if (i + 1 < M) {
                    double next = eps * l + slope * counts[i + 1];
                    hi = ((next + mu) * 0.5 - mu) / sigma;
                }

                double ok = 0.5 * (erf(hi / sq2) - erf(lo / sq2));
                ok = max(0.0, min(1.0, ok));
                estimated_error += 1.0 - ok;
            }
            estimated_error /= M;
        }

        double expected_score = pow(0.9, 100.0 * estimated_error) / n;
        if (expected_score > best_score) {
            best_score = expected_score;
            best_n = n;
        }
    }
    return best_n;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> M >> eps;

    int ged_n;
    if (M <= 11) ged_n = 4;
    else if (M <= 34) ged_n = 5;
    else ged_n = 6;

    int density_n = choose_density_n();
    bool use_ged = (eps < 0.01) || (density_n <= ged_n && eps <= 0.03);

    if (use_ged) {
        N = ged_n;
        L = N * (N - 1) / 2;

        memset(bit_index, -1, sizeof(bit_index));
        int ptr = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                bit_index[i][j] = bit_index[j][i] = ptr++;
            }
        }

        vector<int> p(N);
        iota(p.begin(), p.end(), 0);
        do {
            permutations_list.push_back(p);
        } while (next_permutation(p.begin(), p.end()));

        vector<char> seen(1 << L, 0);
        for (int mask = 0; mask < (1 << L); ++mask) {
            unsigned short c = canonical((unsigned short)mask);
            if (!seen[c]) {
                seen[c] = 1;
                canonical_masks.push_back(c);
            }
        }
        sort(canonical_masks.begin(), canonical_masks.end());

        vector<int> min_dist(canonical_masks.size(), L + 1);
        vector<char> used(canonical_masks.size(), 0);

        int first = 0;
        selected_masks.push_back(canonical_masks[first]);
        used[first] = 1;

        for (int i = 0; i < (int)canonical_masks.size(); ++i) {
            min_dist[i] = graph_distance(canonical_masks[i], canonical_masks[first]);
        }

        while ((int)selected_masks.size() < M) {
            int best = -1;
            int best_value = -1;
            for (int i = 0; i < (int)canonical_masks.size(); ++i) {
                if (!used[i] && min_dist[i] > best_value) {
                    best_value = min_dist[i];
                    best = i;
                }
            }
            if (best < 0) break;

            used[best] = 1;
            selected_masks.push_back(canonical_masks[best]);
            for (int i = 0; i < (int)canonical_masks.size(); ++i) {
                if (!used[i]) {
                    min_dist[i] = min(min_dist[i],
                                      graph_distance(canonical_masks[i], canonical_masks[best]));
                }
            }
        }

        while ((int)selected_masks.size() < M) selected_masks.push_back(selected_masks[0]);

        cout << N << '\n';
        for (unsigned short mask : selected_masks) {
            cout << mask_to_string(mask) << '\n';
        }
        cout.flush();

        for (int q = 0; q < 100; ++q) {
            string h;
            cin >> h;
            unsigned short received = string_to_mask(h);

            int answer = 0;
            int best_dist = L + 1;
            for (int i = 0; i < M; ++i) {
                int d = graph_distance(received, selected_masks[i]);
                if (d < best_dist) {
                    best_dist = d;
                    answer = i;
                }
            }
            cout << answer << '\n';
            cout.flush();
        }
    } else {
        N = density_n;
        L = N * (N - 1) / 2;
        edge_counts = make_counts(N, M);

        cout << N << '\n';
        for (int i = 0; i < M; ++i) {
            string g(L, '0');
            for (int j = 0; j < edge_counts[i]; ++j) g[j] = '1';
            cout << g << '\n';
        }
        cout.flush();

        const double slope = 1.0 - 2.0 * eps;
        for (int q = 0; q < 100; ++q) {
            string h;
            cin >> h;
            int observed = (int)count(h.begin(), h.end(), '1');

            int answer = 0;
            double best = 1e100;
            for (int i = 0; i < M; ++i) {
                double expected = eps * L + slope * edge_counts[i];
                double diff = abs(observed - expected);
                if (diff < best) {
                    best = diff;
                    answer = i;
                }
            }

            cout << answer << '\n';
            cout.flush();
        }
    }

    return 0;
}
# EVOLVE-BLOCK-END