# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

/* Doc: Encode with deterministic block patterns. Signature = sorted expected noisy degrees + expected edge count.
   Select patterns via farthest-first in this signature space; decode by nearest neighbor on the same signature. */

/* Squared Euclidean distance between equal-length vectors */
static inline double sse(const vector<double>& a, const vector<double>& b) {
    double s = 0.0; int n = (int)a.size();
    for (int i = 0; i < n; ++i) { double d = a[i] - b[i]; s += d * d; }
    return s;
}

/* Doc: Return s positive integers that sum to N, approximately geometric with ratio r to separate degree levels. */
static inline vector<int> build_block_sizes(int N, int s, double r) {
    vector<double> w(s); double sum = 0.0;
    for (int i = 0; i < s; ++i) { w[i] = pow(r, i); sum += w[i]; }
    vector<double> exact(s);
    vector<int> base(s);
    int total = 0;
    for (int i = 0; i < s; ++i) { exact[i] = (double)N * w[i] / sum; base[i] = max(1, (int)floor(exact[i])); total += base[i]; }
    vector<pair<double,int>> frac;
    for (int i = 0; i < s; ++i) frac.emplace_back(exact[i] - floor(exact[i]), i);
    sort(frac.begin(), frac.end(), greater<pair<double,int>>());
    while (total < N) { for (int t = 0; t < s && total < N; ++t) { base[frac[t].second]++; total++; } }
    while (total > N) { for (int i = s - 1; i >= 0 && total > N; --i) if (base[i] > 1) { base[i]--; total--; } }
    return base;
}

/* Doc: Build sorted vector of expected noisy degrees under BSC noise: mu_i = eps*(N-1) + (1-2eps)*deg(block_i). */
static inline vector<double> build_mu_sorted(const vector<int>& sz, const vector<unsigned int>& mask, const vector<unsigned char>& diag, double eps) {
    int s = (int)sz.size(); int N = 0; for (int v : sz) N += v;
    vector<double> mu; mu.reserve(N);
    const double a = eps * (N - 1), b = 1.0 - 2.0 * eps;
    for (int bidx = 0; bidx < s; ++bidx) {
        int d = diag[bidx] ? (sz[bidx] - 1) : 0;
        unsigned int m = mask[bidx];
        for (int c = 0; c < s; ++c) if (c != bidx && ((m >> c) & 1U)) d += sz[c];
        double mv = a + b * (double)d;
        for (int t = 0; t < sz[bidx]; ++t) mu.push_back(mv);
    }
    sort(mu.begin(), mu.end());
    return mu;
}

/* Doc: Random symmetric block pattern: diag[b] in {0,1}, cross-block bits symmetric for i<j. */
static inline void gen_random_pattern(int s, mt19937& rng, vector<unsigned int>& mask, vector<unsigned char>& diag) {
    mask.assign(s, 0U); diag.assign(s, 0);
    uniform_int_distribution<int> bit01(0, 1);
    for (int i = 0; i < s; ++i) diag[i] = (unsigned char)bit01(rng);
    for (int i = 0; i < s; ++i) for (int j = i + 1; j < s; ++j) if (bit01(rng)) { mask[i] |= (1U << j); mask[j] |= (1U << i); }
}

/* Doc: Add a handful of structured patterns (empty/full/diag/off/banded/threshold) to diversify the pool. */
static inline void gen_structured_patterns(int s, vector<vector<unsigned int>>& masks, vector<vector<unsigned char>>& diags) {
    auto add = [&](const vector<unsigned int>& m, const vector<unsigned char>& d){ masks.push_back(m); diags.push_back(d); };
    vector<unsigned int> m(s,0); vector<unsigned char> d0(s,0), d1(s,1);
    // empty, full
    add(m, d0);
    vector<unsigned int> full(s,0);
    for (int i = 0; i < s; ++i) for (int j = 0; j < s; ++j) if (i!=j) full[i] |= (1U<<j);
    add(full, d1);
    // diag-only, off-only
    add(m, d1);
    add(full, d0);
    // banded |i-j|<=t
    for (int t = 0; t < s; ++t) {
        vector<unsigned int> mm(s,0); vector<unsigned char> dd(s, (unsigned char)(t&1));
        for (int i = 0; i < s; ++i) for (int j = 0; j < s; ++j) if (i!=j && abs(i-j)<=t) mm[i] |= (1U<<j);
        add(mm, dd);
    }
    // threshold i+j <= T
    for (int T = 0; T <= 2*(s-1); T += max(1,s/3)) {
        vector<unsigned int> mm(s,0); vector<unsigned char> dd(s, (unsigned char)((T/2)&1));
        for (int i = 0; i < s; ++i) for (int j = i+1; j < s; ++j) if (i + j <= T) { mm[i] |= (1U<<j); mm[j] |= (1U<<i); }
        add(mm, dd);
    }
}

/* Doc: Pick N as small as possible for score while ensuring enough distinct block patterns and noise robustness.
   Start near a noise-dependent baseline, then increase N until capacity ~ 2^{s(s-1)/2 + b2} >= M (or exponent >=30). */
static inline int chooseN(int M, double eps) {
    auto s_of_N = [](int N){ return min(8, max(4, (int)floor(log2((double)N)) - 1)); };
    // noise-aware baseline with small-N bias at low eps
    int N = (int)llround(10.0 + 100.0 * eps);
    int Nmin = (eps < 0.15 ? 10 : (eps < 0.28 ? 14 : 18));
    N = max(N, Nmin);
    N = min(N, 100);
    double r = max(1.35, min(2.35, 1.5 + 1.5 * eps));
    while (true) {
        int s = s_of_N(N);
        auto sz = build_block_sizes(N, s, r);
        int b2 = 0; for (int v : sz) if (v >= 2) ++b2;
        int expo = s*(s-1)/2 + b2;
        if (expo >= 30 || (1 << expo) >= M) break;
        if (N >= 100) break;
        ++N;
    }
    if (M > 80 && eps > 0.25) N = min(100, N + 2); // stabilize in high-noise/high-M
    return max(8, N);
}

/* Build adjacency bitstring from block pattern and vertex->block mapping */
static inline string build_graph_bits(int N, const vector<pair<unsigned char,unsigned char>>& pairs,
                                      const vector<int>& belong, const vector<unsigned int>& mask, const vector<unsigned char>& diag) {
    int L = N * (N - 1) / 2; string s(L, '0');
    for (int p = 0; p < L; ++p) {
        auto ab = pairs[p];
        int bu = belong[ab.first], bv = belong[ab.second];
        bool conn = (bu == bv) ? (diag[bu] != 0) : (((mask[bu] >> bv) & 1U) != 0);
        if (conn) s[p] = '1';
    }
    return s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int M; double eps;
    if (!(cin >> M >> eps)) return 0;

    // Tune block-size ratio by noise level to widen degree gaps as noise increases
    double BLOCK_R = max(1.35, min(2.35, 1.5 + 1.5 * eps));

    // Choose N and s (number of blocks)
    int N = chooseN(M, eps);
    int s = min(8, max(4, (int)floor(log2((double)N)) - 1 + (eps < 0.08 ? 1 : 0) - (eps > 0.28 ? 1 : 0)));
    vector<int> sz = build_block_sizes(N, s, BLOCK_R);

    // Vertex -> block mapping
    vector<int> belong(N, 0);
    for (int i = 0, cur = 0, b = 0; b < s; ++b) for (int t = 0; t < sz[b]; ++t) belong[cur++] = b;

    // Precompute (i,j) pairs and constants
    int L = N * (N - 1) / 2;
    vector<pair<unsigned char,unsigned char>> pairs;
    pairs.reserve(L);
    for (int i = 0; i < N; ++i) for (int j = i + 1; j < N; ++j)
        pairs.emplace_back((unsigned char)i, (unsigned char)j);

    // Variance-normalized weights for fused scoring (degrees + edge count)
    double invVarDeg = 1.0 / ((double)(N - 1) * eps * (1.0 - eps) + 1e-12);
    double invVarM   = 1.0 / ((double)L * eps * (1.0 - eps) + 1e-12);
    double degW = 1.0 + 0.8 * eps;

    // Build candidate pool of block patterns (structured + random + complements)
    int seed = 146527 + M * 1000 + (int)llround(eps * 100.0) * 7919;
    mt19937 rng(seed);
    vector<vector<unsigned int>> cand_masks;
    vector<vector<unsigned char>> cand_diags;

    gen_structured_patterns(s, cand_masks, cand_diags);

    auto add_comp = [&](const vector<unsigned int>& m, const vector<unsigned char>& d){
        vector<unsigned int> mc(s, 0u); vector<unsigned char> dc(s, 0);
        unsigned int all = (s >= 31 ? 0x7FFFFFFFu : ((1u << s) - 1u));
        for (int i = 0; i < s; ++i) { unsigned int fulli = all & ~(1u << i); mc[i] = fulli ^ m[i]; dc[i] = (unsigned char)(1 - d[i]); }
        cand_masks.push_back(mc); cand_diags.push_back(dc);
    };

    int RAND_CANDS = (eps <= 0.12 ? max(6 * M, 384) : min(3072, max(10 * M, 512)));
    for (int t = 0; t < RAND_CANDS; ++t) {
        vector<unsigned int> m; vector<unsigned char> d;
        gen_random_pattern(s, rng, m, d);
        cand_masks.push_back(m); cand_diags.push_back(d);
        add_comp(m, d);
    }

    struct Cand { vector<unsigned int> mask; vector<unsigned char> diag; vector<double> mu; double mu_m; };
    vector<Cand> pool; pool.reserve(cand_masks.size());
    for (size_t i = 0; i < cand_masks.size(); ++i) {
        vector<double> mu = build_mu_sorted(sz, cand_masks[i], cand_diags[i], eps);
        long long edges = 0;
        for (int b = 0; b < s; ++b) if (cand_diags[i][b]) edges += 1LL * sz[b] * (sz[b] - 1) / 2;
        for (int b = 0; b < s; ++b) for (int c = b + 1; c < s; ++c)
            if ((cand_masks[i][b] >> c) & 1U) edges += 1LL * sz[b] * sz[c];
        double mu_m = eps * (double)L + (1.0 - 2.0 * eps) * (double)edges;
        pool.push_back({cand_masks[i], cand_diags[i], move(mu), mu_m});
    }

    // Greedy farthest-point sampling using fused metric on (mu_sorted, mu_m)
    auto fused_dist = [&](int a, int b)->double{
        double d = sse(pool[a].mu, pool[b].mu) * invVarDeg * degW;
        double dm = pool[a].mu_m - pool[b].mu_m;
        return d + dm * dm * invVarM;
    };
    vector<int> chosen;
    int C = (int)pool.size();
    int seedIdx = 0; double bestVar = -1.0;
    for (int i = 0; i < C; ++i) {
        const auto& mu = pool[i].mu;
        double mean = 0.0; for (double v : mu) mean += v; mean /= mu.size();
        double var = 0.0; for (double v : mu) { double d = v - mean; var += d * d; }
        if (var > bestVar) { bestVar = var; seedIdx = i; }
    }
    chosen.push_back(seedIdx);
    for (int k = 1; k < M; ++k) {
        int nxt = -1; double far = -1.0;
        for (int i = 0; i < C; ++i) {
            bool used = false; for (int id : chosen) if (id == i) { used = true; break; }
            if (used) continue;
            double md = 1e300;
            for (int id : chosen) md = min(md, fused_dist(i, id));
            if (md > far) { far = md; nxt = i; }
        }
        if (nxt < 0) nxt = chosen.back();
        chosen.push_back(nxt);
    }

    // Output graphs
    cout << N << '\n';
    vector<vector<double>> mu_sorted_vec(M, vector<double>(N, 0.0));
    vector<double> mu_m_vec(M, 0.0);
    for (int k = 0; k < M; ++k) {
        const auto& pat = pool[chosen[k]];
        mu_sorted_vec[k] = pat.mu; mu_m_vec[k] = pat.mu_m;
        string sbits = build_graph_bits(N, pairs, belong, pat.mask, pat.diag);
        cout << sbits << '\n';
    }
    cout.flush();

    // Online decoding: fused nearest neighbor using degree multiset + edge count
    for (int q = 0; q < 100; ++q) {
        string H; if (!(cin >> H)) return 0;
        vector<int> deg(N, 0); int m = 0;
        for (int p = 0; p < L; ++p) if (H[p] == '1') { auto ab = pairs[p]; deg[(int)ab.first]++; deg[(int)ab.second]++; ++m; }
        vector<double> dh(N); for (int i = 0; i < N; ++i) dh[i] = (double)deg[i];
        sort(dh.begin(), dh.end());

        int best = 0; double bestS = 1e300;
        for (int k = 0; k < M; ++k) {
            double sdeg = sse(dh, mu_sorted_vec[k]) * invVarDeg * degW;
            double dm = (double)m - mu_m_vec[k];
            double score = sdeg + dm * dm * invVarM;
            if (score < bestS) { bestS = score; best = k; }
        }
        cout << best << '\n';
        cout.flush();
    }
    return 0;
}
# EVOLVE-BLOCK-END