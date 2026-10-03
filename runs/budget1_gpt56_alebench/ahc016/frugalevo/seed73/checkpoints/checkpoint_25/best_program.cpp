# EVOLVE-BLOCK-START
#ifndef ONLINE_JUDGE
// #define DEBUG_OUTPUT // Uncomment for local debug prints
#endif

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <set>
#include <array>
#include <iomanip> 
#include <cmath>   
#include <chrono>  
#include <map>
#include <functional>

// Max N for which we attempt full GED based strategy.
constexpr int N_MAX_GED_CAP = 6; 

// Adjacency matrix for H_k received in query, or for G_i during pairwise GED. Max N=100
bool CURRENT_GRAPH_ADJ_QUERY[100][100]; 

int N_ACTUAL; 
int L_ACTUAL; // N_ACTUAL * (N_ACTUAL - 1) / 2
double NOISE_RATE = 0.0;

// Stores chosen G_j graphs as adjacency matrices (for GED strategy, N <= N_MAX_GED_CAP)
std::vector<std::array<std::array<bool, N_MAX_GED_CAP>, N_MAX_GED_CAP>> G_ADJS_CHOSEN_GED;

// For large N strategy (edge density)
std::vector<std::string> G_STRINGS_CHOSEN_LARGE_N; 
std::vector<int> G_EDGE_COUNTS_LARGE_N;

// Sorted degree multisets of the clique-partition codewords.
std::vector<std::vector<int>> CLIQUE_CODE_DEGREES;

std::vector<int> P_VERTS_PERM_QUERY; // Permutation vector for GED in query
std::mt19937 RND_ENGINE; 

// Temp storage for canonical mask generation (N <= N_MAX_GED_CAP)
bool CANON_TMP_ADJ[N_MAX_GED_CAP][N_MAX_GED_CAP]; 
std::vector<int> CANON_P_PERM; 

enum class Strategy {
    GED,
    EDGE_COUNT
};
Strategy current_strategy;

const std::vector<uint16_t> PRECOMPUTED_CANONICAL_MASKS_N6 = {
    0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 27, 29, 31, 37, 39, 43, 45, 47, 53, 55, 61, 
    63, 73, 75, 77, 79, 91, 93, 95, 111, 117, 119, 125, 127, 141, 143, 157, 159, 173, 175, 
    181, 183, 189, 191, 205, 207, 221, 223, 237, 239, 253, 255, 285, 287, 315, 317, 319, 
    349, 351, 379, 381, 383, 413, 415, 445, 447, 477, 479, 509, 511, 565, 567, 573, 575, 
    589, 591, 605, 607, 637, 639, 701, 703, 717, 719, 733, 735, 749, 751, 765, 767, 797, 
    799, 829, 831, 861, 863, 893, 895, 957, 959, 989, 991, 1021, 1023, 1149, 1151, 1213, 
    1215, 1245, 1247, 1277, 1279, 1533, 1535, 1661, 1663, 1789, 1791, 1917, 1919, 2045, 
    2047, 2109, 2111, 2141, 2143, 2173, 2175, 2205, 2207, 2237, 2239, 2269, 2271, 2301, 
    2303, 2685, 2687, 2813, 2815, 2941, 2943, 3069, 3071, 3277, 3279, 3285, 3287, 3293, 
    3295, 3309, 3311, 3325, 3327, 3357, 3359, 3389, 3391, 3421, 3423, 3453, 3455, 3517, 
    3519, 3549, 3551, 3581, 3583, 3613, 3615, 3645, 3647, 3709, 3711, 3773, 3775, 3837, 
    3839, 4095, 8191, 16383, 32767
}; // Total 156 graphs for N=6.


void mask_to_adj_matrix_small_N(uint16_t mask, int N_nodes, bool adj_matrix[][N_MAX_GED_CAP]) {
    int bit_idx = 0;
    for (int i = 0; i < N_nodes; ++i) {
        adj_matrix[i][i] = false;
        for (int j = i + 1; j < N_nodes; ++j) {
            adj_matrix[i][j] = adj_matrix[j][i] = ((mask >> bit_idx) & 1);
            bit_idx++;
        }
    }
}

uint16_t adj_matrix_to_mask_small_N(int N_nodes, const bool adj_matrix[][N_MAX_GED_CAP], const std::vector<int>& p_perm) {
    uint16_t mask = 0;
    int bit_idx = 0;
    for (int i = 0; i < N_nodes; ++i) {
        for (int j = i + 1; j < N_nodes; ++j) {
            if (adj_matrix[p_perm[i]][p_perm[j]]) { 
                mask |= (1U << bit_idx);
            }
            bit_idx++;
        }
    }
    return mask;
}

uint16_t get_canonical_mask(uint16_t mask_val) { 
    int current_L_for_canon = N_ACTUAL * (N_ACTUAL - 1) / 2;
    if (current_L_for_canon == 0) return 0; 

    mask_to_adj_matrix_small_N(mask_val, N_ACTUAL, CANON_TMP_ADJ);
    
    std::iota(CANON_P_PERM.begin(), CANON_P_PERM.end(), 0); 
    uint16_t min_mask_representation = adj_matrix_to_mask_small_N(N_ACTUAL, CANON_TMP_ADJ, CANON_P_PERM);

    while (std::next_permutation(CANON_P_PERM.begin(), CANON_P_PERM.end())) {
        uint16_t current_perm_mask = adj_matrix_to_mask_small_N(N_ACTUAL, CANON_TMP_ADJ, CANON_P_PERM);
        min_mask_representation = std::min(min_mask_representation, current_perm_mask);
    }
    return min_mask_representation;
}

int calculate_edit_distance_one_perm_small_N(
    const std::array<std::array<bool, N_MAX_GED_CAP>, N_MAX_GED_CAP>& g_j_adj_template 
) {
    int diff_count = 0;
    for (int i = 0; i < N_ACTUAL; ++i) { 
        for (int j = i + 1; j < N_ACTUAL; ++j) { 
            bool template_has_edge = g_j_adj_template[i][j]; 
            bool current_Hk_has_edge = CURRENT_GRAPH_ADJ_QUERY[P_VERTS_PERM_QUERY[i]][P_VERTS_PERM_QUERY[j]];
            if (current_Hk_has_edge != template_has_edge) {
                diff_count++;
            }
        }
    }
    return diff_count;
}

int min_edit_distance_global_perm_small_N(
    const std::array<std::array<bool, N_MAX_GED_CAP>, N_MAX_GED_CAP>& g_j_adj_template
) { 
    if (L_ACTUAL == 0) return 0;

    std::iota(P_VERTS_PERM_QUERY.begin(), P_VERTS_PERM_QUERY.end(), 0); 
    int min_dist = L_ACTUAL + 1; 
    
    long long N_factorial = 1;
    for(int i=1; i<=N_ACTUAL; ++i) N_factorial *= i;

    long long ops_count = 0;
    do {
        int current_dist = calculate_edit_distance_one_perm_small_N(g_j_adj_template);
        min_dist = std::min(min_dist, current_dist);
        if (min_dist == 0) break; 
        
        ops_count++;
        if (ops_count >= N_factorial) break; 
    } while (std::next_permutation(P_VERTS_PERM_QUERY.begin(), P_VERTS_PERM_QUERY.end()));
    
    return min_dist;
}


std::vector<uint16_t> available_canonical_masks;
std::vector<std::vector<int>> all_pairwise_ged_cache; 
std::map<uint16_t, int> mask_to_idx_map; 
std::vector<int> chosen_mask_indices_greedy; 

std::string generate_random_graph_string_large_n(int num_edges, int current_L) { 
    std::string s_out(current_L, '0');
    if (num_edges <= 0 || current_L == 0) return s_out;
    if (num_edges >= current_L) {
        std::fill(s_out.begin(), s_out.end(), '1');
        return s_out;
    }
    std::vector<int> edge_indices(current_L);
    std::iota(edge_indices.begin(), edge_indices.end(), 0);
    std::shuffle(edge_indices.begin(), edge_indices.end(), RND_ENGINE);
    for (int i = 0; i < num_edges; ++i) {
        s_out[edge_indices[i]] = '1';
    }
    return s_out;
}

/**
 * Builds deterministic unions of cliques and selects well-separated
 * integer partitions using a bounded farthest-first procedure.
 */
void build_clique_partition_codewords(int M, int N, int L) {
    std::vector<std::vector<int>> partitions;
    std::vector<int> current;

    std::function<void(int, int)> enumerate =
        [&](int remaining, int largest) {
            if ((int)partitions.size() >= 4096) return;
            if (remaining == 0) {
                partitions.push_back(current);
                return;
            }

            for (int x = std::min(remaining, largest); x >= 1; --x) {
                current.push_back(x);
                enumerate(remaining - x, x);
                current.pop_back();
                if ((int)partitions.size() >= 4096) return;
            }
        };

    enumerate(N, N);
    if (partitions.empty()) partitions.push_back({N});

    // Convert each partition to its sorted degree multiset.
    std::vector<std::vector<int>> partition_degrees(partitions.size());
    std::vector<int> partition_edges(partitions.size(), 0);

    for (int p = 0; p < (int)partitions.size(); ++p) {
        for (int size : partitions[p]) {
            for (int i = 0; i < size; ++i)
                partition_degrees[p].push_back(size - 1);
            partition_edges[p] += size * (size - 1) / 2;
        }
        std::sort(partition_degrees[p].begin(), partition_degrees[p].end());
    }

    // Select structurally distant partitions rather than adjacent
    // lexicographic partitions, which can have nearly identical degrees.
    std::vector<int> selected;
    std::vector<char> used(partitions.size(), false);

    selected.push_back(0);
    used[0] = true;

    if (partitions.size() > 1) {
        int last = (int)partitions.size() - 1;
        selected.push_back(last);
        used[last] = true;
    }

    while ((int)selected.size() < M &&
           (int)selected.size() < (int)partitions.size()) {
        int best = -1;
        long long best_score = -1;

        for (int p = 0; p < (int)partitions.size(); ++p) {
            if (used[p]) continue;

            long long min_distance = (1LL << 60);
            for (int q : selected) {
                long long d = 0;
                for (int v = 0; v < N; ++v) {
                    long long delta =
                        partition_degrees[p][v] - partition_degrees[q][v];
                    d += delta * delta;
                }

                // Edge count is a secondary structural invariant.
                d += 2LL * std::abs(
                    partition_edges[p] - partition_edges[q]);
                min_distance = std::min(min_distance, d);
            }

            if (min_distance > best_score) {
                best_score = min_distance;
                best = p;
            }
        }

        if (best < 0) break;
        used[best] = true;
        selected.push_back(best);
    }

    while ((int)selected.size() < M)
        selected.push_back(selected.size() % partitions.size());

    G_STRINGS_CHOSEN_LARGE_N.assign(M, std::string(L, '0'));
    G_EDGE_COUNTS_LARGE_N.assign(M, 0);
    CLIQUE_CODE_DEGREES.assign(M, std::vector<int>(N, 0));

    for (int code = 0; code < M; ++code) {
        const std::vector<int>& sizes = partitions[selected[code]];
        std::vector<int> clique_of_vertex(N, -1);

        int vertex = 0;
        for (int c = 0; c < (int)sizes.size(); ++c) {
            for (int t = 0; t < sizes[c]; ++t)
                clique_of_vertex[vertex++] = c;
        }

        std::string graph(L, '0');
        int position = 0;

        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                if (clique_of_vertex[i] == clique_of_vertex[j])
                    graph[position] = '1';
                ++position;
            }
        }

        G_STRINGS_CHOSEN_LARGE_N[code] = std::move(graph);
        G_EDGE_COUNTS_LARGE_N[code] =
            partition_edges[selected[code]];
        CLIQUE_CODE_DEGREES[code] =
            partition_degrees[selected[code]];
    }
}

/**
 * Counts set bits in the upper-triangular graph representation.
 */
int count_set_bits_in_string(const std::string& s) {
    return std::count(s.begin(), s.end(), '1');
}

/**
 * Computes the sorted degree multiset of a packed graph string.
 */
std::vector<int> sorted_degrees_from_string(
    const std::string& s, int N) {
    std::vector<int> degrees(N, 0);
    int position = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j, ++position) {
            if (position < (int)s.size() && s[position] == '1') {
                ++degrees[i];
                ++degrees[j];
            }
        }
    }

    std::sort(degrees.begin(), degrees.end());
    return degrees;
}

void string_to_adj_matrix_query(const std::string& s, int N_nodes) {
    int char_idx = 0;
    for(int i=0; i<N_nodes; ++i) { 
        CURRENT_GRAPH_ADJ_QUERY[i][i] = false;
        for(int j=i+1; j<N_nodes; ++j) {
            if (char_idx < (int)s.length()) {
                CURRENT_GRAPH_ADJ_QUERY[i][j] = CURRENT_GRAPH_ADJ_QUERY[j][i] = (s[char_idx++] == '1');
            } else { 
                CURRENT_GRAPH_ADJ_QUERY[i][j] = CURRENT_GRAPH_ADJ_QUERY[j][i] = false;
            }
        }
    }
}


/**
 * Builds a joint clique/complement codebook and selects codewords by
 * incremental farthest-first distance over sorted degree profiles.
 */
void build_complement_partition_codewords(int M, int N, int L) {
    struct Candidate {
        std::vector<int> degrees;
        int edges;
        bool complement;
        std::vector<int> blocks;
    };

    std::vector<std::vector<int>> partitions;
    std::vector<int> current;

    std::function<void(int, int)> enumerate =
        [&](int remaining, int largest) {
            if ((int)partitions.size() >= 4096) return;
            if (remaining == 0) {
                partitions.push_back(current);
                return;
            }
            for (int x = std::min(remaining, largest); x >= 1; --x) {
                current.push_back(x);
                enumerate(remaining - x, x);
                current.pop_back();
                if ((int)partitions.size() >= 4096) return;
            }
        };

    enumerate(N, N);
    if (partitions.empty()) partitions.push_back({N});

    std::vector<Candidate> candidates;
    candidates.reserve(std::min(8192, 2 * (int)partitions.size()));

    for (const auto& blocks : partitions) {
        int clique_edges = 0;
        std::vector<int> clique_degrees;
        clique_degrees.reserve(N);

        for (int s : blocks) {
            clique_edges += s * (s - 1) / 2;
            for (int i = 0; i < s; ++i)
                clique_degrees.push_back(s - 1);
        }
        std::sort(clique_degrees.begin(), clique_degrees.end());

        Candidate clique;
        clique.degrees = clique_degrees;
        clique.edges = clique_edges;
        clique.complement = false;
        clique.blocks = blocks;
        candidates.push_back(std::move(clique));

        Candidate inverse;
        inverse.degrees.resize(N);
        for (int i = 0; i < N; ++i)
            inverse.degrees[i] = N - 1 - clique_degrees[i];
        std::sort(inverse.degrees.begin(), inverse.degrees.end());
        inverse.edges = L - clique_edges;
        inverse.complement = true;
        inverse.blocks = blocks;
        candidates.push_back(std::move(inverse));
    }

    /**
     * Selects codewords by farthest-first Mahalanobis separation.
     *
     * Degree vectors are centered once because the inverse covariance removes
     * the all-ones component. Distances are then evaluated from cached norms
     * and dot products, preserving the exact noise-whitened metric while
     * avoiding repeated construction of degree differences.
     */
    const int candidate_count = static_cast<int>(candidates.size());
    std::vector<std::vector<long double>> centered_degrees(
        candidate_count, std::vector<long double>(N, 0.0L));
    std::vector<long double> centered_norm(candidate_count, 0.0L);

    for (int i = 0; i < candidate_count; ++i) {
        long double mean = 0.0L;
        for (int v = 0; v < N; ++v)
            mean += static_cast<long double>(candidates[i].degrees[v]);
        mean /= static_cast<long double>(N);

        for (int v = 0; v < N; ++v) {
            centered_degrees[i][v] =
                static_cast<long double>(candidates[i].degrees[v]) - mean;
            centered_norm[i] +=
                centered_degrees[i][v] * centered_degrees[i][v];
        }
    }

    auto distance = [&](int a, int b) {
        long double dot = 0.0L;
        for (int v = 0; v < N; ++v)
            dot += centered_degrees[a][v] * centered_degrees[b][v];

        const long double squared_distance =
            std::max(0.0L,
                centered_norm[a] + centered_norm[b] - 2.0L * dot);

        if (NOISE_RATE <= 1e-12) {
            const long double tie_break =
                static_cast<long double>(
                    std::abs(candidates[a].edges - candidates[b].edges));
            return squared_distance + tie_break * 1e-9L;
        }

        const long double scale =
            1.0L - 2.0L * static_cast<long double>(NOISE_RATE);
        const long double variance =
            static_cast<long double>(NOISE_RATE) *
            (1.0L - static_cast<long double>(NOISE_RATE));

        return scale * scale * squared_distance /
            (variance * static_cast<long double>(N - 2));
    };

    std::vector<int> selected;
    std::vector<char> used(candidate_count, false);

    // Keep the extreme-density codewords as deterministic anchors.
    auto add_by_shape = [&](int wanted_edges) {
        int best = -1;
        for (int i = 0; i < candidate_count; ++i) {
            if (used[i] || candidates[i].edges != wanted_edges) continue;
            if (best < 0 || candidates[i].degrees < candidates[best].degrees)
                best = i;
        }
        if (best >= 0) {
            used[best] = true;
            selected.push_back(best);
        }
    };

    add_by_shape(0);
    add_by_shape(L);

    // Incrementally maintain each candidate's distance to its nearest
    // selected codeword.
    std::vector<long double> nearest(
        candidate_count, std::numeric_limits<long double>::infinity());

    for (int i = 0; i < candidate_count; ++i) {
        if (!used[i]) {
            for (int s : selected)
                nearest[i] = std::min(nearest[i], distance(i, s));
        }
    }

    while ((int)selected.size() < M) {
        int best = -1;
        long double best_distance = -1.0L;

        for (int i = 0; i < candidate_count; ++i) {
            if (used[i]) continue;

            if (nearest[i] > best_distance) {
                best_distance = nearest[i];
                best = i;
            } else if (nearest[i] == best_distance && best >= 0 &&
                       candidates[i].edges > candidates[best].edges) {
                best = i;
            }
        }

        if (best < 0) break;

        used[best] = true;
        selected.push_back(best);

        for (int i = 0; i < candidate_count; ++i) {
            if (!used[i])
                nearest[i] = std::min(nearest[i], distance(i, best));
        }
    }

    while ((int)selected.size() < M)
        selected.push_back(selected.size() % candidates.size());

    G_STRINGS_CHOSEN_LARGE_N.assign(M, std::string(L, '0'));
    G_EDGE_COUNTS_LARGE_N.assign(M, 0);
    CLIQUE_CODE_DEGREES.assign(M, std::vector<int>(N, 0));

    for (int code = 0; code < M; ++code) {
        const Candidate& candidate = candidates[selected[code]];
        G_EDGE_COUNTS_LARGE_N[code] = candidate.edges;
        CLIQUE_CODE_DEGREES[code] = candidate.degrees;

        std::vector<int> block_of_vertex(N, -1);
        int vertex = 0;
        for (int b = 0; b < (int)candidate.blocks.size(); ++b) {
            for (int t = 0; t < candidate.blocks[b]; ++t)
                block_of_vertex[vertex++] = b;
        }

        int position = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                const bool same_block =
                    block_of_vertex[i] == block_of_vertex[j];
                const bool edge = same_block ^ candidate.complement;
                G_STRINGS_CHOSEN_LARGE_N[code][position++] =
                    edge ? '1' : '0';
            }
        }
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    
    unsigned int seed_val = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    RND_ENGINE.seed(seed_val);

    int M_graphs;
    double epsilon_noise_rate;
    std::cin >> M_graphs >> epsilon_noise_rate;
    NOISE_RATE = epsilon_noise_rate;
    
    int N_for_GED_strat;
    if (M_graphs <= 11) N_for_GED_strat = 4; 
    else if (M_graphs <= 34) N_for_GED_strat = 5; 
    else N_for_GED_strat = N_MAX_GED_CAP; 

    // Select the legal N maximizing the predicted 100-query survival score.
    // The likelihood model exactly mirrors the edge-count construction below.
    int N_candidate_EC = 4;
    double best_predicted_score = -1.0;

    for (int candidate_N = 4; candidate_N <= 100; ++candidate_N) {
        const int candidate_L = candidate_N * (candidate_N - 1) / 2;
        std::vector<int> candidate_counts(M_graphs, 0);

        if (M_graphs == 1) {
            candidate_counts[0] = candidate_L / 2;
        } else {
            for (int k = 0; k < M_graphs; ++k) {
                candidate_counts[k] = static_cast<int>(
                    std::round(k * candidate_L / (M_graphs - 1.0)));
            }

            for (int k = 0; k + 1 < M_graphs; ++k) {
                if (candidate_counts[k + 1] <= candidate_counts[k])
                    candidate_counts[k + 1] = candidate_counts[k] + 1;
            }

            if (candidate_counts.back() > candidate_L) {
                const int excess = candidate_counts.back() - candidate_L;
                for (int k = 0; k < M_graphs; ++k)
                    candidate_counts[k] -= excess;
            }

            for (int k = 0; k < M_graphs; ++k)
                candidate_counts[k] =
                    std::min(candidate_L, std::max(0, candidate_counts[k]));

            for (int k = 0; k + 1 < M_graphs; ++k)
                candidate_counts[k + 1] =
                    std::max(candidate_counts[k + 1], candidate_counts[k] + 1);

            for (int k = 0; k < M_graphs; ++k)
                candidate_counts[k] =
                    std::min(candidate_L, std::max(0, candidate_counts[k]));
        }

        double average_success = 1.0;
        if (epsilon_noise_rate > 1e-12) {
            const double variance =
                candidate_L * epsilon_noise_rate * (1.0 - epsilon_noise_rate);
            const double sigma = std::sqrt(variance);
            std::vector<double> means(M_graphs);

            for (int k = 0; k < M_graphs; ++k) {
                means[k] = epsilon_noise_rate * candidate_L +
                    (1.0 - 2.0 * epsilon_noise_rate) * candidate_counts[k];
            }

            average_success = 0.0;
            for (int k = 0; k < M_graphs; ++k) {
                const double lower = (k == 0)
                    ? -1e100 : 0.5 * (means[k - 1] + means[k]);
                const double upper = (k + 1 == M_graphs)
                    ? 1e100 : 0.5 * (means[k] + means[k + 1]);

                const double zl = (lower < -1e50)
                    ? -1e100 : (lower - means[k]) / sigma;
                const double zu = (upper > 1e50)
                    ? 1e100 : (upper - means[k]) / sigma;

                const double pl = (zl < -1e50)
                    ? 0.0 : 0.5 * (1.0 + std::erf(zl / std::sqrt(2.0)));
                const double pu = (zu > 1e50)
                    ? 1.0 : 0.5 * (1.0 + std::erf(zu / std::sqrt(2.0)));
                average_success += std::max(0.0, pu - pl);
            }
            average_success /= M_graphs;
        }

        const double predicted_score =
            std::pow(0.9 + 0.1 * average_success, 100.0) / candidate_N;
        if (predicted_score > best_predicted_score) {
            best_predicted_score = predicted_score;
            N_candidate_EC = candidate_N;
        }
    }

    if (N_candidate_EC > N_for_GED_strat) {
        current_strategy = Strategy::EDGE_COUNT;
        N_ACTUAL = N_candidate_EC;
    } else {
        current_strategy = Strategy::GED;
        N_ACTUAL = N_for_GED_strat;
    }
    N_ACTUAL = std::min(100, std::max(4, N_ACTUAL)); // Final check on N_ACTUAL bounds
            
    L_ACTUAL = N_ACTUAL * (N_ACTUAL - 1) / 2;
    std::cout << N_ACTUAL << std::endl;

#ifdef DEBUG_OUTPUT
    std::cerr << "# M=" << M_graphs << ", eps=" << epsilon_noise_rate << std::endl;
    std::cerr << "# Chosen N=" << N_ACTUAL << ", Strategy=" << (current_strategy == Strategy::GED ? "GED" : "EDGE_COUNT") << std::endl;
    std::cerr << "# L_ideal=" << L_ideal << ", N_candidate_EC=" << N_candidate_EC << ", N_for_GED_strat=" << N_for_GED_strat << std::endl;
#endif

    if (current_strategy == Strategy::GED) {
        P_VERTS_PERM_QUERY.resize(N_ACTUAL); CANON_P_PERM.resize(N_ACTUAL);
        
        if (N_ACTUAL == 6) {
            available_canonical_masks = PRECOMPUTED_CANONICAL_MASKS_N6;
        } else { 
            std::set<uint16_t> unique_masks_set;
            if (L_ACTUAL > 0) { 
                for (unsigned int i = 0; i < (1U << L_ACTUAL); ++i) {
                    unique_masks_set.insert(get_canonical_mask(static_cast<uint16_t>(i)));
                }
            } else { 
                unique_masks_set.insert(0); 
            }
            available_canonical_masks.assign(unique_masks_set.begin(), unique_masks_set.end());
        }
        
        int num_total_isos = available_canonical_masks.size();
#ifdef DEBUG_OUTPUT
    std::cerr << "# Num non-isomorphic graphs for N=" << N_ACTUAL << " is " << num_total_isos << std::endl;
#endif
        mask_to_idx_map.clear();
        for(int i=0; i<num_total_isos; ++i) mask_to_idx_map[available_canonical_masks[i]] = i;

        if (num_total_isos > 0) {
            all_pairwise_ged_cache.assign(num_total_isos, std::vector<int>(num_total_isos, 0));
            bool graph_i_adj_cstyle[N_MAX_GED_CAP][N_MAX_GED_CAP]; 
            std::array<std::array<bool, N_MAX_GED_CAP>, N_MAX_GED_CAP> graph_j_adj_stdarray;

            for (int i = 0; i < num_total_isos; ++i) {
                mask_to_adj_matrix_small_N(available_canonical_masks[i], N_ACTUAL, graph_i_adj_cstyle);
                for(int r=0; r<N_ACTUAL; ++r) for(int c=0; c<N_ACTUAL; ++c) CURRENT_GRAPH_ADJ_QUERY[r][c] = graph_i_adj_cstyle[r][c];

                for (int j = i + 1; j < num_total_isos; ++j) {
                    bool temp_adj_for_gj[N_MAX_GED_CAP][N_MAX_GED_CAP];
                    mask_to_adj_matrix_small_N(available_canonical_masks[j], N_ACTUAL, temp_adj_for_gj);
                    for(int r=0; r<N_ACTUAL; ++r) for(int c=0; c<N_ACTUAL; ++c) graph_j_adj_stdarray[r][c] = temp_adj_for_gj[r][c];
                    
                    all_pairwise_ged_cache[i][j] = all_pairwise_ged_cache[j][i] = min_edit_distance_global_perm_small_N(graph_j_adj_stdarray);
                }
            }
        }
        
        chosen_mask_indices_greedy.clear();
        std::vector<bool> is_chosen_idx(num_total_isos, false);

        if (num_total_isos > 0) { 
            if (mask_to_idx_map.count(0)) { 
                int zero_idx = mask_to_idx_map.at(0);
                if (chosen_mask_indices_greedy.size() < (size_t)M_graphs) {
                    chosen_mask_indices_greedy.push_back(zero_idx); 
                    is_chosen_idx[zero_idx] = true;
                }
            }
            if (L_ACTUAL > 0 && chosen_mask_indices_greedy.size() < (size_t)M_graphs) {
                uint16_t complete_mask_val = (1U << L_ACTUAL) - 1; 
                uint16_t canonical_complete_mask = get_canonical_mask(complete_mask_val); 
                if (mask_to_idx_map.count(canonical_complete_mask)) {
                    int complete_idx = mask_to_idx_map.at(canonical_complete_mask);
                    if (!is_chosen_idx[complete_idx]) { 
                         chosen_mask_indices_greedy.push_back(complete_idx);
                         is_chosen_idx[complete_idx] = true;
                    }
                }
            }
        }
        
        for (int k_count = chosen_mask_indices_greedy.size(); k_count < M_graphs; ++k_count) {
            if (chosen_mask_indices_greedy.size() >= (size_t)num_total_isos) { 
                break;
            }

            int best_new_idx_to_add = -1;
            int max_of_min_distances_found = -1;
            long long best_sum_distances_found = -1;

            // Select a maximin candidate, breaking ties by total distance.
            // This preserves the minimum-distance objective while avoiding
            // arbitrary dependence on canonical-mask enumeration order.
            for (int cand_idx = 0; cand_idx < num_total_isos; ++cand_idx) {
                if (is_chosen_idx[cand_idx]) continue;

                int current_cand_min_dist_to_existing_G = L_ACTUAL + 1;
                long long current_cand_sum_distances = 0;

                for (int chosen_idx : chosen_mask_indices_greedy) {
                    const int d = all_pairwise_ged_cache[cand_idx][chosen_idx];
                    current_cand_min_dist_to_existing_G =
                        std::min(current_cand_min_dist_to_existing_G, d);
                    current_cand_sum_distances += d;
                }

                if (current_cand_min_dist_to_existing_G > max_of_min_distances_found ||
                    (current_cand_min_dist_to_existing_G == max_of_min_distances_found &&
                     current_cand_sum_distances > best_sum_distances_found)) {
                    max_of_min_distances_found = current_cand_min_dist_to_existing_G;
                    best_sum_distances_found = current_cand_sum_distances;
                    best_new_idx_to_add = cand_idx;
                }
            }
            
            if (best_new_idx_to_add != -1) { 
                chosen_mask_indices_greedy.push_back(best_new_idx_to_add); 
                is_chosen_idx[best_new_idx_to_add] = true; 
            } else {
                break; 
            }
        }
        
        int num_distinct_chosen_graphs = chosen_mask_indices_greedy.size();
        if (num_distinct_chosen_graphs < M_graphs) {
            int fallback_idx = 0; 
            if (num_total_isos > 0) { 
                if (mask_to_idx_map.count(0)) { 
                    fallback_idx = mask_to_idx_map.at(0); 
                } 
            }
            
            for (int k_idx = num_distinct_chosen_graphs; k_idx < M_graphs; ++k_idx) {
                 if (num_total_isos > 0) {
                    chosen_mask_indices_greedy.push_back(fallback_idx);
                 } else { 
                    chosen_mask_indices_greedy.push_back(0); 
                 }
            }
        }
#ifdef DEBUG_OUTPUT
    std::cerr << "# Chosen mask indices (size " << chosen_mask_indices_greedy.size() << "): ";
    if (!available_canonical_masks.empty()){ // Check before accessing
        for(int idx : chosen_mask_indices_greedy) {
            if (idx < available_canonical_masks.size()) std::cerr << idx << " (" << available_canonical_masks[idx] << ") ";
            else std::cerr << idx << " (OOB) ";
        }
    }
    std::cerr << std::endl;
#endif
        
        G_ADJS_CHOSEN_GED.resize(M_graphs);
        for (int k_idx = 0; k_idx < M_graphs; ++k_idx) {
            uint16_t mask_to_print = 0; 
            if (k_idx < chosen_mask_indices_greedy.size() && 
                !available_canonical_masks.empty() && 
                chosen_mask_indices_greedy[k_idx] < available_canonical_masks.size()) {
                 mask_to_print = available_canonical_masks[chosen_mask_indices_greedy[k_idx]];
            } else if (L_ACTUAL == 0 && k_idx < chosen_mask_indices_greedy.size()) { 
                 mask_to_print = 0;
            }
            
            bool temp_adj_cstyle[N_MAX_GED_CAP][N_MAX_GED_CAP];
            mask_to_adj_matrix_small_N(mask_to_print, N_ACTUAL, temp_adj_cstyle);
            for(int r=0; r<N_ACTUAL; ++r) for(int c=0; c<N_ACTUAL; ++c) G_ADJS_CHOSEN_GED[k_idx][r][c] = temp_adj_cstyle[r][c];
            
            std::string s_out = "";
            if (L_ACTUAL > 0) {
                for (int bit_idx = 0; bit_idx < L_ACTUAL; ++bit_idx) {
                    s_out += ((mask_to_print >> bit_idx) & 1) ? '1' : '0';
                }
            }
            std::cout << s_out << std::endl;
        }

    } else {
        build_complement_partition_codewords(
            M_graphs, N_ACTUAL, L_ACTUAL);

        for (int k = 0; k < M_graphs; ++k)
            std::cout << G_STRINGS_CHOSEN_LARGE_N[k] << std::endl;
    }
    std::cout.flush(); // Explicit flush after all G_k are printed

    for (int q_idx = 0; q_idx < 100; ++q_idx) {
        std::string h_str; std::cin >> h_str;
        if (current_strategy == Strategy::GED) {
            if (M_graphs == 0) { std::cout << 0 << std::endl; std::cout.flush(); continue; }
            if (G_ADJS_CHOSEN_GED.empty()){ 
#ifdef DEBUG_OUTPUT
                std::cerr << "# Query " << q_idx << ": G_ADJS_CHOSEN_GED is empty but M_graphs=" << M_graphs << ". Outputting 0." << std::endl;
#endif
                std::cout << 0 << std::endl; std::cout.flush(); continue; 
            }
            
            string_to_adj_matrix_query(h_str, N_ACTUAL);

            int best_g_idx = 0; int min_dist_found = L_ACTUAL + 2; 
            for (int j=0; j < M_graphs; ++j) { 
                if (j >= G_ADJS_CHOSEN_GED.size()) { 
#ifdef DEBUG_OUTPUT
                    std::cerr << "# Query " << q_idx << ": Index j=" << j << " out of bounds for G_ADJS_CHOSEN_GED (size " << G_ADJS_CHOSEN_GED.size() << ")" << std::endl;
#endif
                    continue; 
                }
                int dist = min_edit_distance_global_perm_small_N(G_ADJS_CHOSEN_GED[j]);
                if (dist < min_dist_found) { 
                    min_dist_found = dist; 
                    best_g_idx = j; 
                }
            }
            std::cout << best_g_idx << std::endl;

        } else {
            if (M_graphs == 0 || CLIQUE_CODE_DEGREES.empty()) {
                std::cout << 0 << std::endl;
                std::cout.flush();
                continue;
            }

            const std::vector<int> observed =
                sorted_degrees_from_string(h_str, N_ACTUAL);
            const int observed_edges =
                count_set_bits_in_string(h_str);

            const double scale =
                1.0 - 2.0 * epsilon_noise_rate;
            const double degree_offset =
                epsilon_noise_rate * (N_ACTUAL - 1);

            int best_g_idx = 0;
            double best_score = 1e100;

            for (int j = 0; j < M_graphs; ++j) {
                double score = 0.0;

                /*
                 * Degree-likelihood decoder:
                 * compare each sorted degree using the binomial variance of
                 * the noisy incident edges, while retaining a small
                 * unnormalised term to prevent very small variances from
                 * dominating the score near epsilon=0.
                 */
                for (int v = 0; v < N_ACTUAL; ++v) {
                    const double expected =
                        degree_offset +
                        scale * CLIQUE_CODE_DEGREES[j][v];
                    const double delta =
                        observed[v] - expected;

                    // Every incident original edge is flipped independently.
                    // Therefore the degree-noise variance is determined by
                    // epsilon, not by the observed noisy edge probability.
                    const double degree_variance =
                        std::max(1e-9,
                            (N_ACTUAL - 1) *
                            epsilon_noise_rate *
                            (1.0 - epsilon_noise_rate));

                    score += delta * delta / degree_variance;
                    score += 0.05 * delta * delta;
                }

                const double expected_edges =
                    epsilon_noise_rate * L_ACTUAL +
                    scale * G_EDGE_COUNTS_LARGE_N[j];
                const double edge_delta =
                    observed_edges - expected_edges;

                /*
                 * The edge count is correlated with the degree multiset, so
                 * use it as a downweighted global consistency check.  Its
                 * variance is computed from the predicted noisy edge
                 * probability, making the weight stable across N and density.
                 */
                // The total edge count is the sum of independent flip errors
                // over all original edges, so its variance is L*eps*(1-eps).
                const double edge_variance =
                    std::max(1e-9,
                        L_ACTUAL *
                        epsilon_noise_rate *
                        (1.0 - epsilon_noise_rate));

                score += 0.25 * edge_delta * edge_delta / edge_variance;

                if (score < best_score) {
                    best_score = score;
                    best_g_idx = j;
                }
            }

            std::cout << best_g_idx << std::endl;
        }
        std::cout.flush(); // Explicit flush after each query prediction
    }
    return 0;
}
# EVOLVE-BLOCK-END