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

// Max N for which we attempt full GED based strategy.
constexpr int N_MAX_GED_CAP = 6; 

// Adjacency matrix for H_k received in query, or for G_i during pairwise GED. Max N=100
bool CURRENT_GRAPH_ADJ_QUERY[100][100]; 

int N_ACTUAL; 
int L_ACTUAL; // N_ACTUAL * (N_ACTUAL - 1) / 2

// Stores chosen G_j graphs as adjacency matrices (for GED strategy, N <= N_MAX_GED_CAP)
std::vector<std::array<std::array<bool, N_MAX_GED_CAP>, N_MAX_GED_CAP>> G_ADJS_CHOSEN_GED;

// For large N strategy (edge density)
std::vector<std::string> G_STRINGS_CHOSEN_LARGE_N; 
std::vector<int> G_EDGE_COUNTS_LARGE_N; 

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

int count_set_bits_in_string(const std::string& s) {
    return std::count(s.begin(), s.end(), '1');
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


int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    
    unsigned int seed_val = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    RND_ENGINE.seed(seed_val);

    int M_graphs;
    double epsilon_noise_rate;
    std::cin >> M_graphs >> epsilon_noise_rate;
    
    int N_for_GED_strat;
    if (M_graphs <= 11) N_for_GED_strat = 4; 
    else if (M_graphs <= 34) N_for_GED_strat = 5; 
    else N_for_GED_strat = N_MAX_GED_CAP; 

    const double K_SEP = 2.5; 
    double L_ideal;
    double L_ideal_numerator = K_SEP * K_SEP * (M_graphs > 1 ? (M_graphs - 1.0) * (M_graphs - 1.0) : 1.0) * 
                               epsilon_noise_rate * (1.0 - epsilon_noise_rate);
    double L_ideal_denominator_factor = (0.5 - epsilon_noise_rate);
    double L_ideal_denominator = L_ideal_denominator_factor * L_ideal_denominator_factor;
    
    if (std::abs(0.5 - epsilon_noise_rate) < 1e-9) { 
        L_ideal = (100.0 * 99.0) / 2.0; 
    } else {
        L_ideal = L_ideal_numerator / L_ideal_denominator;
    }
    if (L_ideal < 0) L_ideal = 0; 
    
    int N_candidate_EC = 4; 
    if (L_ideal > 1e-9) { 
         double discriminant = 1.0 + 8.0 * L_ideal; 
         if (discriminant >=0) { 
            N_candidate_EC = static_cast<int>(std::ceil((1.0 + std::sqrt(discriminant)) / 2.0));
         } else { 
            N_candidate_EC = 100; 
         }
    }
    N_candidate_EC = std::max(4, N_candidate_EC); 
    N_candidate_EC = std::min(100, N_candidate_EC); 

    if (epsilon_noise_rate < 0.01) {
        current_strategy = Strategy::GED; N_ACTUAL = N_for_GED_strat;
    } else { 
        if (N_candidate_EC > N_for_GED_strat) { 
             current_strategy = Strategy::EDGE_COUNT; N_ACTUAL = N_candidate_EC;
        } else { 
            current_strategy = Strategy::GED; N_ACTUAL = N_for_GED_strat;
        }
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

            for (int cand_idx = 0; cand_idx < num_total_isos; ++cand_idx) {
                if (is_chosen_idx[cand_idx]) continue;

                int current_cand_min_dist_to_existing_G;
                if (chosen_mask_indices_greedy.empty()) { 
                     current_cand_min_dist_to_existing_G = L_ACTUAL + 1; 
                } else {
                    current_cand_min_dist_to_existing_G = L_ACTUAL + 1;
                    for (int chosen_idx : chosen_mask_indices_greedy) {
                        current_cand_min_dist_to_existing_G = std::min(current_cand_min_dist_to_existing_G, all_pairwise_ged_cache[cand_idx][chosen_idx]);
                    }
                }
                
                if (current_cand_min_dist_to_existing_G > max_of_min_distances_found) {
                    max_of_min_distances_found = current_cand_min_dist_to_existing_G;
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
        G_EDGE_COUNTS_LARGE_N.resize(M_graphs); G_STRINGS_CHOSEN_LARGE_N.resize(M_graphs);
        if (M_graphs == 1) { 
             G_EDGE_COUNTS_LARGE_N[0] = (L_ACTUAL > 0) ? L_ACTUAL / 2 : 0; 
        } else { 
            for (int k=0; k<M_graphs; ++k) G_EDGE_COUNTS_LARGE_N[k] = static_cast<int>(std::round((double)k * L_ACTUAL / (M_graphs - 1.0)));
            
            for (int k=0; k<M_graphs-1; ++k) {
                if (G_EDGE_COUNTS_LARGE_N[k+1] <= G_EDGE_COUNTS_LARGE_N[k]) {
                    G_EDGE_COUNTS_LARGE_N[k+1] = G_EDGE_COUNTS_LARGE_N[k] + 1;
                }
            }
            if (M_graphs > 0 && G_EDGE_COUNTS_LARGE_N[M_graphs-1] > L_ACTUAL) { // M_graphs > 0 check
                int exceso = G_EDGE_COUNTS_LARGE_N[M_graphs-1] - L_ACTUAL;
                for (int k=0; k<M_graphs; ++k) {
                    G_EDGE_COUNTS_LARGE_N[k] -= exceso;
                }
            }
            for (int k=0; k<M_graphs; ++k) G_EDGE_COUNTS_LARGE_N[k] = std::min(L_ACTUAL, std::max(0, G_EDGE_COUNTS_LARGE_N[k]));
            
            for (int k=0; k<M_graphs-1; ++k) {
                 G_EDGE_COUNTS_LARGE_N[k+1] = std::max(G_EDGE_COUNTS_LARGE_N[k+1], G_EDGE_COUNTS_LARGE_N[k] + 1);
            }
            for (int k=0; k<M_graphs; ++k) G_EDGE_COUNTS_LARGE_N[k] = std::min(L_ACTUAL, std::max(0, G_EDGE_COUNTS_LARGE_N[k]));
        }

        for (int k=0; k<M_graphs; ++k) {
            G_STRINGS_CHOSEN_LARGE_N[k] = generate_random_graph_string_large_n(G_EDGE_COUNTS_LARGE_N[k], L_ACTUAL);
            std::cout << G_STRINGS_CHOSEN_LARGE_N[k] << std::endl;
        }
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
            if (M_graphs == 0) { std::cout << 0 << std::endl; std::cout.flush(); continue; }
            if (G_EDGE_COUNTS_LARGE_N.empty()){ 
#ifdef DEBUG_OUTPUT
                std::cerr << "# Query " << q_idx << ": G_EDGE_COUNTS_LARGE_N is empty but M_graphs=" << M_graphs << ". Outputting 0." << std::endl;
#endif
                std::cout << 0 << std::endl; std::cout.flush(); continue; 
            }

            int edges_Hk = count_set_bits_in_string(h_str);
            int best_g_idx = 0; double min_abs_diff_expected_edges = -1.0; 
            for (int j=0; j<M_graphs; ++j) { 
                if (j >= G_EDGE_COUNTS_LARGE_N.size()) {
#ifdef DEBUG_OUTPUT
                     std::cerr << "# Query " << q_idx << ": Index j=" << j << " out of bounds for G_EDGE_COUNTS_LARGE_N (size " << G_EDGE_COUNTS_LARGE_N.size() << ")" << std::endl;
#endif
                    continue;
                }
                double expected_edges_Hk_from_Gj = (double)G_EDGE_COUNTS_LARGE_N[j] * (1.0 - 2.0*epsilon_noise_rate) + (double)L_ACTUAL * epsilon_noise_rate;
                double diff = std::abs((double)edges_Hk - expected_edges_Hk_from_Gj);
                if (min_abs_diff_expected_edges < -0.5 || diff < min_abs_diff_expected_edges) { 
                    min_abs_diff_expected_edges = diff; 
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