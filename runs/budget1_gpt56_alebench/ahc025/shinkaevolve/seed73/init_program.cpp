# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <set>
#include <map>
#include <chrono>
#include <random>

// Timer
std::chrono::steady_clock::time_point program_start_time;
std::chrono::milliseconds time_limit_ms(1850); 

// Global problem parameters and query counter/cache
int N_items_global, D_groups_global, Q_total_global;
int queries_made = 0;

std::map<std::pair<int, int>, char> comparison_results_cache_1v1;
std::map<int, std::map<std::pair<int, int>, char>> comparison_results_cache_1v2_specific;

std::mt19937 rng_engine;

// Function to perform a query via standard I/O
char perform_query_actual(const std::vector<int>& L_items, const std::vector<int>& R_items) {
    queries_made++;
    // Debug: #c assignments_array[0] ... assignments_array[N-1]
    // std::cout << "# Query " << queries_made << std::endl;
    std::cout << L_items.size() << " " << R_items.size();
    for (int item_idx : L_items) {
        std::cout << " " << item_idx;
    }
    for (int item_idx : R_items) {
        std::cout << " " << item_idx;
    }
    std::cout << std::endl;

    char result_char;
    std::cin >> result_char;
    return result_char;
}

char compare_single_items(int item_idx1, int item_idx2) {
    if (item_idx1 == item_idx2) return '=';

    std::pair<int, int> query_pair_key = {std::min(item_idx1, item_idx2), std::max(item_idx1, item_idx2)};

    auto it = comparison_results_cache_1v1.find(query_pair_key);
    if (it != comparison_results_cache_1v1.end()) {
        char cached_res = it->second;
        if (item_idx1 == query_pair_key.first) return cached_res;
        return (cached_res == '<' ? '>' : (cached_res == '>' ? '<' : '='));
    }

    if (queries_made >= Q_total_global) {
        return '='; 
    }

    char res_direct = perform_query_actual({item_idx1}, {item_idx2});

    if (item_idx1 < item_idx2) {
        comparison_results_cache_1v1[query_pair_key] = res_direct;
    } else { 
        char reversed_res = (res_direct == '<' ? '>' : (res_direct == '>' ? '<' : '='));
        comparison_results_cache_1v1[query_pair_key] = reversed_res;
    }
    return res_direct;
}

char compare_1v2_items_specific(int item_curr, int item_prev, int item_s_aux) {
    // Assuming item_curr, item_prev, item_s_aux are distinct indices as per problem context
    // L = {item_curr}, R = {item_prev, item_s_aux}
    // L and R must be disjoint, already true. Each set non-empty.
    // Items within R must be distinct (item_prev != item_s_aux). This is handled by caller logic in X_j estimation.
    
    std::pair<int, int> R_pair_key = {std::min(item_prev, item_s_aux), std::max(item_prev, item_s_aux)};

    auto it_LHS = comparison_results_cache_1v2_specific.find(item_curr);
    if (it_LHS != comparison_results_cache_1v2_specific.end()) {
        auto it_RHS = it_LHS->second.find(R_pair_key);
        if (it_RHS != it_LHS->second.end()) {
            return it_RHS->second;
        }
    }

    if (queries_made >= Q_total_global) {
        return '='; 
    }

    char res_direct = perform_query_actual({item_curr}, {item_prev, item_s_aux});
    comparison_results_cache_1v2_specific[item_curr][R_pair_key] = res_direct;
    return res_direct;
}

void merge_for_sort(std::vector<int>& items_to_sort, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    std::vector<int> L_half(n1), R_half(n2);
    for (int i = 0; i < n1; i++) L_half[i] = items_to_sort[left + i];
    for (int j = 0; j < n2; j++) R_half[j] = items_to_sort[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        char cmp_res = compare_single_items(L_half[i], R_half[j]);
        if (cmp_res == '<' || cmp_res == '=') {
            items_to_sort[k++] = L_half[i++];
        } else { 
            items_to_sort[k++] = R_half[j++];
        }
    }
    while (i < n1) items_to_sort[k++] = L_half[i++];
    while (j < n2) items_to_sort[k++] = R_half[j++];
}

void merge_sort_items(std::vector<int>& items_to_sort, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort_items(items_to_sort, left, mid);
        merge_sort_items(items_to_sort, mid + 1, right);
        merge_for_sort(items_to_sort, left, mid, right);
    }
}

long long BASE_WEIGHT = 100000; 

double estimate_log2(double val) {
    if (val <= 1.0) return 0.0;
    return std::log2(val);
}

int calculate_estimated_query_cost(int N_val, int k_pivots_val) {
    if (k_pivots_val <= 0) return 0;
    if (k_pivots_val == 1) {
        return (N_val > 1) ? (N_val - 1) : 0;
    }

    double cost = 0;
    cost += static_cast<double>(k_pivots_val) * estimate_log2(static_cast<double>(k_pivots_val)); 
    for (int j = 2; j < k_pivots_val; ++j) { 
        if (j-1 > 0) cost += estimate_log2(static_cast<double>(j - 1)); 
    }
    cost += static_cast<double>(N_val - k_pivots_val) * estimate_log2(static_cast<double>(k_pivots_val)); 
    return static_cast<int>(std::ceil(cost));
}

double calculate_variance_from_sums(double sum_sq_group_totals, double total_weight_double, int D_val) {
    if (D_val <= 0) return 1e18; 
    double mean_weight = total_weight_double / D_val;
    double variance = sum_sq_group_totals / D_val - mean_weight * mean_weight;
    return std::max(0.0, variance); 
}


int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    program_start_time = std::chrono::steady_clock::now();
    uint64_t random_seed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    rng_engine.seed(random_seed);

    std::cin >> N_items_global >> D_groups_global >> Q_total_global;

    std::vector<long long> estimated_weights(N_items_global);

    int k_pivots_chosen = (N_items_global > 0) ? 1 : 0;
    if (N_items_global > 1) { 
        for (int cur_k_val = N_items_global; cur_k_val >= 1; --cur_k_val) {
            if (calculate_estimated_query_cost(N_items_global, cur_k_val) <= Q_total_global) {
                k_pivots_chosen = cur_k_val;
                break;
            }
        }
    }
    k_pivots_chosen = std::min(k_pivots_chosen, N_items_global); 
    if (N_items_global == 0) k_pivots_chosen = 0;


    std::vector<int> pivot_item_indices(k_pivots_chosen);
    if (k_pivots_chosen > 0) {
        std::vector<int> all_item_indices_temp(N_items_global);
        std::iota(all_item_indices_temp.begin(), all_item_indices_temp.end(), 0);
        std::shuffle(all_item_indices_temp.begin(), all_item_indices_temp.end(), rng_engine);
        for (int i = 0; i < k_pivots_chosen; ++i) pivot_item_indices[i] = all_item_indices_temp[i];
    }
    
    std::vector<int> sorted_pivot_item_indices = pivot_item_indices;

    // Factors from previous attempt (more aggressive & symmetric):
    const int FACTOR_GT_NUM = 200; 
    const int FACTOR_LT_NUM = 50;  
    const int FACTOR_XJ_FALLBACK_NUM = 100; 

    if (k_pivots_chosen == 0) { 
        for (int i = 0; i < N_items_global; ++i) estimated_weights[i] = BASE_WEIGHT;
    } else if (k_pivots_chosen == 1) { 
        estimated_weights[pivot_item_indices[0]] = BASE_WEIGHT;
        for (int i = 0; i < N_items_global; ++i) {
            if (i == pivot_item_indices[0]) continue;
            char res = compare_single_items(i, pivot_item_indices[0]);
            if (res == '=') estimated_weights[i] = BASE_WEIGHT;
            else if (res == '<') estimated_weights[i] = std::max(1LL, BASE_WEIGHT * FACTOR_LT_NUM / 100); 
            else estimated_weights[i] = std::max(1LL, BASE_WEIGHT * FACTOR_GT_NUM / 100);
        }
    } else { // k_pivots_chosen >= 2
        merge_sort_items(sorted_pivot_item_indices, 0, k_pivots_chosen - 1);
        
        int p0_idx = sorted_pivot_item_indices[0];
        estimated_weights[p0_idx] = BASE_WEIGHT;
        
        int p1_idx = sorted_pivot_item_indices[1];
        char res_p1_vs_p0 = compare_single_items(p1_idx, p0_idx);

        if (res_p1_vs_p0 == '=') {
            estimated_weights[p1_idx] = estimated_weights[p0_idx];
        } else if (res_p1_vs_p0 == '<') { 
            estimated_weights[p1_idx] = std::max(1LL, estimated_weights[p0_idx] * FACTOR_LT_NUM / 100);
        } else { 
            estimated_weights[p1_idx] = std::max(1LL, estimated_weights[p0_idx] * FACTOR_GT_NUM / 100); 
        }
        // Ensure monotonicity and strictness if comparison was strict
        if (estimated_weights[p1_idx] < estimated_weights[p0_idx]) { 
             estimated_weights[p1_idx] = estimated_weights[p0_idx];
        }
        if (res_p1_vs_p0 == '>' && estimated_weights[p1_idx] == estimated_weights[p0_idx]) { 
             estimated_weights[p1_idx] = estimated_weights[p0_idx] + 1;
        }

        const long long MAX_XJ_INITIAL_HIGH_BOUND = BASE_WEIGHT * (1LL * N_items_global / std::max(1, D_groups_global) + 10); // Increased +5 to +10 for safety margin

        for (int j = 2; j < k_pivots_chosen; ++j) {
            int current_pivot_idx = sorted_pivot_item_indices[j];
            int prev_pivot_idx = sorted_pivot_item_indices[j-1];

            char res_curr_vs_prev = compare_single_items(current_pivot_idx, prev_pivot_idx);
            if (res_curr_vs_prev == '=') {
                estimated_weights[current_pivot_idx] = estimated_weights[prev_pivot_idx];
            } else if (res_curr_vs_prev == '<') { 
                estimated_weights[current_pivot_idx] = std::max(1LL, estimated_weights[prev_pivot_idx] * FACTOR_LT_NUM / 100);
            } else { 
                long long X_low_bound_val = 1; 
                long long X_high_bound_val = MAX_XJ_INITIAL_HIGH_BOUND; 
                bool x_low_modified = false;
                bool x_high_modified = false;

                int s_search_low_arr_idx = 0, s_search_high_arr_idx = j - 2;
                
                int num_s_candidates = (s_search_high_arr_idx - s_search_low_arr_idx + 1);
                int queries_for_this_Xj = 0;
                if (num_s_candidates > 0) {
                     queries_for_this_Xj = static_cast<int>(std::ceil(estimate_log2(static_cast<double>(num_s_candidates))));
                     if (num_s_candidates == 1) queries_for_this_Xj = 1; 
                }

                for(int bs_iter = 0; bs_iter < queries_for_this_Xj && queries_made < Q_total_global; ++bs_iter) {
                    if (s_search_low_arr_idx > s_search_high_arr_idx) break; 
                    int s_mid_arr_idx = s_search_low_arr_idx + (s_search_high_arr_idx - s_search_low_arr_idx) / 2;
                    int item_s_aux_idx = sorted_pivot_item_indices[s_mid_arr_idx];
                    
                    // Skip if s_aux is same as prev_pivot_idx; R items must be distinct for query.
                    // This should not happen if s_aux is chosen from p0...p_{j-2} and prev_pivot is p_{j-1}.
                    // if (item_s_aux_idx == prev_pivot_idx) continue; // Should not be necessary
                                        
                    char res_1v2 = compare_1v2_items_specific(current_pivot_idx, prev_pivot_idx, item_s_aux_idx);

                    if (res_1v2 == '=') { 
                        X_low_bound_val = X_high_bound_val = estimated_weights[item_s_aux_idx];
                        x_low_modified = x_high_modified = true;
                        break; 
                    } else if (res_1v2 == '<') { 
                        X_high_bound_val = estimated_weights[item_s_aux_idx];
                        x_high_modified = true;
                        s_search_high_arr_idx = s_mid_arr_idx - 1;
                    } else { // res_1v2 == '>'
                        X_low_bound_val = estimated_weights[item_s_aux_idx];
                        x_low_modified = true;
                        s_search_low_arr_idx = s_mid_arr_idx + 1;
                    }
                }
                
                long long estimated_X_j;
                if (x_low_modified && !x_high_modified) { // X_j > X_low_bound_val (max s_aux smaller than X_j)
                    estimated_X_j = X_low_bound_val * FACTOR_GT_NUM / 100;
                } else if (!x_low_modified && x_high_modified) { // X_j < X_high_bound_val (min s_aux larger than X_j)
                    estimated_X_j = X_high_bound_val * FACTOR_LT_NUM / 100;
                } else if (x_low_modified && x_high_modified) { // X_j is bracketed
                    // Reverted to ARITHMETIC MEAN for X_j
                    estimated_X_j = (X_low_bound_val + X_high_bound_val) / 2;
                } else { // Fallback if binary search didn't narrow down X_j
                    estimated_X_j = estimated_weights[prev_pivot_idx] * FACTOR_XJ_FALLBACK_NUM / 100;
                    if (estimated_weights[prev_pivot_idx] > 0 && estimated_X_j == 0) estimated_X_j = 1; 
                    else if (estimated_weights[prev_pivot_idx] == 0) { 
                         estimated_X_j = std::max(1LL, BASE_WEIGHT * FACTOR_XJ_FALLBACK_NUM / 100);
                    }
                }
                estimated_X_j = std::max(1LL, estimated_X_j); 
                
                estimated_weights[current_pivot_idx] = estimated_weights[prev_pivot_idx] + estimated_X_j;
            }
            // Ensure monotonicity and strictness
            if(estimated_weights[current_pivot_idx] < estimated_weights[prev_pivot_idx]) {
                 estimated_weights[current_pivot_idx] = estimated_weights[prev_pivot_idx];
            }
            if (res_curr_vs_prev == '>' && estimated_weights[current_pivot_idx] == estimated_weights[prev_pivot_idx]) {
                estimated_weights[current_pivot_idx] = estimated_weights[prev_pivot_idx] + 1;
            }
        }

        // Estimate weights for non-pivot items
        for (int i=0; i<N_items_global; ++i) {
            bool is_pivot_flag = false; 
            for(int p_idx_val=0; p_idx_val<k_pivots_chosen; ++p_idx_val) {
                if(sorted_pivot_item_indices[p_idx_val] == i) {
                    is_pivot_flag = true;
                    break;
                }
            }
            if (is_pivot_flag) continue; 

            int bs_low_arr_idx = 0, bs_high_arr_idx = k_pivots_chosen - 1;
            int found_pivot_idx_for_eq = -1; 

            while(bs_low_arr_idx <= bs_high_arr_idx) {
                if (queries_made >= Q_total_global && found_pivot_idx_for_eq == -1) break; // Stop if out of queries unless already found exact
                int mid_p_arr_idx = bs_low_arr_idx + (bs_high_arr_idx - bs_low_arr_idx) / 2;
                char res_item_vs_p = compare_single_items(i, sorted_pivot_item_indices[mid_p_arr_idx]);

                if (res_item_vs_p == '=') {
                    found_pivot_idx_for_eq = mid_p_arr_idx;
                    break;
                } else if (res_item_vs_p == '<') {
                    bs_high_arr_idx = mid_p_arr_idx - 1;
                } else { 
                    bs_low_arr_idx = mid_p_arr_idx + 1;
                }
            }
            
            if (found_pivot_idx_for_eq != -1) { 
                estimated_weights[i] = estimated_weights[sorted_pivot_item_indices[found_pivot_idx_for_eq]];
                continue;
            }
            
            int insert_pos_arr_idx = bs_low_arr_idx; 
            
            if (insert_pos_arr_idx == 0) { // Smaller than p0
                long long w_p0 = estimated_weights[sorted_pivot_item_indices[0]];
                if (k_pivots_chosen >= 2) { 
                    long long w_p1 = estimated_weights[sorted_pivot_item_indices[1]];
                    // Ensure w_p1 != 0 before division, and w_p0 must be < w_p1 for this extrapolation to make sense
                    if (w_p1 > w_p0 && w_p0 > 0 && w_p1 != 0) { // w_p1 should not be 0 if weights are >=1
                         estimated_weights[i] = std::max(1LL, w_p0 * w_p0 / w_p1); 
                    } else { 
                        estimated_weights[i] = std::max(1LL, w_p0 * FACTOR_LT_NUM / 100);
                    }
                } else { // Only p0 exists
                     estimated_weights[i] = std::max(1LL, w_p0 * FACTOR_LT_NUM / 100);
                }
            } else if (insert_pos_arr_idx == k_pivots_chosen) { // Larger than p_{k-1}
                long long w_pk_1 = estimated_weights[sorted_pivot_item_indices[k_pivots_chosen-1]];
                 if (k_pivots_chosen >= 2) { 
                    long long w_pk_2 = estimated_weights[sorted_pivot_item_indices[k_pivots_chosen-2]];
                    // Ensure w_pk_2 != 0 and w_pk_2 < w_pk_1
                    if (w_pk_1 > w_pk_2 && w_pk_2 > 0 && w_pk_2 != 0) { // w_pk_2 should not be 0
                        estimated_weights[i] = std::max(1LL, w_pk_1 * w_pk_1 / w_pk_2); 
                    } else { 
                        estimated_weights[i] = std::max(1LL, w_pk_1 * FACTOR_GT_NUM / 100);
                    }
                 } else { // Only p0 exists (which is p_{k-1} here)
                     estimated_weights[i] = std::max(1LL, w_pk_1 * FACTOR_GT_NUM / 100);
                 }
            } else { // Between p_{idx-1} and p_{idx}
                long long w_prev_p = estimated_weights[sorted_pivot_item_indices[insert_pos_arr_idx-1]];
                long long w_next_p = estimated_weights[sorted_pivot_item_indices[insert_pos_arr_idx]];
                // Geometric mean for interpolation is generally preferred for exponential-like data
                if (w_prev_p > 0 && w_next_p > 0) { 
                    estimated_weights[i] = static_cast<long long>(std::sqrt(static_cast<double>(w_prev_p) * w_next_p));
                } else { // Fallback for safety or if one weight is zero (should be >=1)
                    estimated_weights[i] = (w_prev_p + w_next_p) / 2;
                }
                // Ensure estimate is within the bounds of the two pivots it's between
                estimated_weights[i] = std::max(w_prev_p, estimated_weights[i]);
                estimated_weights[i] = std::min(w_next_p, estimated_weights[i]);
            }
            if (estimated_weights[i] <=0) estimated_weights[i] = 1; 
        }
    }

    // Final check: all weights must be at least 1.
    for(int i=0; i<N_items_global; ++i) {
        if (estimated_weights[i] <= 0) { 
            // This state indicates a flaw in estimation logic or extreme case.
            // Fallback to a reasonable default like BASE_WEIGHT or 1.
            // Previous version used BASE_WEIGHT. Smallest possible is 1.
            // Using 1 might be safer if other weights are also small.
            // However, if most are large, BASE_WEIGHT might be better.
            // Sticking to previous fallback.
            estimated_weights[i] = BASE_WEIGHT; 
        }
    }
    
    // Exhaust remaining queries
    int dummy_item_0_idx = 0;
    int dummy_item_1_idx = 1; 
    // N_items_global >= 30, so 0 and 1 are valid and distinct indices.
    while(queries_made < Q_total_global) {
        perform_query_actual({dummy_item_0_idx}, {dummy_item_1_idx});
        // Cycle one of the items to make queries slightly different, though not critical for correctness
        dummy_item_1_idx = (dummy_item_1_idx + 1) % N_items_global;
        if (dummy_item_1_idx == dummy_item_0_idx) { // Ensure distinctness
            dummy_item_1_idx = (dummy_item_1_idx + 1) % N_items_global;
        }
    }
    
    // --- Assignment Phase: Greedy followed by Simulated Annealing ---
    std::vector<int> assignment_array(N_items_global);
    std::vector<long long> group_sums_array(D_groups_global, 0);
    long long total_sum_est_val = 0;

    std::vector<std::vector<int>> group_items_indices(D_groups_global);
    std::vector<int> item_pos_in_group_vector(N_items_global); 

    std::vector<std::pair<long long, int>> items_sorted_for_greedy(N_items_global);
    for(int i=0; i<N_items_global; ++i) {
        items_sorted_for_greedy[i] = {-estimated_weights[i], i}; 
    }
    std::sort(items_sorted_for_greedy.begin(), items_sorted_for_greedy.end());

    for(int i=0; i<N_items_global; ++i) {
        int item_actual_idx = items_sorted_for_greedy[i].second;
        long long item_w = estimated_weights[item_actual_idx];
        int best_grp_current = 0;
        if (D_groups_global > 1) { 
            long long min_sum_in_group = group_sums_array[0];
            // Small optimization: if multiple groups have same min_sum, pick one randomly or by index
            // Current logic picks smallest index. This is fine.
            for(int j=1; j<D_groups_global; ++j) {
                if (group_sums_array[j] < min_sum_in_group) {
                    min_sum_in_group = group_sums_array[j];
                    best_grp_current = j;
                }
            }
        }
        assignment_array[item_actual_idx] = best_grp_current;
        group_sums_array[best_grp_current] += item_w;
        group_items_indices[best_grp_current].push_back(item_actual_idx); 
        item_pos_in_group_vector[item_actual_idx] = group_items_indices[best_grp_current].size() - 1;
        total_sum_est_val += item_w;
    }
    
    double current_sum_sq_group_totals = 0;
    for(long long s : group_sums_array) {
        current_sum_sq_group_totals += static_cast<double>(s) * s;
    }
    double current_var = calculate_variance_from_sums(current_sum_sq_group_totals, static_cast<double>(total_sum_est_val), D_groups_global);
    
    // SA Parameters
    double T_initial_factor = 0.25; 
    double T = std::max(1.0, current_var * T_initial_factor); 
    if (total_sum_est_val > 0 && current_var < 1e-9 && D_groups_global > 0) { 
        T = std::max(1.0, static_cast<double>(total_sum_est_val) / std::max(1,N_items_global) * 0.1); 
    } else if (total_sum_est_val == 0 && D_groups_global > 0) { 
        T = std::max(1.0, static_cast<double>(BASE_WEIGHT) * N_items_global / D_groups_global * 0.01 ); 
    }
    if (D_groups_global <= 1) T = 0; 
    
    double cool_rate = 0.9999; 
    int sa_iters_count = 0;
    std::uniform_real_distribution<double> unif_dist(0.0, 1.0);
    int no_improvement_streak = 0;
    const int REHEAT_STREAK_THRESH_FACTOR = N_items_global > 50 ? 10 : 20; 
    const int CHECK_TIME_INTERVAL = 256;


    while(D_groups_global > 1 && N_items_global > 0) { 
        sa_iters_count++;
        if (sa_iters_count % CHECK_TIME_INTERVAL == 0) { 
            auto time_now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(time_now - program_start_time) >= time_limit_ms) {
                break; 
            }
            T *= cool_rate; 
            if (no_improvement_streak > N_items_global * REHEAT_STREAK_THRESH_FACTOR && T < current_var * 0.05 && current_var > 1.0 + 1e-9) { 
                 T = std::max(1.0, current_var * T_initial_factor * 0.5); 
                 no_improvement_streak = 0;
            }
        }
        if (T < 1e-12 && current_var > 1e-9) T = 1e-9; // Floor T if var high but T too low
        if (T < 1e-12 && current_var < (1.0 + 1e-9)) break; // Converged or T too low


        int move_type_rand_val = rng_engine();
        // Adjust probability of swap vs relocate: 1/3 swap, 2/3 relocate
        bool try_swap_move = ( (move_type_rand_val % 3 == 0) ); 

        if (!try_swap_move) { // Relocate an item
            if (N_items_global == 0) continue;
            int item_to_move_idx = rng_engine() % N_items_global;
            int old_grp_idx = assignment_array[item_to_move_idx];
            
            if (D_groups_global <=1) continue; 
            int new_grp_idx = rng_engine() % D_groups_global;
            while(new_grp_idx == old_grp_idx) new_grp_idx = rng_engine() % D_groups_global;
            
            long long item_w_val = estimated_weights[item_to_move_idx];
            
            long long old_sum_grp_A = group_sums_array[old_grp_idx];
            long long old_sum_grp_B = group_sums_array[new_grp_idx];
            long long new_sum_grp_A = old_sum_grp_A - item_w_val;
            long long new_sum_grp_B = old_sum_grp_B + item_w_val;

            double new_sum_sq_group_totals_cand = current_sum_sq_group_totals;
            new_sum_sq_group_totals_cand -= static_cast<double>(old_sum_grp_A)*old_sum_grp_A + static_cast<double>(old_sum_grp_B)*old_sum_grp_B;
            new_sum_sq_group_totals_cand += static_cast<double>(new_sum_grp_A)*new_sum_grp_A + static_cast<double>(new_sum_grp_B)*new_sum_grp_B;
            double new_var = calculate_variance_from_sums(new_sum_sq_group_totals_cand, static_cast<double>(total_sum_est_val), D_groups_global);
            
            double delta_V = new_var - current_var;

            if (delta_V < 0 || (T > 1e-12 && unif_dist(rng_engine) < std::exp(-delta_V / T)) ) { 
                current_var = new_var;
                current_sum_sq_group_totals = new_sum_sq_group_totals_cand;
                group_sums_array[old_grp_idx] = new_sum_grp_A;
                group_sums_array[new_grp_idx] = new_sum_grp_B;
                assignment_array[item_to_move_idx] = new_grp_idx; 
                
                int pos_in_old_vec = item_pos_in_group_vector[item_to_move_idx];
                if (!group_items_indices[old_grp_idx].empty()) { 
                    int last_item_in_old_grp_vec = group_items_indices[old_grp_idx].back();
                    if (item_to_move_idx != last_item_in_old_grp_vec) { 
                         group_items_indices[old_grp_idx][pos_in_old_vec] = last_item_in_old_grp_vec; 
                         item_pos_in_group_vector[last_item_in_old_grp_vec] = pos_in_old_vec; 
                    }
                    group_items_indices[old_grp_idx].pop_back();
                }

                group_items_indices[new_grp_idx].push_back(item_to_move_idx);
                item_pos_in_group_vector[item_to_move_idx] = group_items_indices[new_grp_idx].size() - 1;

                if (delta_V < -1e-9) no_improvement_streak = 0; else no_improvement_streak++;
            } else { 
                no_improvement_streak++;
            }
        } else { // Try swap move
            if (D_groups_global <= 1) continue; 

            int grp1_idx = rng_engine() % D_groups_global;
            int grp2_idx = rng_engine() % D_groups_global;
            while(grp2_idx == grp1_idx) grp2_idx = rng_engine() % D_groups_global;

            if(group_items_indices[grp1_idx].empty() || group_items_indices[grp2_idx].empty()) {
                no_improvement_streak++;
                continue; 
            }

            int item1_original_idx = group_items_indices[grp1_idx][rng_engine() % group_items_indices[grp1_idx].size()];
            int item2_original_idx = group_items_indices[grp2_idx][rng_engine() % group_items_indices[grp2_idx].size()];

            long long w1 = estimated_weights[item1_original_idx];
            long long w2 = estimated_weights[item2_original_idx];

            // If w1 == w2, swap has no effect on sums, so delta_V = 0.
            // This move is only useful if it helps escape local minimum for other reasons,
            // or if it's accepted by chance and enables further moves.
            // If w1 == w2, delta_V will be 0. Acceptance depends on T (always if T>0).
            // No need to explicitly check for w1==w2.

            long long old_sum_grp1 = group_sums_array[grp1_idx];
            long long old_sum_grp2 = group_sums_array[grp2_idx];
            long long new_sum_grp1 = old_sum_grp1 - w1 + w2;
            long long new_sum_grp2 = old_sum_grp2 - w2 + w1;

            double new_sum_sq_group_totals_cand = current_sum_sq_group_totals;
            new_sum_sq_group_totals_cand -= static_cast<double>(old_sum_grp1)*old_sum_grp1 + static_cast<double>(old_sum_grp2)*old_sum_grp2;
            new_sum_sq_group_totals_cand += static_cast<double>(new_sum_grp1)*new_sum_grp1 + static_cast<double>(new_sum_grp2)*new_sum_grp2;
            double new_var = calculate_variance_from_sums(new_sum_sq_group_totals_cand, static_cast<double>(total_sum_est_val), D_groups_global);
            
            double delta_V = new_var - current_var;

            if (delta_V < 0 || (T > 1e-12 && unif_dist(rng_engine) < std::exp(-delta_V / T)) ) { 
                current_var = new_var;
                current_sum_sq_group_totals = new_sum_sq_group_totals_cand;
                group_sums_array[grp1_idx] = new_sum_grp1;
                group_sums_array[grp2_idx] = new_sum_grp2;
                
                assignment_array[item1_original_idx] = grp2_idx;
                assignment_array[item2_original_idx] = grp1_idx;

                // Update item tracking structures
                int pos1_in_G1 = item_pos_in_group_vector[item1_original_idx];
                // group_items_indices[grp1_idx] cannot be empty here as item1 was picked from it.
                int back1_of_G1 = group_items_indices[grp1_idx].back();
                if (item1_original_idx != back1_of_G1) {
                    group_items_indices[grp1_idx][pos1_in_G1] = back1_of_G1;
                    item_pos_in_group_vector[back1_of_G1] = pos1_in_G1;
                }
                group_items_indices[grp1_idx].pop_back();
                
                int pos2_in_G2 = item_pos_in_group_vector[item2_original_idx];
                int back2_of_G2 = group_items_indices[grp2_idx].back();
                if (item2_original_idx != back2_of_G2) {
                    group_items_indices[grp2_idx][pos2_in_G2] = back2_of_G2;
                    item_pos_in_group_vector[back2_of_G2] = pos2_in_G2;
                }
                group_items_indices[grp2_idx].pop_back();
                
                group_items_indices[grp2_idx].push_back(item1_original_idx);
                item_pos_in_group_vector[item1_original_idx] = group_items_indices[grp2_idx].size() - 1;
                
                group_items_indices[grp1_idx].push_back(item2_original_idx);
                item_pos_in_group_vector[item2_original_idx] = group_items_indices[grp1_idx].size() - 1;

                if (delta_V < -1e-9) no_improvement_streak = 0; else no_improvement_streak++;
            } else { 
                no_improvement_streak++;
            }
        }
    }

    for (int i = 0; i < N_items_global; ++i) {
        std::cout << assignment_array[i] << (i == N_items_global - 1 ? "" : " ");
    }
    std::cout << std::endl;

    return 0;
}
# EVOLVE-BLOCK-END