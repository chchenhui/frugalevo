# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono> // For seeding RNG
// #include <iomanip> // For debugging output

// Constants
const int GRID_SIZE = 10;
const int NUM_TURNS = 100;
const int NUM_FLAVORS = 3; // Flavors are 1, 2, 3

// Directions: F, B, L, R (Up, Down, Left, Right on typical grid with (0,0) top-left)
const int DR[] = {-1, 1, 0, 0}; 
const int DC[] = {0, 0, -1, 1}; 
const char DIR_CHARS[] = {'F', 'B', 'L', 'R'};
const int NUM_DIRECTIONS = 4;

// Global data initialized once
std::array<int, NUM_TURNS> G_FLAVOR_SEQUENCE;
std::array<int, NUM_FLAVORS + 1> G_flavor_total_counts; 
std::array<std::pair<int, int>, NUM_FLAVORS + 1> G_target_col_ranges; 
std::array<bool, NUM_FLAVORS + 1> G_flavor_active; 

// Lookahead parameters
const int MAX_LOOKAHEAD_DEPTH = 2; 
// Keep enough margin under the official 2-second limit for interactive I/O and
// host-load variance.  The original {23, 9} frequently exhausted the CPU limit
// before emitting all 100 actions on this benchmark host.
static constexpr std::array<int, MAX_LOOKAHEAD_DEPTH> NUM_SAMPLES_CONFIG = {18, 7};


struct XorshiftRNG {
    uint64_t x;
    XorshiftRNG() : x(std::chrono::steady_clock::now().time_since_epoch().count()) {}
    
    uint64_t next() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return x;
    }
    
    int uniform_int(int min_val, int max_val) {
        if (min_val > max_val) return min_val; 
        if (min_val == max_val) return min_val;
        uint64_t range = static_cast<uint64_t>(max_val) - min_val + 1;
        return min_val + static_cast<int>(next() % range);
    }
};
XorshiftRNG rng; 


struct Candy {
    int r, c, flavor;
};

struct GameState {
    std::array<std::array<int, GRID_SIZE>, GRID_SIZE> board; 
    std::vector<Candy> candies_list; 
    int turn_num_1_indexed; 

    GameState() : turn_num_1_indexed(0) {
        for (int i = 0; i < GRID_SIZE; ++i) {
            board[i].fill(0); 
        }
        candies_list.reserve(NUM_TURNS);
    }

    GameState(const GameState& other) = default; 
    GameState& operator=(const GameState& other) = default;
    GameState(GameState&& other) noexcept = default;
    GameState& operator=(GameState&& other) noexcept = default;

    void place_candy(int r, int c, int flavor) {
        board[r][c] = flavor;
        candies_list.push_back({r, c, flavor});
    }

    std::pair<int, int> find_pth_empty_cell(int p_1_indexed) const {
        int count = 0;
        for (int r_idx = 0; r_idx < GRID_SIZE; ++r_idx) {
            for (int c_idx = 0; c_idx < GRID_SIZE; ++c_idx) {
                if (board[r_idx][c_idx] == 0) { 
                    count++;
                    if (count == p_1_indexed) {
                        return {r_idx, c_idx};
                    }
                }
            }
        }
        return {-1, -1}; 
    }
    
    int count_empty_cells() const {
        return GRID_SIZE * GRID_SIZE - static_cast<int>(candies_list.size());
    }
    
    void apply_tilt(int dir_idx) { 
        if (dir_idx == 0) { // F (Up)
            for (int c = 0; c < GRID_SIZE; ++c) {
                int current_write_r = 0;
                for (int r = 0; r < GRID_SIZE; ++r) {
                    if (board[r][c] != 0) {
                        if (r != current_write_r) {
                            board[current_write_r][c] = board[r][c];
                            board[r][c] = 0;
                        }
                        current_write_r++;
                    }
                }
            }
        } else if (dir_idx == 1) { // B (Down)
            for (int c = 0; c < GRID_SIZE; ++c) {
                int current_write_r = GRID_SIZE - 1;
                for (int r = GRID_SIZE - 1; r >= 0; --r) {
                    if (board[r][c] != 0) {
                        if (r != current_write_r) {
                            board[current_write_r][c] = board[r][c];
                            board[r][c] = 0;
                        }
                        current_write_r--;
                    }
                }
            }
        } else if (dir_idx == 2) { // L (Left)
            for (int r = 0; r < GRID_SIZE; ++r) {
                int current_write_c = 0;
                for (int c = 0; c < GRID_SIZE; ++c) {
                    if (board[r][c] != 0) {
                        if (c != current_write_c) {
                            board[r][current_write_c] = board[r][c];
                            board[r][c] = 0;
                        }
                        current_write_c++;
                    }
                }
            }
        } else { // R (Right, dir_idx == 3)
            for (int r = 0; r < GRID_SIZE; ++r) {
                int current_write_c = GRID_SIZE - 1;
                for (int c = GRID_SIZE - 1; c >= 0; --c) {
                    if (board[r][c] != 0) {
                        if (c != current_write_c) {
                            board[r][current_write_c] = board[r][c];
                            board[r][c] = 0;
                        }
                        current_write_c--;
                    }
                }
            }
        }
        rebuild_candies_list_from_board();
    }

    void rebuild_candies_list_from_board() {
        candies_list.clear(); 
        for (int r_idx = 0; r_idx < GRID_SIZE; ++r_idx) {
            for (int c_idx = 0; c_idx < GRID_SIZE; ++c_idx) {
                if (board[r_idx][c_idx] != 0) {
                    candies_list.push_back({r_idx, c_idx, board[r_idx][c_idx]});
                }
            }
        }
    }

    long long calculate_sum_sq_comp_size() const {
        long long total_sq_sum = 0;
        std::array<std::array<bool, GRID_SIZE>, GRID_SIZE> visited;
        for (int i = 0; i < GRID_SIZE; ++i) visited[i].fill(false);

        std::array<std::pair<int, int>, GRID_SIZE * GRID_SIZE> q_arr; 

        for (int r_start = 0; r_start < GRID_SIZE; ++r_start) {
            for (int c_start = 0; c_start < GRID_SIZE; ++c_start) {
                if (board[r_start][c_start] != 0 && !visited[r_start][c_start]) {
                    int current_flavor = board[r_start][c_start];
                    long long current_comp_size = 0;
                    
                    q_arr[0] = {r_start, c_start};
                    visited[r_start][c_start] = true;
                    int head = 0; 
                    int tail = 1; 
                    
                    while(head < tail){
                        current_comp_size++; 
                        const std::pair<int,int>& curr_cell = q_arr[head];
                        const int curr_r = curr_cell.first;
                        const int curr_c = curr_cell.second;
                        head++; 

                        for (int i = 0; i < NUM_DIRECTIONS; ++i) {
                            int nr = curr_r + DR[i];
                            int nc = curr_c + DC[i];
                            if (nr >= 0 && nr < GRID_SIZE && nc >= 0 && nc < GRID_SIZE &&
                                !visited[nr][nc] && board[nr][nc] == current_flavor) {
                                visited[nr][nc] = true;
                                q_arr[tail++] = {nr, nc}; 
                            }
                        }
                    }
                    total_sq_sum += current_comp_size * current_comp_size;
                }
            }
        }
        return total_sq_sum;
    }
    
    double calculate_distance_penalty_CoM() const {
        if (candies_list.empty()) return 0.0;

        std::array<double, NUM_FLAVORS + 1> sum_r; sum_r.fill(0.0);
        std::array<double, NUM_FLAVORS + 1> sum_c; sum_c.fill(0.0);
        std::array<int, NUM_FLAVORS + 1> counts; counts.fill(0);

        for (const auto& candy : candies_list) {
            counts[candy.flavor]++;
            sum_r[candy.flavor] += candy.r;
            sum_c[candy.flavor] += candy.c;
        }

        std::array<std::pair<double, double>, NUM_FLAVORS + 1> com_coords;
        for (int fl = 1; fl <= NUM_FLAVORS; ++fl) {
            if (counts[fl] > 0) {
                com_coords[fl] = {sum_r[fl] / counts[fl], sum_c[fl] / counts[fl]};
            }
        }

        double total_manhattan_dist_penalty = 0;
        for (const auto& candy : candies_list) {
            if (counts[candy.flavor] > 1) { 
                const auto& com = com_coords[candy.flavor];
                total_manhattan_dist_penalty += std::abs(static_cast<double>(candy.r) - com.first) + 
                                                std::abs(static_cast<double>(candy.c) - com.second);
            }
        }
        return total_manhattan_dist_penalty;
    }

    double calculate_region_penalty() const {
        if (candies_list.empty()) return 0.0;
        double penalty = 0.0;
        for (const auto& candy : candies_list) {
            if (!G_flavor_active[candy.flavor]) continue;

            const auto& range = G_target_col_ranges[candy.flavor];
            int min_target_c = range.first;
            int max_target_c = range.second;
            
            if (min_target_c > max_target_c) continue; 

            if (candy.c < min_target_c) {
                penalty += (min_target_c - candy.c);
            } else if (candy.c > max_target_c) {
                penalty += (candy.c - max_target_c);
            }
        }
        return penalty;
    }
    
    double calculate_edge_bonus() const {
        double bonus_val = 0.0;
        const double PER_CANDY_BONUS_FACTOR = 0.5; 

        for (const auto& candy : candies_list) {
            if (!G_flavor_active[candy.flavor]) continue;

            const auto& range = G_target_col_ranges[candy.flavor];
            int min_target_c = range.first;
            int max_target_c = range.second;
            
            if (min_target_c > max_target_c) continue; 

            bool in_correct_strip = (candy.c >= min_target_c && candy.c <= max_target_c);
            
            if (in_correct_strip) {
                if (candy.r == 0 || candy.r == GRID_SIZE - 1) { 
                    bonus_val += PER_CANDY_BONUS_FACTOR;
                }
                if ((candy.c == 0 && min_target_c == 0) || 
                    (candy.c == GRID_SIZE - 1 && max_target_c == GRID_SIZE - 1)) {
                     bonus_val += PER_CANDY_BONUS_FACTOR;
                }
            }
        }
        return bonus_val;
    }
        
    double evaluate() const { 
        if (candies_list.empty() && turn_num_1_indexed == 0) return 0.0;

        long long sum_sq_comp = calculate_sum_sq_comp_size();
        double dist_penalty_com = calculate_distance_penalty_CoM();
        double region_penalty_val = calculate_region_penalty();
        double edge_bonus_val = calculate_edge_bonus();
        
        double current_turn_double = static_cast<double>(turn_num_1_indexed);
        
        // Coefficients from Iteration 2 (best scoring), with small tweak to C
        double A_coeff_conn = 15.0 + 1.1 * current_turn_double; 
        double B_coeff_com_base = std::max(0.0, 170.0 - 1.7 * current_turn_double); 
        // Final iteration tweak for C_coeff_region_penalty_direct:
        double C_coeff_region_penalty_direct = std::max(2.0, 27.0 - 0.17 * current_turn_double); 
        double D_coeff_edge_bonus = 5.0 + 0.2 * current_turn_double; 
                
        return A_coeff_conn * sum_sq_comp 
             - B_coeff_com_base * dist_penalty_com 
             - C_coeff_region_penalty_direct * region_penalty_val
             + D_coeff_edge_bonus * edge_bonus_val;
    }
};

// Forward declaration
double eval_lookahead(const GameState& state_after_tilt, int turn_T_of_candy_just_processed, int depth_remaining);

/**
 * Chooses the best current tilt by stochastic lookahead, extending the search
 * through the exact terminal board when only a few candies remain.
 */
char decide_tilt_direction_logic(const GameState& current_gs_after_placement) {
    double best_overall_eval = std::numeric_limits<double>::lowest();
    int best_dir_idx = 0;
    int turn_T_for_lookahead_base = current_gs_after_placement.turn_num_1_indexed;

    // Near completion every remaining placement can be enumerated cheaply,
    // and the leaf value can be the exact contest objective.
    int lookahead_depth = MAX_LOOKAHEAD_DEPTH;
    if (turn_T_for_lookahead_base >= 97) {
        lookahead_depth = NUM_TURNS - turn_T_for_lookahead_base;
    }

    for (int i = 0; i < NUM_DIRECTIONS; ++i) {
        GameState gs_after_tilt_T = current_gs_after_placement;
        gs_after_tilt_T.apply_tilt(i);

        double value = eval_lookahead(
            gs_after_tilt_T, turn_T_for_lookahead_base, lookahead_depth
        );
        if (value > best_overall_eval) {
            best_overall_eval = value;
            best_dir_idx = i;
        }
    }
    return DIR_CHARS[best_dir_idx];
}


/**
 * Computes an expected best-response value using a randomly rotated systematic
 * sample of future empty-cell ranks; completed boards use the exact numerator.
 */
double eval_lookahead(const GameState& state_after_tilt, int turn_T_of_candy_just_processed, int depth_remaining) {
    // The score denominator is constant for one input, so this is the exact
    // objective at a full board up to a positive multiplicative constant.
    if (turn_T_of_candy_just_processed == NUM_TURNS) {
        return static_cast<double>(state_after_tilt.calculate_sum_sq_comp_size());
    }
    if (depth_remaining == 0) {
        return state_after_tilt.evaluate();
    }

    int num_empty = state_after_tilt.count_empty_cells();
    if (num_empty == 0) {
        return state_after_tilt.evaluate();
    }

    int next_candy_flavor = G_FLAVOR_SEQUENCE[turn_T_of_candy_just_processed];

    // During additional endgame plies, use the wider first-ply budget. Since
    // only a few cells remain, min(sample_count, num_empty) fully enumerates.
    int sample_count_this_depth =
        (depth_remaining >= 2 ? NUM_SAMPLES_CONFIG[0] : NUM_SAMPLES_CONFIG[1]);
    int actual_num_samples = std::min(sample_count_this_depth, num_empty);
    if (actual_num_samples == 0) {
        return state_after_tilt.evaluate();
    }

    // Cover separated ranks without replacement. Random rotation prevents a
    // persistent bias toward particular rows in row-major empty-cell order.
    int sample_offset = (actual_num_samples == num_empty)
        ? 0
        : rng.uniform_int(0, num_empty - 1);

    double sum_over_sampled_placements = 0.0;
    for (int s = 0; s < actual_num_samples; ++s) {
        int p_val_1_indexed_sample;
        if (actual_num_samples == num_empty) {
            p_val_1_indexed_sample = s + 1;
        } else {
            int sampled_rank_0_indexed =
                (sample_offset + (s * num_empty) / actual_num_samples) % num_empty;
            p_val_1_indexed_sample = sampled_rank_0_indexed + 1;
        }

        GameState S_after_placement = state_after_tilt;
        std::pair<int, int> candy_loc =
            S_after_placement.find_pth_empty_cell(p_val_1_indexed_sample);
        S_after_placement.place_candy(
            candy_loc.first, candy_loc.second, next_candy_flavor
        );
        S_after_placement.turn_num_1_indexed =
            turn_T_of_candy_just_processed + 1;

        double max_eval_for_this_placement = std::numeric_limits<double>::lowest();
        for (int dir_idx_next_tilt = 0; dir_idx_next_tilt < NUM_DIRECTIONS; ++dir_idx_next_tilt) {
            GameState S_after_next_tilt = S_after_placement;
            S_after_next_tilt.apply_tilt(dir_idx_next_tilt);
            double val = eval_lookahead(
                S_after_next_tilt,
                S_after_placement.turn_num_1_indexed,
                depth_remaining - 1
            );
            if (val > max_eval_for_this_placement) {
                max_eval_for_this_placement = val;
            }
        }
        sum_over_sampled_placements += max_eval_for_this_placement;
    }

    return sum_over_sampled_placements / actual_num_samples;
}


/**
 * Reads the flavor sequence and assigns vertical target strips, prioritizing
 * larger final flavor populations when integral strip widths leave leftovers.
 */
void initialize_global_data() {
    G_flavor_total_counts.fill(0);
    for (int t = 0; t < NUM_TURNS; ++t) {
        std::cin >> G_FLAVOR_SEQUENCE[t];
        G_flavor_total_counts[G_FLAVOR_SEQUENCE[t]]++;
    }

    G_flavor_active.fill(false);
    std::vector<std::pair<int, int>> sorter_flavor_count_id; 
    for (int fl = 1; fl <= NUM_FLAVORS; ++fl) {
        if (G_flavor_total_counts[fl] > 0) {
            G_flavor_active[fl] = true;
            sorter_flavor_count_id.push_back({G_flavor_total_counts[fl], fl});
        }
    }
    std::sort(sorter_flavor_count_id.begin(), sorter_flavor_count_id.end(), 
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        if (a.first != b.first) {
            return a.first > b.first; 
        }
        return a.second < b.second; 
    });

    std::vector<int> active_flavor_ids_sorted_by_priority;
    for(const auto& p : sorter_flavor_count_id) {
        active_flavor_ids_sorted_by_priority.push_back(p.second);
    }
    
    std::vector<int> assigned_widths(NUM_FLAVORS + 1, 0);
    int total_assigned_width_sum = 0;

    if (!active_flavor_ids_sorted_by_priority.empty()) {
        double total_candies_for_proportion = 0;
        for(int fl_id : active_flavor_ids_sorted_by_priority) { 
            total_candies_for_proportion += G_flavor_total_counts[fl_id];
        }
        if (total_candies_for_proportion == 0) total_candies_for_proportion = 1; 

        // Start with proportional widths, then spend leftover columns on the
        // largest populations.  Larger groups benefit disproportionately from
        // having more maneuvering space for a single connected component.
        for (int fl_id : active_flavor_ids_sorted_by_priority) {
            assigned_widths[fl_id] = static_cast<int>(std::floor(
                static_cast<double>(GRID_SIZE) *
                G_flavor_total_counts[fl_id] / total_candies_for_proportion
            ));
            total_assigned_width_sum += assigned_widths[fl_id];
        }

        // Give each leftover column to the flavor with the largest fractional
        // ideal width. Here ideal_width = count / 10, so count % 10 orders
        // those remainders exactly without floating-point comparisons.
        int remaining_width_to_assign = GRID_SIZE - total_assigned_width_sum;
        std::vector<int> remainder_priority = active_flavor_ids_sorted_by_priority;
        std::sort(remainder_priority.begin(), remainder_priority.end(),
            [](int a, int b) {
                int ra = G_flavor_total_counts[a] % GRID_SIZE;
                int rb = G_flavor_total_counts[b] % GRID_SIZE;
                if (ra != rb) return ra > rb;
                if (G_flavor_total_counts[a] != G_flavor_total_counts[b]) {
                    return G_flavor_total_counts[a] > G_flavor_total_counts[b];
                }
                return a < b;
            });
        for (int i = 0; i < remaining_width_to_assign; ++i) {
            assigned_widths[remainder_priority[i]]++;
        }
    }

    int current_col_start = 0;
    for (int fl_id_in_sorted_order : active_flavor_ids_sorted_by_priority) { 
        if (assigned_widths[fl_id_in_sorted_order] > 0) {
            G_target_col_ranges[fl_id_in_sorted_order] = {current_col_start, current_col_start + assigned_widths[fl_id_in_sorted_order] - 1};
            current_col_start += assigned_widths[fl_id_in_sorted_order];
        } else { 
            G_target_col_ranges[fl_id_in_sorted_order] = {current_col_start, current_col_start - 1};
        }
    }
    
    for (int fl = 1; fl <= NUM_FLAVORS; ++fl) {
        if (!G_flavor_active[fl]) { 
            G_target_col_ranges[fl] = {0, -1}; 
        }
    }
}


int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    initialize_global_data();

    GameState current_gs;
    for (int t_0_indexed = 0; t_0_indexed < NUM_TURNS; ++t_0_indexed) {
        current_gs.turn_num_1_indexed = t_0_indexed + 1; 
        
        int p_val_1_indexed; 
        std::cin >> p_val_1_indexed;
        
        std::pair<int, int> candy_loc = current_gs.find_pth_empty_cell(p_val_1_indexed);
        
        current_gs.place_candy(candy_loc.first, candy_loc.second, G_FLAVOR_SEQUENCE[t_0_indexed]); 
        
        char chosen_dir_char = decide_tilt_direction_logic(current_gs);
        
        std::cout << chosen_dir_char << std::endl; 
        
        int dir_idx_to_apply = 0; 
        for(int k=0; k<NUM_DIRECTIONS; ++k) {
            if(DIR_CHARS[k] == chosen_dir_char) {
                dir_idx_to_apply = k;
                break;
            }
        }
        current_gs.apply_tilt(dir_idx_to_apply);
    }

    return 0;
}
# EVOLVE-BLOCK-END
