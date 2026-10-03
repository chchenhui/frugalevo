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
#include <unordered_map>
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
int G_last_dir_idx = -1; // -1 indicates no previous tilt; used for axis continuity bias

/* Docstring: Enable lightweight expectimax lookahead (up to depth 2).
   We sample a few random placements of the next candy and, for each,
   evaluate the best next tilt. At depth 2 we also sample one more step.
   Depth 0 returns immediate board eval. */
// Lookahead parameters
const int MAX_LOOKAHEAD_DEPTH = 2;
// Sample count per depth (depth 1 then depth 2). Still very fast on 10x10.
static constexpr std::array<int, MAX_LOOKAHEAD_DEPTH> NUM_SAMPLES_CONFIG = {24, 12};


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

// Zobrist hashing and a small transposition table to memoize lookahead values.
// This greatly reduces duplicate computations from convergent tilt sequences.
std::array<std::array<std::array<uint64_t, NUM_FLAVORS + 1>, GRID_SIZE>, GRID_SIZE> G_zobrist;
static inline uint64_t compute_board_hash(const std::array<std::array<int, GRID_SIZE>, GRID_SIZE>& board) {
    uint64_t h = 0;
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            int v = board[r][c];
            if (v) h ^= G_zobrist[r][c][v];
        }
    }
    return h;
}
// Transposition table keyed by (board hash, turn, depth)
static std::unordered_map<uint64_t, double> G_TT;
static constexpr size_t G_TT_MAX = 800000;


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
    
    /* Docstring: Apply a tilt by compacting each row/column toward the target edge.
       Returns whether any candy actually moved. We intentionally do NOT rebuild
       candies_list here; later evaluations scan the board directly, and the count
       of candies (for empty-cell computation) remains correct since tilt doesn't
       change it. */
    bool apply_tilt(int dir_idx) { 
        bool changed = false;
        if (dir_idx == 0) { // F (Up)
            for (int c = 0; c < GRID_SIZE; ++c) {
                int current_write_r = 0;
                for (int r = 0; r < GRID_SIZE; ++r) {
                    int v = board[r][c];
                    if (v != 0) {
                        if (r != current_write_r) {
                            board[current_write_r][c] = v;
                            board[r][c] = 0;
                            changed = true;
                        }
                        current_write_r++;
                    }
                }
            }
        } else if (dir_idx == 1) { // B (Down)
            for (int c = 0; c < GRID_SIZE; ++c) {
                int current_write_r = GRID_SIZE - 1;
                for (int r = GRID_SIZE - 1; r >= 0; --r) {
                    int v = board[r][c];
                    if (v != 0) {
                        if (r != current_write_r) {
                            board[current_write_r][c] = v;
                            board[r][c] = 0;
                            changed = true;
                        }
                        current_write_r--;
                    }
                }
            }
        } else if (dir_idx == 2) { // L (Left)
            for (int r = 0; r < GRID_SIZE; ++r) {
                int current_write_c = 0;
                for (int c = 0; c < GRID_SIZE; ++c) {
                    int v = board[r][c];
                    if (v != 0) {
                        if (c != current_write_c) {
                            board[r][current_write_c] = v;
                            board[r][c] = 0;
                            changed = true;
                        }
                        current_write_c++;
                    }
                }
            }
        } else { // R (Right, dir_idx == 3)
            for (int r = 0; r < GRID_SIZE; ++r) {
                int current_write_c = GRID_SIZE - 1;
                for (int c = GRID_SIZE - 1; c >= 0; --c) {
                    int v = board[r][c];
                    if (v != 0) {
                        if (c != current_write_c) {
                            board[r][current_write_c] = v;
                            board[r][c] = 0;
                            changed = true;
                        }
                        current_write_c--;
                    }
                }
            }
        }
        return changed;
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
    
    /* Docstring: Compute per-flavor dispersion as sum of Manhattan distances
       from each candy to its flavor's center-of-mass. Scan the board directly
       so this remains correct even when candies_list positions are stale. */
    double calculate_distance_penalty_CoM() const {
        std::array<double, NUM_FLAVORS + 1> sum_r; sum_r.fill(0.0);
        std::array<double, NUM_FLAVORS + 1> sum_c; sum_c.fill(0.0);
        std::array<int, NUM_FLAVORS + 1> counts; counts.fill(0);

        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                int v = board[r][c];
                if (v == 0) continue;
                counts[v]++;
                sum_r[v] += r;
                sum_c[v] += c;
            }
        }

        std::array<std::pair<double, double>, NUM_FLAVORS + 1> com_coords;
        for (int fl = 1; fl <= NUM_FLAVORS; ++fl) {
            if (counts[fl] > 0) {
                com_coords[fl] = {sum_r[fl] / counts[fl], sum_c[fl] / counts[fl]};
            }
        }

        double total_manhattan_dist_penalty = 0.0;
        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                int v = board[r][c];
                if (v == 0) continue;
                if (counts[v] > 1) {
                    const auto& com = com_coords[v];
                    total_manhattan_dist_penalty += std::abs(static_cast<double>(r) - com.first)
                                                  + std::abs(static_cast<double>(c) - com.second);
                }
            }
        }
        return total_manhattan_dist_penalty;
    }

    /* Docstring: Penalty for candies that lie outside their assigned
       flavor column strip. Scan the board for robustness under simulated tilts. */
    double calculate_region_penalty() const {
        double penalty = 0.0;
        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                int v = board[r][c];
                if (v == 0) continue;
                if (!G_flavor_active[v]) continue;

                const auto& range = G_target_col_ranges[v];
                int min_target_c = range.first;
                int max_target_c = range.second;
                if (min_target_c > max_target_c) continue;

                if (c < min_target_c) penalty += (min_target_c - c);
                else if (c > max_target_c) penalty += (c - max_target_c);
            }
        }
        return penalty;
    }
    
    /* Docstring: Small bonus inside the correct strip for being on outer edges
       and corners; encourages compact blobs aligned to boundaries. */
    double calculate_edge_bonus() const {
        double bonus_val = 0.0;
        const double PER_CANDY_BONUS_FACTOR = 0.5; 

        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                int v = board[r][c];
                if (v == 0) continue;
                if (!G_flavor_active[v]) continue;

                const auto& range = G_target_col_ranges[v];
                int min_target_c = range.first;
                int max_target_c = range.second;
                if (min_target_c > max_target_c) continue;

                bool in_correct_strip = (c >= min_target_c && c <= max_target_c);
                if (!in_correct_strip) continue;

                if (r == 0 || r == GRID_SIZE - 1) {
                    bonus_val += PER_CANDY_BONUS_FACTOR;
                }
                if ((c == 0 && min_target_c == 0) ||
                    (c == GRID_SIZE - 1 && max_target_c == GRID_SIZE - 1)) {
                    bonus_val += PER_CANDY_BONUS_FACTOR;
                }
            }
        }
        return bonus_val;
    }
        
    /* Docstring: Heuristic evaluation combining:
       - sum of squares of same-flavor connected components (BFS),
       - per-flavor center-of-mass Manhattan dispersion penalty,
       - penalty for candies outside their assigned column strip,
       - small edge/corner bonus inside correct strip,
       - same-flavor adjacency pairs bonus (cohesion),
       - interface penalty for different-flavor touching pairs (perimeter control).
       Coefficients vary with turn to emphasize connectivity later while using
       local cues early; scales are conservative to avoid overfitting. */
    double evaluate() const {
        if (turn_num_1_indexed == 0) return 0.0;

        long long sum_sq_comp = calculate_sum_sq_comp_size();
        double dist_penalty_com = calculate_distance_penalty_CoM();
        double region_penalty_val = calculate_region_penalty();
        double edge_bonus_val = calculate_edge_bonus();

        // Count adjacent pairs (right/down only to avoid double-counting)
        int adjacency_pairs = 0;
        int mismatch_pairs = 0;
        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                int v = board[r][c];
                if (v == 0) continue;
                if (r + 1 < GRID_SIZE) {
                    int w = board[r + 1][c];
                    if (w == v) adjacency_pairs++;
                    else if (w != 0) mismatch_pairs++;
                }
                if (c + 1 < GRID_SIZE) {
                    int w = board[r][c + 1];
                    if (w == v) adjacency_pairs++;
                    else if (w != 0) mismatch_pairs++;
                }
            }
        }

        double current_turn_double = static_cast<double>(turn_num_1_indexed);

        double A_coeff_conn = 15.0 + 1.1 * current_turn_double;
        double B_coeff_com_base = std::max(0.0, 170.0 - 1.7 * current_turn_double);
        double C_coeff_region_penalty_direct = std::max(2.0, 27.0 - 0.17 * current_turn_double);
        double D_coeff_edge_bonus = 5.0 + 0.2 * current_turn_double;
        double E_coeff_adjacency = 180.0 + 3.0 * current_turn_double;
        // Penalize heterogeneous boundaries to reduce perimeter and mixing
        double F_coeff_mismatch = 120.0 + 2.0 * current_turn_double;

        return A_coeff_conn * sum_sq_comp
             - B_coeff_com_base * dist_penalty_com
             - C_coeff_region_penalty_direct * region_penalty_val
             + D_coeff_edge_bonus * edge_bonus_val
             + E_coeff_adjacency * adjacency_pairs
             - F_coeff_mismatch * mismatch_pairs;
    }
};



/* Docstring: Expectimax-style lookahead evaluator with memoization (TT).
   - Base cases: return immediate evaluation.
   - Otherwise, sample a few next placements uniformly (without replacement),
     maximize over the next tilt for each sample, and average.
   - Use a Zobrist-hash keyed transposition table on (board, turn, depth)
     to avoid recomputing converged states across branches. */
double eval_lookahead(const GameState& state_after_tilt, int turn_T_of_candy_just_processed, int depth_remaining) {
    if (depth_remaining == 0 || turn_T_of_candy_just_processed == NUM_TURNS) {
        return state_after_tilt.evaluate();
    }

    int num_empty = state_after_tilt.count_empty_cells();
    if (num_empty == 0) {
        return state_after_tilt.evaluate();
    }

    // Probe cache
    uint64_t h = compute_board_hash(state_after_tilt.board);
    // Mix in turn and depth to distinguish future stochastic contexts
    uint64_t key = h
                 ^ (uint64_t(turn_T_of_candy_just_processed) * 0x9E3779B97F4A7C15ULL)
                 ^ (uint64_t(depth_remaining) * 0xC2B2AE3D27D4EB4FULL);
    auto it = G_TT.find(key);
    if (it != G_TT.end()) return it->second;

    int next_candy_flavor = G_FLAVOR_SEQUENCE[turn_T_of_candy_just_processed];
    int sample_count_param_idx = MAX_LOOKAHEAD_DEPTH - depth_remaining;
    int sample_count_this_depth = NUM_SAMPLES_CONFIG[sample_count_param_idx];
    int actual_num_samples = std::min(sample_count_this_depth, num_empty);

    if (actual_num_samples == 0) {
        double base = state_after_tilt.evaluate();
        if (G_TT.size() > G_TT_MAX) G_TT.clear();
        G_TT.emplace(key, base);
        return base;
    }

    double sum_over_sampled_placements = 0.0;

    if (actual_num_samples == num_empty) {
        // Enumerate all empty cells deterministically (1..num_empty)
        for (int s = 0; s < actual_num_samples; ++s) {
            int p_val_1_indexed_sample = s + 1;

            GameState S_after_placement = state_after_tilt;
            std::pair<int, int> candy_loc = S_after_placement.find_pth_empty_cell(p_val_1_indexed_sample);
            S_after_placement.place_candy(candy_loc.first, candy_loc.second, next_candy_flavor);
            S_after_placement.turn_num_1_indexed = turn_T_of_candy_just_processed + 1;

            double max_eval_for_this_placement = std::numeric_limits<double>::lowest();
            for (int dir_idx_next_tilt = 0; dir_idx_next_tilt < NUM_DIRECTIONS; ++dir_idx_next_tilt) {
                GameState S_after_next_tilt = S_after_placement;
                (void)S_after_next_tilt.apply_tilt(dir_idx_next_tilt);
                double val = eval_lookahead(S_after_next_tilt, S_after_placement.turn_num_1_indexed, depth_remaining - 1);
                if (val > max_eval_for_this_placement) {
                    max_eval_for_this_placement = val;
                }
            }
            sum_over_sampled_placements += max_eval_for_this_placement;
        }
    } else {
        // Deterministic stratified sampling across the empty-cell index space
        for (int s = 0; s < actual_num_samples; ++s) {
            int x = static_cast<int>(((s + 0.5) * num_empty) / actual_num_samples) + 1;
            if (x < 1) x = 1;
            if (x > num_empty) x = num_empty;

            GameState S_after_placement = state_after_tilt;
            std::pair<int, int> candy_loc = S_after_placement.find_pth_empty_cell(x);
            S_after_placement.place_candy(candy_loc.first, candy_loc.second, next_candy_flavor);
            S_after_placement.turn_num_1_indexed = turn_T_of_candy_just_processed + 1;

            double max_eval_for_this_placement = std::numeric_limits<double>::lowest();
            for (int dir_idx_next_tilt = 0; dir_idx_next_tilt < NUM_DIRECTIONS; ++dir_idx_next_tilt) {
                GameState S_after_next_tilt = S_after_placement;
                (void)S_after_next_tilt.apply_tilt(dir_idx_next_tilt);
                double val = eval_lookahead(S_after_next_tilt, S_after_placement.turn_num_1_indexed, depth_remaining - 1);
                if (val > max_eval_for_this_placement) {
                    max_eval_for_this_placement = val;
                }
            }
            sum_over_sampled_placements += max_eval_for_this_placement;
        }
    }

    double result = sum_over_sampled_placements / actual_num_samples;
    if (G_TT.size() > G_TT_MAX) G_TT.clear();
    G_TT.emplace(key, result);
    return result;
}

/* Docstring: Choose the best tilt among the 4 directions using expectimax
   (up to depth 2): simulate each tilt, then look ahead by sampling the next
   placement(s) and maximizing over the next tilt(s).
   Tie-breaking bias (tiny, deterministic):
   - prefer continuing the same axis (F/B vs L/R)
   - prefer exactly the same direction as last tilt
   - in early turns, slightly prefer F/L to build a top-left corner.
   The bias is tiny so it only acts as tie-breaker. */
char decide_tilt_direction_logic(const GameState& current_gs_after_placement) {
    double best_score_with_bias = std::numeric_limits<double>::lowest();
    int best_dir_idx = 0;

    int turn_T_for_lookahead_base = current_gs_after_placement.turn_num_1_indexed;

    for (int i = 0; i < NUM_DIRECTIONS; ++i) {
        GameState gs_after_tilt_T = current_gs_after_placement;
        bool changed_by_tilt = gs_after_tilt_T.apply_tilt(i);

        double base_eval = eval_lookahead(gs_after_tilt_T, turn_T_for_lookahead_base, MAX_LOOKAHEAD_DEPTH);

        // Tiny deterministic bias for tie-breaking
        double bias = 0.0;
        if (G_last_dir_idx >= 0) {
            bool same_axis = ((i < 2) == (G_last_dir_idx < 2));
            if (same_axis) bias += 1e-9;
            if (i == G_last_dir_idx) bias += 2e-9;
        }
        if (turn_T_for_lookahead_base <= 35) {
            if (i == 0 || i == 2) bias += 5e-10; // early preference for F/L
        }
        // Prefer tilts that actually move candies (avoid wasting a move)
        if (!changed_by_tilt) bias -= 5e-10;

        double eval_with_bias = base_eval + bias;
        if (eval_with_bias > best_score_with_bias) {
            best_score_with_bias = eval_with_bias;
            best_dir_idx = i;
        }
    }
    return DIR_CHARS[best_dir_idx];
}









void initialize_global_data() {
    G_flavor_total_counts.fill(0);
    for (int t = 0; t < NUM_TURNS; ++t) {
        std::cin >> G_FLAVOR_SEQUENCE[t];
        G_flavor_total_counts[G_FLAVOR_SEQUENCE[t]]++;
    }

    // Deterministic RNG seed derived from the full flavor sequence.
    // This stabilizes decisions across runs on the same input.
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    for (int t = 0; t < NUM_TURNS; ++t) {
        seed ^= static_cast<uint64_t>(G_FLAVOR_SEQUENCE[t]) + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);
    }
    rng.x = seed | 1ULL;

    // Initialize zobrist table and reserve TT
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            for (int v = 1; v <= NUM_FLAVORS; ++v) {
                G_zobrist[r][c][v] = rng.next();
            }
        }
    }
    G_TT.clear();
    G_TT.reserve(1 << 20);

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
    for (const auto& p : sorter_flavor_count_id) {
        active_flavor_ids_sorted_by_priority.push_back(p.second);
    }
    
    std::vector<int> assigned_widths(NUM_FLAVORS + 1, 0);
    int total_assigned_width_sum = 0;

    if (!active_flavor_ids_sorted_by_priority.empty()) {
        double total_candies_for_proportion = 0;
        for (int fl_id : active_flavor_ids_sorted_by_priority) {
            total_candies_for_proportion += G_flavor_total_counts[fl_id];
        }
        if (total_candies_for_proportion == 0) total_candies_for_proportion = 1;

        for (int fl_id : active_flavor_ids_sorted_by_priority) {
            assigned_widths[fl_id] = static_cast<int>(std::floor(
                static_cast<double>(GRID_SIZE) * G_flavor_total_counts[fl_id] / total_candies_for_proportion
            ));
            total_assigned_width_sum += assigned_widths[fl_id];
        }

        int remaining_width_to_assign = GRID_SIZE - total_assigned_width_sum;
        for (int i = 0; i < remaining_width_to_assign; ++i) {
            assigned_widths[active_flavor_ids_sorted_by_priority[i % active_flavor_ids_sorted_by_priority.size()]]++;
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
        G_last_dir_idx = dir_idx_to_apply;
        (void)current_gs.apply_tilt(dir_idx_to_apply);
    }

    return 0;
}
# EVOLVE-BLOCK-END