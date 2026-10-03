# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <chrono>
#include <random>
#include <cmath>
// #include <queue> // Not strictly needed now for beam search pruning strategy
#include <utility> // For std::pair, std::move
#include <span>    // For std::span (C++20)

// Using 0-indexed internally for stacks and box values for convenience with vectors
// Box values 0 to N-1, stack indices 0 to M-1.
// Input is 1-indexed for box values (1 to N) and stack indices (1 to M).
// Output must be 1-indexed for box values and stack indices.
// target_stack_idx=0 for carry-out operation.

// Constants for heuristic evaluation
const double HEURISTIC_EMPTY_STACK_BONUS_SCORE = 1000.0;
const double STACK_HEIGHT_PENALTY_FACTOR = 0.1;
const int HEURISTIC_LOOKAHEAD_WINDOW = 5;
const double HEURISTIC_COVER_CRITICAL_PENALTY_PER_BOX_ABOVE = 3.0;
const double HEURISTIC_MIN_LABEL_IN_DEST_FACTOR = 0.05;

// SA Parameters
const double TIME_LIMIT_SECONDS_TOTAL = 1.95; 
double BEAM_SEARCH_TIME_LIMIT_SECONDS_PARAM = 0.30; 
const int BEAM_WIDTH = 5; 
double T_INITIAL_SA = 75.0;  
double T_FINAL_SA = 0.01;   

// --- Random Number Generation ---
struct RandomGenerator {
    std::mt19937 rng;
    RandomGenerator() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {}

    int an_int(int min_val, int max_val) {
        if (min_val > max_val) return min_val;
        std::uniform_int_distribution<int> dist(min_val, max_val);
        return dist(rng);
    }

    double a_double(double min_val, double max_val) {
        std::uniform_real_distribution<double> dist(min_val, max_val);
        return dist(rng);
    }
} RGen;

auto GLOBAL_START_TIME = std::chrono::steady_clock::now();

// --- State Definition ---
struct State {
    std::vector<std::vector<int>> stacks;
    std::vector<std::pair<int, int>> box_pos; // {stack_idx, height_idx}
    long long energy_cost;
    std::vector<std::pair<int, int>> ops_history;
    int N_val;
    int M_val;
    bool record_ops_flag;

    State() : energy_cost(0), N_val(0), M_val(0), record_ops_flag(true) {}

    State(int N_in, int M_in, const std::vector<std::vector<int>>& initial_stacks_input, bool rec_ops = true)
        : energy_cost(0), N_val(N_in), M_val(M_in), record_ops_flag(rec_ops) {
        stacks.resize(M_val);
        for (int i = 0; i < M_val; ++i) {
            stacks[i].reserve(N_val + 20); 
        }
        box_pos.resize(N_val);
        if (record_ops_flag) {
            ops_history.reserve(N_val * 2 + 50); 
        }

        for (int i = 0; i < M_val; ++i) {
            for (size_t j = 0; j < initial_stacks_input[i].size(); ++j) {
                int box_id = initial_stacks_input[i][j] - 1; // 0-indexed
                stacks[i].push_back(box_id);
                box_pos[box_id] = {i, (int)j};
            }
        }
    }

    State(const State& other) = default;
    State& operator=(const State& other) = default;
    State(State&& other) noexcept = default;
    State& operator=(State&& other) noexcept = default;

    double evaluate_destination_stack_choice(
        int current_target_box_val, // 0-indexed
        std::span<const int> block_to_move, // 0-indexed box values
        int dest_stack_idx) const { // 0-indexed
        const auto& dest_stack_content = stacks[dest_stack_idx];
        double current_score = 0;

        if (dest_stack_content.empty()) {
            current_score -= HEURISTIC_EMPTY_STACK_BONUS_SCORE;
        } else {
            current_score += (double)dest_stack_content.size() * STACK_HEIGHT_PENALTY_FACTOR;
            int min_label_in_dest_stack = N_val; 
            for (int box_val_in_dest : dest_stack_content) {
                min_label_in_dest_stack = std::min(min_label_in_dest_stack, box_val_in_dest);
            }
            current_score -= (double)min_label_in_dest_stack * HEURISTIC_MIN_LABEL_IN_DEST_FACTOR;
        }

        for (int box_in_dest : dest_stack_content) {
            if (box_in_dest > current_target_box_val && box_in_dest < current_target_box_val + HEURISTIC_LOOKAHEAD_WINDOW + 1) {
                current_score += HEURISTIC_COVER_CRITICAL_PENALTY_PER_BOX_ABOVE * block_to_move.size();
            }
        }

        for (size_t i = 0; i < block_to_move.size(); ++i) {
            int box_in_block = block_to_move[i];
            if (box_in_block > current_target_box_val && box_in_block < current_target_box_val + HEURISTIC_LOOKAHEAD_WINDOW + 1) {
                int boxes_on_top_in_block = block_to_move.size() - 1 - i;
                current_score += HEURISTIC_COVER_CRITICAL_PENALTY_PER_BOX_ABOVE * boxes_on_top_in_block;
            }
        }
        return current_score;
    }

    void apply_op1_move(int first_box_in_block_val, int num_moved_boxes, int dest_stack_idx) { // All 0-indexed
        int src_stack_idx = box_pos[first_box_in_block_val].first;
        int first_box_height_idx_in_src = box_pos[first_box_in_block_val].second;

        auto& src_stack_vec = stacks[src_stack_idx];
        auto& dest_stack_vec = stacks[dest_stack_idx];

        auto P_k_start_iter = src_stack_vec.begin() + first_box_height_idx_in_src;
        auto P_k_end_iter = src_stack_vec.begin() + first_box_height_idx_in_src + num_moved_boxes;

        int old_dest_stack_height = dest_stack_vec.size();
        dest_stack_vec.insert(dest_stack_vec.end(),
                              std::make_move_iterator(P_k_start_iter),
                              std::make_move_iterator(P_k_end_iter));

        for (int i = 0; i < num_moved_boxes; ++i) {
            int moved_box_val = dest_stack_vec[old_dest_stack_height + i];
            box_pos[moved_box_val] = {dest_stack_idx, old_dest_stack_height + i};
        }

        src_stack_vec.erase(P_k_start_iter, P_k_end_iter);

        energy_cost += (num_moved_boxes + 1);
        if (record_ops_flag) {
            ops_history.push_back({first_box_in_block_val + 1, dest_stack_idx + 1});
        }
    }

    void apply_op2_carry_out(int target_box_val) { // 0-indexed
        int stack_idx = box_pos[target_box_val].first;
        stacks[stack_idx].pop_back();
        if (record_ops_flag) {
            ops_history.push_back({target_box_val + 1, 0});
        }
    }
};

struct BeamNode {
    State current_board_state;
    std::vector<int> partial_plan_D; 

    BeamNode() = default;
    BeamNode(State state, std::vector<int> plan)
        : current_board_state(std::move(state)), partial_plan_D(std::move(plan)) {}
    
    bool operator<(const BeamNode& other) const { 
        return current_board_state.energy_cost < other.current_board_state.energy_cost;
    }
};

std::vector<int> generate_initial_plan_beam_search(
    const std::vector<std::vector<int>>& initial_stacks_param,
    int N_CONST, int M_CONST, int beam_width_param, double max_duration_for_beam_search) {
    
    std::vector<BeamNode> beam;
    beam.reserve(beam_width_param); 
    
    State initial_state_for_beam(N_CONST, M_CONST, initial_stacks_param, false); 
    beam.emplace_back(std::move(initial_state_for_beam), std::vector<int>());
    if (N_CONST > 0) beam.back().partial_plan_D.reserve(N_CONST);
    
    std::vector<BeamNode> candidates;
    if (M_CONST > 0) candidates.reserve(beam_width_param * M_CONST + 5);
    else candidates.reserve(beam_width_param + 5);


    for (int k_target_box = 0; k_target_box < N_CONST; ++k_target_box) {
        double elapsed_seconds_so_far = std::chrono::duration<double>(std::chrono::steady_clock::now() - GLOBAL_START_TIME).count();
        bool time_is_up = elapsed_seconds_so_far > max_duration_for_beam_search;
        
        if (time_is_up) {
            if (beam.empty()) { 
                 std::vector<int> emergency_plan(N_CONST); 
                 if (N_CONST == 0) return emergency_plan;
                 for (int i = 0; i < N_CONST; ++i) emergency_plan[i] = RGen.an_int(0, M_CONST - 1);
                 return emergency_plan;
            }
            std::sort(beam.begin(), beam.end()); 
            BeamNode& best_node_so_far = beam[0]; 
                
            for (int k_future = k_target_box; k_future < N_CONST; ++k_future) {
                State& S_greedy = best_node_so_far.current_board_state;
                int f_target_val = k_future; 
                int f_src_idx = S_greedy.box_pos[f_target_val].first;
                int f_h_idx = S_greedy.box_pos[f_target_val].second;
                int f_num_top = S_greedy.stacks[f_src_idx].size() - 1 - f_h_idx;

                if (f_num_top == 0) { 
                    best_node_so_far.partial_plan_D.push_back(f_src_idx); 
                } else { 
                    int f_block_first_val = S_greedy.stacks[f_src_idx][f_h_idx + 1];
                    
                    std::span<const int> block_span_greedy;
                    if (f_num_top > 0) {
                        block_span_greedy = std::span<const int>(S_greedy.stacks[f_src_idx].data() + f_h_idx + 1, f_num_top);
                    }
                    
                    double min_h_eval_score = std::numeric_limits<double>::max();
                    int best_d_greedy = (M_CONST > 1) ? (f_src_idx + 1) % M_CONST : 0; 
                    
                    for (int d_cand = 0; d_cand < M_CONST; ++d_cand) {
                        if (d_cand == f_src_idx) continue; 
                        double h_eval_score = S_greedy.evaluate_destination_stack_choice(k_future, block_span_greedy, d_cand);
                        if (h_eval_score < min_h_eval_score) { 
                            min_h_eval_score = h_eval_score; 
                            best_d_greedy = d_cand; 
                        }
                    }
                    best_node_so_far.partial_plan_D.push_back(best_d_greedy);
                    S_greedy.apply_op1_move(f_block_first_val, f_num_top, best_d_greedy);
                }
                S_greedy.apply_op2_carry_out(f_target_val);
            }
            return best_node_so_far.partial_plan_D;
        }

        candidates.clear();
        for (auto& current_beam_node : beam) {
            State& S_curr = current_beam_node.current_board_state; 
            int target_val = k_target_box; 
            int src_idx = S_curr.box_pos[target_val].first;
            int h_idx = S_curr.box_pos[target_val].second;
            int num_top = S_curr.stacks[src_idx].size() - 1 - h_idx;

            if (num_top == 0) { 
                State next_S = S_curr; 
                std::vector<int> next_plan = current_beam_node.partial_plan_D;
                next_plan.push_back(src_idx); 
                next_S.apply_op2_carry_out(target_val);
                candidates.emplace_back(std::move(next_S), std::move(next_plan));
            } else { 
                int block_first_val = S_curr.stacks[src_idx][h_idx + 1];
                
                std::span<const int> block_span_bs;
                if (num_top > 0) {
                   block_span_bs = std::span<const int>(S_curr.stacks[src_idx].data() + h_idx + 1, num_top);
                }

                for (int dest_cand = 0; dest_cand < M_CONST; ++dest_cand) {
                    if (dest_cand == src_idx) continue;
                    State next_S = S_curr; 
                    std::vector<int> next_plan = current_beam_node.partial_plan_D;
                    next_plan.push_back(dest_cand);
                    next_S.apply_op1_move(block_first_val, num_top, dest_cand);
                    next_S.apply_op2_carry_out(target_val);
                    candidates.emplace_back(std::move(next_S), std::move(next_plan));
                }
            }
        }
        
        if (candidates.empty()) { 
            std::vector<int> emergency_plan(N_CONST);
            if (N_CONST == 0) return emergency_plan;
            if (beam.empty() && N_CONST > 0) { 
                 for (int i = 0; i < N_CONST; ++i) emergency_plan[i] = RGen.an_int(0, M_CONST - 1);
                 return emergency_plan;
            }
            // If candidates is empty but beam was not (e.g. M=1 case for Op1 where no valid dest_cand)
            // For M=10, this shouldn't happen unless beam_width is too small or other issues.
            // Fallback to random plan completion from best current beam node.
            // This is tricky, for now, if candidates is empty, signal failure for this path.
            // The outer logic will pick best from beam if one exists or ultimately generate full random.
            // If `beam` was non-empty but all paths led to no candidates, return best plan so far or random.
            // The current fallback is just random full plan.
            for (int i = 0; i < N_CONST; ++i) emergency_plan[i] = RGen.an_int(0, M_CONST - 1);
            return emergency_plan;
        }

        std::sort(candidates.begin(), candidates.end());

        beam.clear();
        for (size_t i = 0; i < std::min((size_t)beam_width_param, candidates.size()); ++i) {
            beam.push_back(std::move(candidates[i]));
        }
        
        if (beam.empty() && N_CONST > 0) { 
            std::vector<int> emergency_plan(N_CONST);
            for (int i = 0; i < N_CONST; ++i) emergency_plan[i] = RGen.an_int(0, M_CONST - 1);
            return emergency_plan;
        }
    }

    if (beam.empty()){ 
        std::vector<int> emergency_plan(N_CONST);
        if (N_CONST == 0) return emergency_plan;
        for (int i = 0; i < N_CONST; ++i) emergency_plan[i] = RGen.an_int(0, M_CONST - 1);
        return emergency_plan;
    }
    std::sort(beam.begin(), beam.end()); 
    return beam[0].partial_plan_D;
}

struct SimulationResult {
    long long energy_cost;
    std::vector<std::pair<int, int>> ops_history;
};

std::pair<State, long long> simulate_up_to_k(
    const std::vector<std::vector<int>>& initial_stacks_param,
    const std::vector<int>& plan_D,
    int N_CONST, int M_CONST,
    int k_limit_box_idx) { 
    
    State current_sim_state(N_CONST, M_CONST, initial_stacks_param, false); 
    
    for (int k_target_box = 0; k_target_box < k_limit_box_idx; ++k_target_box) {
        int target_box_val = k_target_box; 
        int src_stack_idx = current_sim_state.box_pos[target_box_val].first;
        int height_idx = current_sim_state.box_pos[target_box_val].second;
        int num_boxes_on_top = current_sim_state.stacks[src_stack_idx].size() - 1 - height_idx;

        if (num_boxes_on_top > 0) { 
            int op1_first_box_in_block_val = current_sim_state.stacks[src_stack_idx][height_idx + 1];
            int actual_dest_stack_idx = plan_D[k_target_box];
            
            if (actual_dest_stack_idx == src_stack_idx && M_CONST > 1) {
                actual_dest_stack_idx = (src_stack_idx + 1) % M_CONST;
            }
            
            if (M_CONST > 1) {
                 current_sim_state.apply_op1_move(op1_first_box_in_block_val, num_boxes_on_top, actual_dest_stack_idx);
            }
        }
        current_sim_state.apply_op2_carry_out(target_box_val);
    }
    long long final_energy = current_sim_state.energy_cost; // Store before move
    return {std::move(current_sim_state), final_energy};
}

SimulationResult run_simulation_from_intermediate_state(
    State intermediate_state, 
    const std::vector<int>& plan_D,
    int k_start_box_idx,      
    int N_CONST, int M_CONST,
    bool record_ops_for_suffix) {

    State current_sim_state = std::move(intermediate_state); 
    
    if (record_ops_for_suffix) {
        current_sim_state.ops_history.clear(); 
        current_sim_state.record_ops_flag = true;
    } else {
        current_sim_state.record_ops_flag = false;
    }

    for (int k_target_box = k_start_box_idx; k_target_box < N_CONST; ++k_target_box) {
        int target_box_val = k_target_box; 
        int src_stack_idx = current_sim_state.box_pos[target_box_val].first;
        int height_idx = current_sim_state.box_pos[target_box_val].second;
        int num_boxes_on_top = current_sim_state.stacks[src_stack_idx].size() - 1 - height_idx;

        if (num_boxes_on_top > 0) {
            int op1_first_box_in_block_val = current_sim_state.stacks[src_stack_idx][height_idx + 1];
            int actual_dest_stack_idx = plan_D[k_target_box];
            
            if (actual_dest_stack_idx == src_stack_idx && M_CONST > 1) {
                actual_dest_stack_idx = (src_stack_idx + 1) % M_CONST;
            }

            if (M_CONST > 1) {
                 current_sim_state.apply_op1_move(op1_first_box_in_block_val, num_boxes_on_top, actual_dest_stack_idx);
            }
        }
        current_sim_state.apply_op2_carry_out(target_box_val);
    }
    return {current_sim_state.energy_cost, std::move(current_sim_state.ops_history)};
}

SimulationResult run_simulation(const std::vector<std::vector<int>>& initial_stacks_param,
                                const std::vector<int>& plan_D, int N_CONST, int M_CONST, bool record_all_ops = true) {
    State initial_state_for_full_sim(N_CONST, M_CONST, initial_stacks_param, record_all_ops);
    return run_simulation_from_intermediate_state(std::move(initial_state_for_full_sim), plan_D, 0, N_CONST, M_CONST, record_all_ops);
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    GLOBAL_START_TIME = std::chrono::steady_clock::now();

    int N_CONST, M_CONST;
    std::cin >> N_CONST >> M_CONST; 
    std::vector<std::vector<int>> initial_stacks_main(M_CONST, std::vector<int>(N_CONST / M_CONST));
    for (int i = 0; i < M_CONST; ++i) {
        for (int j = 0; j < N_CONST / M_CONST; ++j) {
            std::cin >> initial_stacks_main[i][j];
        }
    }
    
    double beam_search_duration_budget = TIME_LIMIT_SECONDS_TOTAL * BEAM_SEARCH_TIME_LIMIT_SECONDS_PARAM;
    std::vector<int> current_plan_D = generate_initial_plan_beam_search(
        initial_stacks_main, 
        N_CONST, 
        M_CONST, 
        BEAM_WIDTH, 
        beam_search_duration_budget
    );
    
    if (current_plan_D.size() < (size_t)N_CONST && N_CONST > 0) {
        current_plan_D.resize(N_CONST, 0); 
    }
     if (N_CONST == 0) { 
        current_plan_D.clear(); 
    }

    SimulationResult current_sim_res_eval = run_simulation(initial_stacks_main, current_plan_D, N_CONST, M_CONST, false); 
    long long current_energy = current_sim_res_eval.energy_cost;
    
    std::vector<int> best_plan_D = current_plan_D;
    long long best_energy = current_energy;

    const int MAX_BLOCK_CHANGE_LEN = (N_CONST == 0) ? 0 : std::max(1, N_CONST / 15);
    double time_for_sa_start_offset = std::chrono::duration<double>(std::chrono::steady_clock::now() - GLOBAL_START_TIME).count();
    
    while (true) {
        double elapsed_seconds_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - GLOBAL_START_TIME).count();
        if (elapsed_seconds_total >= TIME_LIMIT_SECONDS_TOTAL - 0.02) break; 

        double elapsed_seconds_sa_phase = elapsed_seconds_total - time_for_sa_start_offset;
        double total_time_for_sa_phase = (TIME_LIMIT_SECONDS_TOTAL - 0.02) - time_for_sa_start_offset; 
        if (total_time_for_sa_phase <= 0.001) break; 

        double progress_ratio = std::min(1.0, std::max(0.0, elapsed_seconds_sa_phase / total_time_for_sa_phase));
        double current_temp = T_INITIAL_SA * std::pow(T_FINAL_SA / T_INITIAL_SA, progress_ratio);
        current_temp = std::max(current_temp, T_FINAL_SA);

        std::vector<int> new_plan_D = current_plan_D;
        long long new_energy;
        
        int k_change_start_idx = N_CONST; 

        State state_at_change_point; 
        bool can_use_partial_simulation = false; 

        double op_choice_rand = RGen.a_double(0.0, 1.0);

        if (op_choice_rand < 0.35 && N_CONST > 0) { 
            int idx_to_change = RGen.an_int(0, N_CONST - 1);
            k_change_start_idx = idx_to_change;
            new_plan_D[idx_to_change] = RGen.an_int(0, M_CONST - 1); // M_CONST is 10, so M_CONST-1 is 9
            
            // For partial sim, state needs to be based on prefix of new_plan_D
            auto sim_pair = simulate_up_to_k(initial_stacks_main, new_plan_D, N_CONST, M_CONST, k_change_start_idx);
            state_at_change_point = std::move(sim_pair.first);
            can_use_partial_simulation = true;

        } else if (op_choice_rand < 0.80 && N_CONST > 0) { 
            int block_op_start_idx_rand = RGen.an_int(0, N_CONST - 1); 
            int len = RGen.an_int(1, MAX_BLOCK_CHANGE_LEN);
            
            k_change_start_idx = N_CONST; 
            for(int i=0; i<len; ++i) {
                int current_k_in_plan_rand = (block_op_start_idx_rand + i) % N_CONST;
                k_change_start_idx = std::min(k_change_start_idx, current_k_in_plan_rand);
                new_plan_D[current_k_in_plan_rand] = RGen.an_int(0, M_CONST - 1);
            }
            // For partial sim, state needs to be based on prefix of new_plan_D
            auto sim_pair = simulate_up_to_k(initial_stacks_main, new_plan_D, N_CONST, M_CONST, k_change_start_idx);
            state_at_change_point = std::move(sim_pair.first);
            can_use_partial_simulation = true;

        } else if (N_CONST > 0) { 
            int k_to_recompute = RGen.an_int(0, N_CONST - 1);
            k_change_start_idx = k_to_recompute;

            // For greedy, decisions are based on current_plan_D's prefix
            // So, simulate current_plan_D up to k_change_start_idx
            auto sim_pair_greedy = simulate_up_to_k(initial_stacks_main, current_plan_D, N_CONST, M_CONST, k_change_start_idx);
            State decision_state = std::move(sim_pair_greedy.first); 

            int target_op_val = k_to_recompute; 
            int src_op_idx = decision_state.box_pos[target_op_val].first;
            int height_op_idx = decision_state.box_pos[target_op_val].second;
            int num_top_op = decision_state.stacks[src_op_idx].size() - 1 - height_op_idx;

            if (num_top_op > 0 && M_CONST > 1) { 
                std::span<const int> block_span_sa;
                if (num_top_op > 0) { 
                     block_span_sa = std::span<const int>(decision_state.stacks[src_op_idx].data() + height_op_idx + 1, num_top_op);
                }

                double min_h_score = std::numeric_limits<double>::max();
                int best_dest_idx = (M_CONST > 1) ? (src_op_idx + 1) % M_CONST : 0;
                
                for (int dest_cand = 0; dest_cand < M_CONST; ++dest_cand) {
                    if (dest_cand == src_op_idx) continue;
                    double h_score = decision_state.evaluate_destination_stack_choice(target_op_val, block_span_sa, dest_cand);
                    if (h_score < min_h_score) {
                        min_h_score = h_score;
                        best_dest_idx = dest_cand;
                    }
                }
                new_plan_D[k_to_recompute] = best_dest_idx;
            } else { 
                new_plan_D[k_to_recompute] = src_op_idx; 
            }
            // The state for suffix evaluation should be decision_state,
            // as new_plan_D only differs at or after k_change_start_idx.
            // The prefix energy is decision_state.energy_cost.
            state_at_change_point = std::move(decision_state);
            can_use_partial_simulation = true;
        } else { 
             k_change_start_idx = 0; 
             can_use_partial_simulation = false;
        }

        if (N_CONST == 0) { 
            new_energy = 0;
        } else if (!can_use_partial_simulation || k_change_start_idx == 0) { 
            // If k_change_start_idx is 0, state_at_change_point is initial state with 0 energy.
            // Full simulation is equivalent and perhaps cleaner.
             new_energy = run_simulation(initial_stacks_main, new_plan_D, N_CONST, M_CONST, false).energy_cost;
        } else {
             SimulationResult suffix_res = run_simulation_from_intermediate_state(std::move(state_at_change_point), new_plan_D, k_change_start_idx, N_CONST, M_CONST, false);
             new_energy = suffix_res.energy_cost; 
        }
        
        if (new_energy < current_energy) {
            current_energy = new_energy;
            current_plan_D = new_plan_D; 
            if (new_energy < best_energy) {
                best_energy = new_energy;
                best_plan_D = new_plan_D;
            }
        } else { 
            double delta_energy = new_energy - current_energy;
            if (current_temp > 1e-9 && RGen.a_double(0.0, 1.0) < std::exp(-delta_energy / current_temp)) { 
                current_energy = new_energy;
                current_plan_D = new_plan_D;
            }
        }
    }

    SimulationResult final_sim_result = run_simulation(initial_stacks_main, best_plan_D, N_CONST, M_CONST, true); 
    for (const auto& op : final_sim_result.ops_history) {
        std::cout << op.first << " " << op.second << "\n";
    }

    return 0;
}
# EVOLVE-BLOCK-END