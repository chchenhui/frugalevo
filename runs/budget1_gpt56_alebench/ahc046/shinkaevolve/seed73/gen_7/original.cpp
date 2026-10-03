# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include <tuple>
#include <array>
#include <chrono> 
#include <random> 
#include <cmath> // For std::exp, std::pow
#include <numeric> // For std::iota

// Constants
const int N_GRID = 20;
const int M_TARGETS_INPUT = 40; 
const int NUM_SEGMENTS = M_TARGETS_INPUT - 1; 
const int INF_COST = 1e9;
const int MAX_TOTAL_TURNS = 2 * N_GRID * M_TARGETS_INPUT; // 2*20*40 = 1600

// Randomness
unsigned int RND_SEED = std::chrono::steady_clock::now().time_since_epoch().count();
std::mt19937 rng(RND_SEED);

// Coordinates
struct Pos {
    int r, c;
    bool operator==(const Pos& other) const { return r == other.r && c == other.c; }
    bool operator!=(const Pos& other) const { return !(*this == other); }
    bool operator<(const Pos& other) const { 
        if (r != other.r) return r < other.r;
        return c < other.c;
    }
};
const Pos INVALID_POS = {-1, -1};

// Grid state
using Grid = std::array<std::array<bool, N_GRID>, N_GRID>; // true if block exists

bool is_valid_pos(Pos p) {
    return p.r >= 0 && p.r < N_GRID && p.c >= 0 && p.c < N_GRID;
}

bool is_blocked_pos(Pos p, const Grid& grid) {
    if (!is_valid_pos(p)) return true; 
    return grid[p.r][p.c];
}

void toggle_block_pos(Pos p, Grid& grid) { 
    if (is_valid_pos(p)) {
        grid[p.r][p.c] = !grid[p.r][p.c];
    }
}

// Directions
const int DR[] = {-1, 1, 0, 0}; // U, D, L, R
const int DC[] = {0, 0, -1, 1};
const char DIR_CHARS[] = {'U', 'D', 'L', 'R'};
const int DIR_REV_IDX[] = {1, 0, 3, 2}; // U(0)<->D(1), L(2)<->R(3)

// Global BFS structures for optimization
unsigned int g_bfs_generation_id = 0;
std::array<std::array<unsigned int, N_GRID>, N_GRID> g_bfs_cell_last_visited_generation; 
std::array<std::array<int, N_GRID>, N_GRID> g_bfs_cell_dist;


std::string reconstruct_path_from_came_from(Pos start_pos, Pos dest_pos, 
    const std::array<std::array<std::pair<Pos, std::pair<char, char>>, N_GRID>, N_GRID>& came_from_data) {
    std::string path_actions_str_reversed = ""; Pos p_trace = dest_pos;
    while(p_trace != start_pos && is_valid_pos(p_trace)) { 
        auto const& action_info = came_from_data[p_trace.r][p_trace.c]; 
        path_actions_str_reversed += action_info.second.second; 
        path_actions_str_reversed += action_info.second.first;  
        p_trace = action_info.first; 
    }
    std::reverse(path_actions_str_reversed.begin(), path_actions_str_reversed.end());
    return path_actions_str_reversed;
}

struct BFSResult { int cost; std::string actions_str; };

BFSResult bfs(Pos start_pos, Pos dest_pos, const Grid& grid, Pos intermediate_target_to_avoid, bool avoid_intermediate_target, bool build_action_str) {
    g_bfs_generation_id++; 
    
    std::array<std::array<std::pair<Pos, std::pair<char, char>>, N_GRID>, N_GRID> came_from_local; 
    std::queue<Pos> q;

    if (!is_valid_pos(start_pos)) return {INF_COST, ""};
    if (avoid_intermediate_target && start_pos == intermediate_target_to_avoid && start_pos != dest_pos) {
        return {INF_COST, ""};
    }

    g_bfs_cell_last_visited_generation[start_pos.r][start_pos.c] = g_bfs_generation_id;
    g_bfs_cell_dist[start_pos.r][start_pos.c] = 0;
    q.push(start_pos);

    int min_dist_to_dest = INF_COST;
    if (start_pos == dest_pos) min_dist_to_dest = 0;

    while(!q.empty()){
        Pos curr = q.front();
        q.pop();
        
        int d = g_bfs_cell_dist[curr.r][curr.c];
        
        if (curr == dest_pos) { 
             min_dist_to_dest = std::min(min_dist_to_dest, d); 
        }
        
        if (min_dist_to_dest != INF_COST && d >= min_dist_to_dest && curr != dest_pos) continue; 
        if (d + 1 > N_GRID * N_GRID) continue; 
        
        for (int i = 0; i < 4; ++i) { // Moves
            Pos next_p = {curr.r + DR[i], curr.c + DC[i]};
            if (is_blocked_pos(next_p, grid)) continue;
            if (avoid_intermediate_target && next_p == intermediate_target_to_avoid && next_p != dest_pos) continue;
            
            bool visited_in_current_bfs = (g_bfs_cell_last_visited_generation[next_p.r][next_p.c] == g_bfs_generation_id);
            if (!visited_in_current_bfs || g_bfs_cell_dist[next_p.r][next_p.c] > d + 1) {
                g_bfs_cell_last_visited_generation[next_p.r][next_p.c] = g_bfs_generation_id;
                g_bfs_cell_dist[next_p.r][next_p.c] = d + 1;
                if (build_action_str) came_from_local[next_p.r][next_p.c] = {curr, {'M', DIR_CHARS[i]}};
                q.push(next_p);
            }
        }

        for (int i = 0; i < 4; ++i) { // Slides
            Pos current_slide_p = curr; Pos landed_at_p = curr; 
            while (true) {
                Pos next_tile_in_slide = {current_slide_p.r + DR[i], current_slide_p.c + DC[i]};
                if (is_blocked_pos(next_tile_in_slide, grid)) { landed_at_p = current_slide_p; break; }
                if (avoid_intermediate_target && next_tile_in_slide == intermediate_target_to_avoid && next_tile_in_slide != dest_pos) {
                     landed_at_p = curr; 
                     break; 
                }
                current_slide_p = next_tile_in_slide;
            }

            if (landed_at_p == curr) continue; 
            Pos next_p = landed_at_p;

            bool visited_in_current_bfs = (g_bfs_cell_last_visited_generation[next_p.r][next_p.c] == g_bfs_generation_id);
            if (!visited_in_current_bfs || g_bfs_cell_dist[next_p.r][next_p.c] > d + 1) {
                g_bfs_cell_last_visited_generation[next_p.r][next_p.c] = g_bfs_generation_id;
                g_bfs_cell_dist[next_p.r][next_p.c] = d + 1;
                if (build_action_str) came_from_local[next_p.r][next_p.c] = {curr, {'S', DIR_CHARS[i]}};
                q.push(next_p);
            }
        }
    }
    
    BFSResult res = {INF_COST, ""};
    if (is_valid_pos(dest_pos) && g_bfs_cell_last_visited_generation[dest_pos.r][dest_pos.c] == g_bfs_generation_id) {
        res.cost = g_bfs_cell_dist[dest_pos.r][dest_pos.c];
        if (build_action_str && res.cost != INF_COST) { 
            res.actions_str = reconstruct_path_from_came_from(start_pos, dest_pos, came_from_local);
        }
    }
    return res;
}

void bfs_all(Pos start_pos, const Grid& grid, 
             Pos intermediate_target_to_avoid, bool strictly_avoid_intermediate,
             std::array<std::array<int, N_GRID>, N_GRID>& dist_out,
             std::array<std::array<std::pair<Pos, std::pair<char, char>>, N_GRID>, N_GRID>& came_from_out,
             bool store_came_from) {
    
    g_bfs_generation_id++;
    std::queue<Pos> q;

    for (int r_idx=0; r_idx<N_GRID; ++r_idx) std::fill(dist_out[r_idx].begin(), dist_out[r_idx].end(), INF_COST);

    if (!is_valid_pos(start_pos)) return;
    if (strictly_avoid_intermediate && start_pos == intermediate_target_to_avoid) {
        return;
    }

    g_bfs_cell_last_visited_generation[start_pos.r][start_pos.c] = g_bfs_generation_id;
    g_bfs_cell_dist[start_pos.r][start_pos.c] = 0;
    q.push(start_pos);

    while(!q.empty()){
        Pos curr = q.front();
        q.pop();
        int d = g_bfs_cell_dist[curr.r][curr.c];
        
        if (d + 1 > N_GRID * N_GRID) continue; 
        
        for (int i = 0; i < 4; ++i) { // Moves
            Pos next_p = {curr.r + DR[i], curr.c + DC[i]};
            if (is_blocked_pos(next_p, grid)) continue;
            if (strictly_avoid_intermediate && next_p == intermediate_target_to_avoid) continue; 
            
            bool visited_in_current_bfs = (g_bfs_cell_last_visited_generation[next_p.r][next_p.c] == g_bfs_generation_id);
            if (!visited_in_current_bfs || g_bfs_cell_dist[next_p.r][next_p.c] > d + 1) {
                g_bfs_cell_last_visited_generation[next_p.r][next_p.c] = g_bfs_generation_id;
                g_bfs_cell_dist[next_p.r][next_p.c] = d + 1;
                if (store_came_from) came_from_out[next_p.r][next_p.c] = {curr, {'M', DIR_CHARS[i]}};
                q.push(next_p);
            }
        }

        for (int i = 0; i < 4; ++i) { // Slides
            Pos current_slide_p = curr; 
            Pos landed_at_p = curr; 
            while (true) {
                Pos next_tile_in_slide = {current_slide_p.r + DR[i], current_slide_p.c + DC[i]};
                if (is_blocked_pos(next_tile_in_slide, grid)) { 
                    landed_at_p = current_slide_p; 
                    break; 
                }
                if (strictly_avoid_intermediate && next_tile_in_slide == intermediate_target_to_avoid) {
                     landed_at_p = curr; 
                     break; 
                }
                current_slide_p = next_tile_in_slide;
            }

            if (landed_at_p == curr) continue; 
            Pos next_p = landed_at_p;

            bool visited_in_current_bfs = (g_bfs_cell_last_visited_generation[next_p.r][next_p.c] == g_bfs_generation_id);
            if (!visited_in_current_bfs || g_bfs_cell_dist[next_p.r][next_p.c] > d + 1) {
                g_bfs_cell_last_visited_generation[next_p.r][next_p.c] = g_bfs_generation_id;
                g_bfs_cell_dist[next_p.r][next_p.c] = d + 1;
                if (store_came_from) came_from_out[next_p.r][next_p.c] = {curr, {'S', DIR_CHARS[i]}};
                q.push(next_p);
            }
        }
    }
    for (int r_idx = 0; r_idx < N_GRID; ++r_idx) {
        for (int c_idx = 0; c_idx < N_GRID; ++c_idx) {
            if (g_bfs_cell_last_visited_generation[r_idx][c_idx] == g_bfs_generation_id) {
                dist_out[r_idx][c_idx] = g_bfs_cell_dist[r_idx][c_idx];
            }
        }
    }
}

Pos G_initial_pos;
std::vector<Pos> G_targets_vec; 

struct SegmentExecResult { int turns = INF_COST; std::string actions_str; };

bool apply_direct_path_strat(Pos cur_P, Pos target_P, const Grid& g, SegmentExecResult& res, bool build_action_str) { 
    if (is_blocked_pos(target_P, g)) return false; 
    BFSResult bfs_res = bfs(cur_P, target_P, g, INVALID_POS, false, build_action_str); 
    if (bfs_res.cost == INF_COST) return false;
    res.turns = bfs_res.cost; 
    if(build_action_str) res.actions_str = bfs_res.actions_str; else res.actions_str.clear();
    return true;
}

bool apply_unblock_and_go_strat(Pos cur_P, Pos target_P, Grid& g , SegmentExecResult& res, bool build_action_str) { 
    if (!is_blocked_pos(target_P, g)) return false; 

    std::array<std::array<int, N_GRID>, N_GRID> dist_from_cur_P;
    std::array<std::array<std::pair<Pos, std::pair<char, char>>, N_GRID>, N_GRID> came_from_data; 
    
    bfs_all(cur_P, g, target_P, true, dist_from_cur_P, came_from_data, build_action_str);

    Pos best_adj_P = INVALID_POS; 
    int cost_to_best_adj_P = INF_COST;
    char alter_dir_char_to_unblock = ' '; 

    for (int i=0; i<4; ++i) { 
        Pos adj_P = {target_P.r + DR[i], target_P.c + DC[i]}; 
        if (!is_valid_pos(adj_P) || is_blocked_pos(adj_P, g)) continue; 
        
        if (dist_from_cur_P[adj_P.r][adj_P.c] < cost_to_best_adj_P) {
            cost_to_best_adj_P = dist_from_cur_P[adj_P.r][adj_P.c];
            best_adj_P = adj_P;
            alter_dir_char_to_unblock = DIR_CHARS[DIR_REV_IDX[i]]; 
        }
    }

    if (best_adj_P == INVALID_POS || cost_to_best_adj_P == INF_COST) return false; 

    res.turns = cost_to_best_adj_P + 1 + 1; 
    if (build_action_str) {
        res.actions_str = reconstruct_path_from_came_from(cur_P, best_adj_P, came_from_data);
        res.actions_str += 'A'; res.actions_str += alter_dir_char_to_unblock;
        res.actions_str += 'M'; res.actions_str += alter_dir_char_to_unblock; 
    } else {
        res.actions_str.clear();
    }
    toggle_block_pos(target_P, g); 
    return true;
}

bool apply_slide_strat(Pos cur_P, Pos target_P, Grid& g , SegmentExecResult& res, int slide_dir_idx, int type, bool build_action_str) {
    if (is_blocked_pos(target_P, g)) return false; 

    int slide_dr = DR[slide_dir_idx], slide_dc = DC[slide_dir_idx]; 
    char slide_dir_char = DIR_CHARS[slide_dir_idx];
    
    Pos slide_start_P = {target_P.r - slide_dr, target_P.c - slide_dc}; 
    Pos block_at_P = {target_P.r + slide_dr, target_P.c + slide_dc};    

    if (!is_valid_pos(slide_start_P)) return false; 
    if (slide_start_P == target_P) return false;

    if (type == 0) { 
        bool wall_exists_for_slide = !is_valid_pos(block_at_P) || is_blocked_pos(block_at_P, g);
        if (!wall_exists_for_slide) return false;

        BFSResult path_to_slide_start_P = bfs(cur_P, slide_start_P, g, 
                                              target_P, true, build_action_str);
        if (path_to_slide_start_P.cost == INF_COST) return false;

        res.turns = path_to_slide_start_P.cost + 1; 
        if (build_action_str) { 
            res.actions_str = path_to_slide_start_P.actions_str; 
            res.actions_str += 'S'; res.actions_str += slide_dir_char; 
        } else {
            res.actions_str.clear();
        }
        return true;

    } else if (type == 1) { 
        if (!is_valid_pos(block_at_P)) return false; 
        if (is_blocked_pos(block_at_P, g)) return false; 

        BFSResult path_cur_to_target_P = bfs(cur_P, target_P, g, INVALID_POS, false, build_action_str); 
        if (path_cur_to_target_P.cost == INF_COST) return false;

        Grid g_after_alter = g; 
        toggle_block_pos(block_at_P, g_after_alter); 
        char alter_dir_char_for_block = DIR_CHARS[slide_dir_idx]; 

        BFSResult path_target_to_slide_start_P = bfs(target_P, slide_start_P, g_after_alter, 
                                                     target_P, true, build_action_str); 
        if (path_target_to_slide_start_P.cost == INF_COST) return false;
        
        res.turns = path_cur_to_target_P.cost + 1 + path_target_to_slide_start_P.cost + 1; 
        if (build_action_str) {
            res.actions_str = path_cur_to_target_P.actions_str; 
            res.actions_str += 'A'; res.actions_str += alter_dir_char_for_block; 
            res.actions_str += path_target_to_slide_start_P.actions_str; 
            res.actions_str += 'S'; res.actions_str += slide_dir_char; 
        } else {
            res.actions_str.clear();
        }
        g = g_after_alter; 
        return true;
    }
    return false; 
}

const int NUM_BASE_STRATEGIES_DIRECT = 1;
const int NUM_BASE_STRATEGIES_UNBLOCK = 1;
const int NUM_BASE_STRATEGIES_SLIDE_TYPE0 = 4;
const int NUM_BASE_STRATEGIES_SLIDE_TYPE1 = 4;
const int NUM_BASE_STRATEGIES = NUM_BASE_STRATEGIES_DIRECT + NUM_BASE_STRATEGIES_UNBLOCK + 
                                NUM_BASE_STRATEGIES_SLIDE_TYPE0 + NUM_BASE_STRATEGIES_SLIDE_TYPE1; // 1+1+4+4 = 10

bool apply_base_strategy_internal(int base_code, Pos cur_P, Pos target_P, Grid& g, SegmentExecResult& res, bool build_action_str) {
    if (base_code == 0) return apply_direct_path_strat(cur_P, target_P, g, res, build_action_str);
    if (base_code == 1) return apply_unblock_and_go_strat(cur_P, target_P, g, res, build_action_str);
    
    int type = -1, dir_idx = -1; 
    if (base_code >= 2 && base_code < 2 + NUM_BASE_STRATEGIES_SLIDE_TYPE0) { 
        type = 0; dir_idx = base_code - 2; 
    }        
    else if (base_code >= 2 + NUM_BASE_STRATEGIES_SLIDE_TYPE0 && 
             base_code < 2 + NUM_BASE_STRATEGIES_SLIDE_TYPE0 + NUM_BASE_STRATEGIES_SLIDE_TYPE1) { 
        type = 1; dir_idx = base_code - (2 + NUM_BASE_STRATEGIES_SLIDE_TYPE0); 
    }   
    else return false; 
    
    return apply_slide_strat(cur_P, target_P, g, res, dir_idx, type, build_action_str);
}

const int NUM_POST_ALTER_OPTIONS_NONE = 1;
const int NUM_POST_ALTER_OPTIONS_ADJACENT = 4;
const int NUM_POST_ALTER_OPTIONS_MOVE_PLUS_ALTER = 12; 
const int NUM_POST_ALTER_OPTIONS_CUMULATIVE_NONE = NUM_POST_ALTER_OPTIONS_NONE; 
const int NUM_POST_ALTER_OPTIONS_CUMULATIVE_ADJACENT = NUM_POST_ALTER_OPTIONS_CUMULATIVE_NONE + NUM_POST_ALTER_OPTIONS_ADJACENT; 
const int NUM_POST_ALTER_OPTIONS = NUM_POST_ALTER_OPTIONS_CUMULATIVE_ADJACENT + NUM_POST_ALTER_OPTIONS_MOVE_PLUS_ALTER; 
const int TOTAL_STRATEGIES_PER_SEGMENT = NUM_BASE_STRATEGIES * NUM_POST_ALTER_OPTIONS; // 10 * 17 = 170
const int GREEDY_REOPTIMIZE_SUBSET_SIZE = 40;


bool apply_combined_strategy(int combined_code, Pos& player_pos_ref , 
                             Pos segment_target_P, Grid& g , 
                             SegmentExecResult& res , bool build_action_str) {
    res.turns = 0; 
    res.actions_str.clear();

    int base_strategy_code = combined_code % NUM_BASE_STRATEGIES;
    int post_alter_option_code = combined_code / NUM_BASE_STRATEGIES; 

    Pos player_original_pos_at_segment_start = player_pos_ref; 
    Grid g_original_at_segment_start = g;

    bool base_success = apply_base_strategy_internal(base_strategy_code, player_original_pos_at_segment_start, segment_target_P, g, res, build_action_str);
    
    if (!base_success) {
        g = g_original_at_segment_start; 
        return false;
    }

    Pos player_pos_after_base = segment_target_P; 

    if (post_alter_option_code == 0) { 
        // No action
    } else if (post_alter_option_code < NUM_POST_ALTER_OPTIONS_CUMULATIVE_ADJACENT) { 
        int alter_dir_idx = post_alter_option_code - NUM_POST_ALTER_OPTIONS_CUMULATIVE_NONE; 
        Pos alter_on_P = {player_pos_after_base.r + DR[alter_dir_idx], player_pos_after_base.c + DC[alter_dir_idx]};

        if (!is_valid_pos(alter_on_P)) { 
            g = g_original_at_segment_start; 
            return false; 
        }

        res.turns++;
        if (build_action_str) {
            res.actions_str += 'A'; 
            res.actions_str += DIR_CHARS[alter_dir_idx];
        }
        toggle_block_pos(alter_on_P, g); 
    } else { 
        int offset_code = post_alter_option_code - NUM_POST_ALTER_OPTIONS_CUMULATIVE_ADJACENT; 
        int D1_idx_move = offset_code / 3; 
        int D2_choice_idx_alter = offset_code % 3; 

        int D2_idx_alter = -1;
        int current_choice_count = 0;
        for (int d_candidate = 0; d_candidate < 4; ++d_candidate) {
            if (d_candidate == DIR_REV_IDX[D1_idx_move]) continue; 
            if (current_choice_count == D2_choice_idx_alter) {
                D2_idx_alter = d_candidate;
                break;
            }
            current_choice_count++;
        }
        
        Pos S1_moved_pos = {player_pos_after_base.r + DR[D1_idx_move], player_pos_after_base.c + DC[D1_idx_move]};
        if (!is_valid_pos(S1_moved_pos) || is_blocked_pos(S1_moved_pos, g)) { 
            g = g_original_at_segment_start; 
            return false; 
        }
        
        Pos S2_target_of_alter = {S1_moved_pos.r + DR[D2_idx_alter], S1_moved_pos.c + DC[D2_idx_alter]};
        if (!is_valid_pos(S2_target_of_alter)) { 
            g = g_original_at_segment_start; 
            return false; 
        }

        res.turns += 2; 
        if (build_action_str) {
            res.actions_str += 'M'; res.actions_str += DIR_CHARS[D1_idx_move];
            res.actions_str += 'A'; res.actions_str += DIR_CHARS[D2_idx_alter];
        }
        toggle_block_pos(S2_target_of_alter, g);
        player_pos_after_base = S1_moved_pos; 
    }

    player_pos_ref = player_pos_after_base; 
    return true;
}

struct PathCacheEntry { Pos player_pos_before_segment; Grid grid_before_segment; int turns_before_segment; };
struct FullEvalResult { int total_turns; std::string actions_log; bool possible; };

FullEvalResult evaluate_choices(const std::vector<int>& choices, Pos initial_P, const std::vector<Pos>& targets,
                                bool build_action_str, int k_eval_start_idx, 
                                const std::vector<PathCacheEntry>* reference_path_cache, 
                                std::vector<PathCacheEntry>* path_cache_for_new_state) { 
    Grid current_grid_sim; Pos player_pos_sim; int total_turns_sim = 0;
    std::string total_actions_log_sim_segments_builder = ""; 

    if (k_eval_start_idx == 0 || reference_path_cache == nullptr || reference_path_cache->empty() || (NUM_SEGMENTS > 0 && k_eval_start_idx >= static_cast<int>(reference_path_cache->size())) ) { 
        for(int r=0; r<N_GRID; ++r) for(int c=0; c<N_GRID; ++c) current_grid_sim[r][c] = false;
        player_pos_sim = initial_P;
        total_turns_sim = 0;
        if (k_eval_start_idx != 0 && NUM_SEGMENTS > 0) k_eval_start_idx = 0; 
    } else { 
        const PathCacheEntry& prev_entry = (*reference_path_cache)[k_eval_start_idx];
        current_grid_sim = prev_entry.grid_before_segment; 
        player_pos_sim = prev_entry.player_pos_before_segment;
        total_turns_sim = prev_entry.turns_before_segment;
        if (total_turns_sim == INF_COST) {
             return {INF_COST, "", false};
        }
    }

    if (path_cache_for_new_state != nullptr && k_eval_start_idx > 0 && reference_path_cache != nullptr && !reference_path_cache->empty() &&
        static_cast<int>(path_cache_for_new_state->size()) >= k_eval_start_idx && static_cast<int>(reference_path_cache->size()) >= k_eval_start_idx) { 
        std::copy(reference_path_cache->begin(), reference_path_cache->begin() + k_eval_start_idx, path_cache_for_new_state->begin());
    }
    
    for (int seg_idx = k_eval_start_idx; seg_idx < NUM_SEGMENTS; ++seg_idx) {
        if (path_cache_for_new_state != nullptr && !path_cache_for_new_state->empty() && static_cast<int>(path_cache_for_new_state->size()) > seg_idx) {
            (*path_cache_for_new_state)[seg_idx].player_pos_before_segment = player_pos_sim;
            (*path_cache_for_new_state)[seg_idx].grid_before_segment = current_grid_sim; 
            (*path_cache_for_new_state)[seg_idx].turns_before_segment = total_turns_sim;
        }

        Pos target_P_for_segment = targets[seg_idx]; 
        SegmentExecResult segment_res; 
        
        bool success = apply_combined_strategy(choices[seg_idx], player_pos_sim, target_P_for_segment, current_grid_sim, segment_res, build_action_str);
        
        if (!success || segment_res.turns == INF_COST || total_turns_sim + segment_res.turns > MAX_TOTAL_TURNS) { 
             if (path_cache_for_new_state != nullptr && !path_cache_for_new_state->empty()) { 
                 for(int fill_inf_idx = seg_idx; fill_inf_idx < NUM_SEGMENTS; ++fill_inf_idx) {
                    if (static_cast<int>(path_cache_for_new_state->size()) > fill_inf_idx)
                        (*path_cache_for_new_state)[fill_inf_idx].turns_before_segment = INF_COST; 
                 }
             }
            return {INF_COST, "", false}; 
        }

        if (build_action_str) total_actions_log_sim_segments_builder += segment_res.actions_str;
        total_turns_sim += segment_res.turns; 
    }
    return {total_turns_sim, total_actions_log_sim_segments_builder, true};
}

auto time_start = std::chrono::steady_clock::now();
double get_elapsed_time_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - time_start).count(); }
const double TIME_LIMIT_MS = 1950.0; 

enum class NeighborhoodOpType { 
    RANDOM_MULTI_SEGMENT,
    FINE_TWEAK_SINGLE_SEGMENT,
    GREEDY_REOPTIMIZE_SINGLE_SEGMENT
};

int main() {
    std::ios_base::sync_with_stdio(false); std::cin.tie(NULL);
    
    int N_in_dummy, M_in_dummy; std::cin >> N_in_dummy >> M_in_dummy; 
    std::cin >> G_initial_pos.r >> G_initial_pos.c;
    
    if (NUM_SEGMENTS > 0) { 
        G_targets_vec.resize(NUM_SEGMENTS); 
        for (int k=0; k < NUM_SEGMENTS; ++k) std::cin >> G_targets_vec[k].r >> G_targets_vec[k].c;
    }

    std::vector<int> current_sa_choices(NUM_SEGMENTS > 0 ? NUM_SEGMENTS : 0); 
    std::vector<int> best_sa_choices(NUM_SEGMENTS > 0 ? NUM_SEGMENTS : 0);
    
    std::vector<PathCacheEntry> current_path_cache(NUM_SEGMENTS > 0 ? NUM_SEGMENTS : 0); 
    std::vector<PathCacheEntry> neighbor_path_cache(NUM_SEGMENTS > 0 ? NUM_SEGMENTS : 0); 
    
    int current_total_turns = INF_COST; 
    int best_total_turns = INF_COST;
    std::string best_actions_log_str = ""; 
    bool best_is_from_sa = false; 
    int initial_greedy_score_turns = INF_COST; 

    if (NUM_SEGMENTS == 0) { 
        // No actions
    } else {
        Grid greedy_grid_sim_build; 
        for(int r=0; r<N_GRID; ++r) for(int c=0; c<N_GRID; ++c) greedy_grid_sim_build[r][c] = false;
        Pos player_pos_sim_build = G_initial_pos;
        std::string greedy_actions_log_build_temp = ""; 
        int greedy_total_turns_build_temp = 0; 
        bool possible_greedy = true;

        for (int k = 0; k < NUM_SEGMENTS; ++k) {
            current_path_cache[k].player_pos_before_segment = player_pos_sim_build;
            current_path_cache[k].grid_before_segment = greedy_grid_sim_build;
            current_path_cache[k].turns_before_segment = greedy_total_turns_build_temp;

            Pos target_P_k = G_targets_vec[k]; 
            int current_best_strategy_code_for_k = -1;
            int current_min_turns_for_segment_k = INF_COST;
            
            for (int code = 0; code < TOTAL_STRATEGIES_PER_SEGMENT; ++code) {
                SegmentExecResult temp_segment_res_eval; 
                Grid temp_grid_eval = greedy_grid_sim_build; 
                Pos temp_player_pos_eval = player_pos_sim_build; 
                
                bool success = apply_combined_strategy(code, temp_player_pos_eval, target_P_k, temp_grid_eval, temp_segment_res_eval, false); 
                
                if (success && temp_segment_res_eval.turns < current_min_turns_for_segment_k) {
                    current_min_turns_for_segment_k = temp_segment_res_eval.turns;
                    current_best_strategy_code_for_k = code;
                }
            }

            if (current_best_strategy_code_for_k == -1 || greedy_total_turns_build_temp + current_min_turns_for_segment_k > MAX_TOTAL_TURNS) { 
                possible_greedy = false; break; 
            }
            
            current_sa_choices[k] = current_best_strategy_code_for_k;
            
            SegmentExecResult final_segment_res_for_k_build;
            apply_combined_strategy(current_best_strategy_code_for_k, 
                                    player_pos_sim_build, 
                                    target_P_k, 
                                    greedy_grid_sim_build, 
                                    final_segment_res_for_k_build, 
                                    true); 

            greedy_actions_log_build_temp += final_segment_res_for_k_build.actions_str;
            greedy_total_turns_build_temp += final_segment_res_for_k_build.turns;
        }

        if(possible_greedy) {
            current_total_turns = greedy_total_turns_build_temp; 
            best_total_turns = greedy_total_turns_build_temp;
            initial_greedy_score_turns = greedy_total_turns_build_temp;
            best_sa_choices = current_sa_choices; 
            best_actions_log_str = greedy_actions_log_build_temp;
        } else { 
            Grid fallback_grid_sim; for(int r=0; r<N_GRID; ++r) for(int c=0; c<N_GRID; ++c) fallback_grid_sim[r][c] = false;
            Pos fallback_player_pos = G_initial_pos;
            int fallback_total_turns = 0;

            for(int k_fallback=0; k_fallback<NUM_SEGMENTS; ++k_fallback) {
                current_path_cache[k_fallback].player_pos_before_segment = fallback_player_pos;
                current_path_cache[k_fallback].grid_before_segment = fallback_grid_sim;
                current_path_cache[k_fallback].turns_before_segment = fallback_total_turns;

                Pos target_P_k_fallback = G_targets_vec[k_fallback];
                int chosen_code_fallback = -1;
                SegmentExecResult res_simple_direct, res_simple_unblock;
                
                Grid temp_grid_direct = fallback_grid_sim; Pos temp_pos_direct = fallback_player_pos;
                bool success_direct = apply_combined_strategy(0, temp_pos_direct, target_P_k_fallback, temp_grid_direct, res_simple_direct, false);

                Grid temp_grid_unblock = fallback_grid_sim; Pos temp_pos_unblock = fallback_player_pos;
                bool success_unblock = apply_combined_strategy(1, temp_pos_unblock, target_P_k_fallback, temp_grid_unblock, res_simple_unblock, false);
                
                if (success_direct && (!success_unblock || res_simple_direct.turns <= res_simple_unblock.turns)) {
                    chosen_code_fallback = 0;
                } else if (success_unblock) {
                    chosen_code_fallback = 1; 
                } else {
                    chosen_code_fallback = std::uniform_int_distribution<>(0, TOTAL_STRATEGIES_PER_SEGMENT - 1)(rng);
                }
                current_sa_choices[k_fallback] = chosen_code_fallback;
                
                SegmentExecResult temp_res_chosen_fallback;
                bool success_chosen_fb = apply_combined_strategy(chosen_code_fallback, fallback_player_pos, target_P_k_fallback, fallback_grid_sim, temp_res_chosen_fallback, false);
                if (!success_chosen_fb || fallback_total_turns + temp_res_chosen_fallback.turns > MAX_TOTAL_TURNS) {
                    for(int fill_idx = k_fallback; fill_idx < NUM_SEGMENTS; ++fill_idx) {
                        if (static_cast<int>(current_path_cache.size()) > fill_idx)
                            current_path_cache[fill_idx].turns_before_segment = INF_COST;
                    }
                    break; 
                }
                fallback_total_turns += temp_res_chosen_fallback.turns;
            }
            
            FullEvalResult fallback_eval = evaluate_choices(current_sa_choices, G_initial_pos, G_targets_vec, false, 0, nullptr, &current_path_cache);
            if (fallback_eval.possible) {
                current_total_turns = fallback_eval.total_turns;
                if (current_total_turns < best_total_turns) { 
                    best_total_turns = current_total_turns;
                    best_sa_choices = current_sa_choices;
                    best_is_from_sa = true; 
                }
            } else { current_total_turns = INF_COST; } 
            
            if (current_total_turns == INF_COST) { 
                for(int k_rand_init=0; k_rand_init<NUM_SEGMENTS; ++k_rand_init) {
                     current_sa_choices[k_rand_init] = std::uniform_int_distribution<>(0, TOTAL_STRATEGIES_PER_SEGMENT - 1)(rng);
                }
                FullEvalResult random_init_eval = evaluate_choices(current_sa_choices, G_initial_pos, G_targets_vec, false, 0, nullptr, &current_path_cache);
                if (random_init_eval.possible) {
                    current_total_turns = random_init_eval.total_turns;
                    if (current_total_turns < best_total_turns) {
                         best_total_turns = current_total_turns;
                         best_sa_choices = current_sa_choices;
                         best_is_from_sa = true;
                    }
                }
            }
        }
        
        double T_param_start = 20.0, T_param_end = 0.01; 
        std::vector<int> segment_indices_for_shuffle(NUM_SEGMENTS);
        if (NUM_SEGMENTS > 0) std::iota(segment_indices_for_shuffle.begin(), segment_indices_for_shuffle.end(), 0);

        int iterations_stuck_at_inf = 0;
        const int MAX_STUCK_ITERATIONS_FOR_RANDOM_RESTART = 50;

        while (get_elapsed_time_ms() < TIME_LIMIT_MS) {
            if (current_total_turns == INF_COST) {
                iterations_stuck_at_inf++;
                if (iterations_stuck_at_inf > MAX_STUCK_ITERATIONS_FOR_RANDOM_RESTART) { 
                    iterations_stuck_at_inf = 0;
                    for(int k_rand_init=0; k_rand_init<NUM_SEGMENTS; ++k_rand_init) {
                        current_sa_choices[k_rand_init] = std::uniform_int_distribution<>(0, TOTAL_STRATEGIES_PER_SEGMENT - 1)(rng);
                    }
                    FullEvalResult random_restart_eval = evaluate_choices(current_sa_choices, G_initial_pos, G_targets_vec, false, 0, nullptr, &current_path_cache);
                    if (random_restart_eval.possible) {
                        current_total_turns = random_restart_eval.total_turns;
                        if (current_total_turns < best_total_turns) {
                            best_total_turns = current_total_turns;
                            best_sa_choices = current_sa_choices;
                            best_is_from_sa = true;
                        }
                    } 
                }
            } else {
                iterations_stuck_at_inf = 0;
            }

            if (NUM_SEGMENTS == 0) break;

            std::vector<int> neighbor_sa_choices_temp = current_sa_choices; 
            int k_eval_start_idx = NUM_SEGMENTS; 
            bool changed_anything_in_choices_vector = false;
            
            double op_type_roll = std::uniform_real_distribution<>(0.0, 1.0)(rng);
            NeighborhoodOpType current_op_type_local;

            if (op_type_roll < 0.50) current_op_type_local = NeighborhoodOpType::RANDOM_MULTI_SEGMENT;
            else if (op_type_roll < 0.85) current_op_type_local = NeighborhoodOpType::FINE_TWEAK_SINGLE_SEGMENT;
            else current_op_type_local = NeighborhoodOpType::GREEDY_REOPTIMIZE_SINGLE_SEGMENT;

            if (current_op_type_local == NeighborhoodOpType::RANDOM_MULTI_SEGMENT) {
                int num_local_changes;
                double r_nc_dist = std::uniform_real_distribution<>(0.0, 1.0)(rng);
                int max_pert_base = std::max(1, NUM_SEGMENTS / 5);
                
                if (r_nc_dist < 0.60) num_local_changes = 1;      
                else if (r_nc_dist < 0.85) num_local_changes = 2; 
                else if (r_nc_dist < 0.95) num_local_changes = 3; 
                else num_local_changes = std::min(NUM_SEGMENTS, 
                    static_cast<int>(4 + std::uniform_int_distribution<>(0, std::max(0, max_pert_base - 4))(rng))
                );
                
                num_local_changes = std::min(num_local_changes, NUM_SEGMENTS);
                num_local_changes = std::max(1, num_local_changes);
                
                changed_anything_in_choices_vector = true;
                double r_mt_dist = std::uniform_real_distribution<>(0.0, 1.0)(rng);
                if (r_mt_dist < 0.80 || num_local_changes >= NUM_SEGMENTS ) { 
                    std::shuffle(segment_indices_for_shuffle.begin(), segment_indices_for_shuffle.end(), rng);
                    int min_k_changed_val = NUM_SEGMENTS;
                    for (int i_change = 0; i_change < num_local_changes; ++i_change) {
                        int k_to_change = segment_indices_for_shuffle[i_change];
                        min_k_changed_val = std::min(min_k_changed_val, k_to_change);
                        
                        int old_code = neighbor_sa_choices_temp[k_to_change];
                        int new_code = old_code;
                        if (TOTAL_STRATEGIES_PER_SEGMENT > 1) {
                            do { new_code = std::uniform_int_distribution<>(0, TOTAL_STRATEGIES_PER_SEGMENT - 1)(rng); } while (new_code == old_code);
                        } else { new_code = 0; }
                        neighbor_sa_choices_temp[k_to_change] = new_code;
                    }
                    k_eval_start_idx = min_k_changed_val;
                } else { 
                    int L = num_local_changes;
                    int k_start_block = std::uniform_int_distribution<>(0, NUM_SEGMENTS - L)(rng);
                    for (int i = 0; i < L; ++i) {
                        int k_to_change = k_start_block + i;
                        int old_code = neighbor_sa_choices_temp[k_to_change];
                        int new_code = old_code;
                         if (TOTAL_STRATEGIES_PER_SEGMENT > 1) {
                            do { new_code = std::uniform_int_distribution<>(0, TOTAL_STRATEGIES_PER_SEGMENT - 1)(rng); } while (new_code == old_code);
                        } else { new_code = 0; }
                        neighbor_sa_choices_temp[k_to_change] = new_code;
                    }
                    k_eval_start_idx = k_start_block;
                }
            } else if (current_op_type_local == NeighborhoodOpType::FINE_TWEAK_SINGLE_SEGMENT) {
                changed_anything_in_choices_vector = true;
                int k_to_change = std::uniform_int_distribution<>(0, NUM_SEGMENTS - 1)(rng);
                k_eval_start_idx = k_to_change;
                int current_strategy_code = neighbor_sa_choices_temp[k_to_change];
                int base_code = current_strategy_code % NUM_BASE_STRATEGIES;
                int post_alter_code = current_strategy_code / NUM_BASE_STRATEGIES;
                
                double tweak_type_rand = std::uniform_real_distribution<>(0.0, 1.0)(rng);
                if (tweak_type_rand < 0.5 && NUM_POST_ALTER_OPTIONS > 1) { 
                    int new_post_alter_code = post_alter_code;
                    do { new_post_alter_code = std::uniform_int_distribution<>(0, NUM_POST_ALTER_OPTIONS - 1)(rng); } while (new_post_alter_code == post_alter_code);
                    neighbor_sa_choices_temp[k_to_change] = new_post_alter_code * NUM_BASE_STRATEGIES + base_code;
                } else if (NUM_BASE_STRATEGIES > 1) { 
                    int new_base_code = base_code;
                    do { new_base_code = std::uniform_int_distribution<>(0, NUM_BASE_STRATEGIES - 1)(rng); } while (new_base_code == base_code);
                    neighbor_sa_choices_temp[k_to_change] = post_alter_code * NUM_BASE_STRATEGIES + new_base_code;
                } else { 
                     if (TOTAL_STRATEGIES_PER_SEGMENT > 1) { 
                        int new_code = current_strategy_code;
                        do { new_code = std::uniform_int_distribution<>(0, TOTAL_STRATEGIES_PER_SEGMENT - 1)(rng); } while (new_code == current_strategy_code);
                        neighbor_sa_choices_temp[k_to_change] = new_code;
                     } else { changed_anything_in_choices_vector = false; }
                }
                if (neighbor_sa_choices_temp[k_to_change] == current_sa_choices[k_to_change]) {
                     changed_anything_in_choices_vector = false;
                }

            } else { // GREEDY_REOPTIMIZE_SINGLE_SEGMENT
                int k_to_reoptimize = std::uniform_int_distribution<>(0, NUM_SEGMENTS - 1)(rng);
                
                if (current_total_turns == INF_COST || current_path_cache.empty() || 
                    k_to_reoptimize >= static_cast<int>(current_path_cache.size()) || 
                    current_path_cache[k_to_reoptimize].turns_before_segment == INF_COST) {
                     changed_anything_in_choices_vector = false; 
                } else {
                    k_eval_start_idx = k_to_reoptimize;

                    Pos player_pos_before_k = current_path_cache[k_to_reoptimize].player_pos_before_segment;
                    Grid grid_before_k = current_path_cache[k_to_reoptimize].grid_before_segment;
                    Pos target_P_k = G_targets_vec[k_to_reoptimize];
                    
                    int original_choice_for_k = current_sa_choices[k_to_reoptimize]; 
                    int best_strategy_for_k = original_choice_for_k;
                    SegmentExecResult best_res_for_k_eval; 
                    
                    Grid temp_grid_eval_current = grid_before_k; Pos temp_player_pos_eval_current = player_pos_before_k;
                    bool current_choice_possible = apply_combined_strategy(original_choice_for_k, temp_player_pos_eval_current, target_P_k, temp_grid_eval_current, best_res_for_k_eval, false);
                    if (!current_choice_possible) best_res_for_k_eval.turns = INF_COST;

                    for (int i = 0; i < GREEDY_REOPTIMIZE_SUBSET_SIZE; ++i) {
                        int code_to_try = std::uniform_int_distribution<>(0, TOTAL_STRATEGIES_PER_SEGMENT - 1)(rng);
                        if (code_to_try == original_choice_for_k && current_choice_possible) {
                            continue; 
                        }

                        SegmentExecResult current_segment_res_eval; 
                        Grid temp_grid_iter_eval = grid_before_k; 
                        Pos temp_player_pos_iter_eval = player_pos_before_k;
                        bool success = apply_combined_strategy(code_to_try, temp_player_pos_iter_eval, target_P_k, temp_grid_iter_eval, current_segment_res_eval, false); 
                        
                        if (success && current_segment_res_eval.turns < best_res_for_k_eval.turns) {
                            best_res_for_k_eval.turns = current_segment_res_eval.turns;
                            best_strategy_for_k = code_to_try;
                        }
                    }
                    neighbor_sa_choices_temp[k_to_reoptimize] = best_strategy_for_k;
                    if (best_strategy_for_k != original_choice_for_k) {
                        changed_anything_in_choices_vector = true;
                    }
                }
            }
            
            if (!changed_anything_in_choices_vector) continue;

            FullEvalResult neighbor_eval_res = evaluate_choices(neighbor_sa_choices_temp, G_initial_pos, G_targets_vec, 
                                                                false, k_eval_start_idx, 
                                                                &current_path_cache, &neighbor_path_cache);

            if (neighbor_eval_res.possible) {
                bool accepted = false;
                if (neighbor_eval_res.total_turns < current_total_turns) { accepted = true; } 
                else if (current_total_turns != INF_COST) { 
                    double temperature = T_param_start; 
                    double progress = get_elapsed_time_ms() / TIME_LIMIT_MS;
                    if (progress < 1.0 && progress >=0.0) { temperature = T_param_start * std::pow(T_param_end / T_param_start, progress); } 
                    else if (progress >= 1.0) { temperature = T_param_end; }
                    temperature = std::max(temperature, T_param_end); 
                    if (temperature > 1e-9) { 
                        double delta_cost = static_cast<double>(neighbor_eval_res.total_turns - current_total_turns); 
                        if (std::exp(-delta_cost / temperature) > std::uniform_real_distribution<>(0.0, 1.0)(rng) ) { accepted = true; }
                    }
                } else { 
                    accepted = true; 
                }

                if (accepted) {
                    current_sa_choices.swap(neighbor_sa_choices_temp); 
                    current_total_turns = neighbor_eval_res.total_turns;
                    if (!current_path_cache.empty() && !neighbor_path_cache.empty()) { 
                         current_path_cache.swap(neighbor_path_cache); 
                    }
                    
                    if (current_total_turns < best_total_turns) {
                        best_total_turns = current_total_turns; 
                        best_sa_choices = current_sa_choices; 
                        best_is_from_sa = true; 
                    }
                }
            }
        }
        
        if (best_total_turns == INF_COST) { 
            best_actions_log_str = ""; 
        } else {
            if (best_is_from_sa || !possible_greedy || best_total_turns < initial_greedy_score_turns) {
                 FullEvalResult final_best_res = evaluate_choices(best_sa_choices, G_initial_pos, G_targets_vec, true, 0, nullptr, nullptr);
                 if (final_best_res.possible) { 
                    best_actions_log_str = final_best_res.actions_log;
                 } else { 
                    best_actions_log_str = ""; 
                 }
            }
        }
    }

    const std::string& final_actions_to_print = best_actions_log_str;
    for (size_t i = 0; i < final_actions_to_print.length(); i += 2) {
        std::cout << final_actions_to_print[i] << " " << final_actions_to_print[i+1] << "\n";
    }
    return 0;
}
# EVOLVE-BLOCK-END