# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
// #include <map> 
// #include <set>  
#include <queue>
#include <cmath> 
#include <iomanip> 
#include <limits> 

// --- Constants ---
constexpr int GRID_SIZE = 30;
constexpr int NUM_TURNS = 300;
constexpr int INF = std::numeric_limits<int>::max(); 

struct Point {
    int r, c;

    bool operator==(const Point& other) const { return r == other.r && c == other.c; }
    bool operator!=(const Point& other) const { return !(*this == other); }
    bool operator<(const Point& other) const { 
        if (r != other.r) return r < other.r;
        return c < other.c;
    }
};
const Point INVALID_POINT = {-1, -1};


// Tunable parameters
constexpr int STAND_OUTSIDE_INNER_SAFE_PENALTY = 1000; 
constexpr int ADJACENT_WALL_PRIORITY_BONUS = 0; 
constexpr int NEAR_PET_PENALTY_POINTS_PER_PET = 0; 
constexpr int NEAR_PET_RADIUS = 2; 
constexpr int MAX_STUCK_TURNS = 10; // Slightly increased

// Directions: Up, Down, Left, Right (indices 0, 1, 2, 3)
const Point DIRS[4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; 
const char DIR_CHARS_BUILD[4] = {'u', 'd', 'l', 'r'}; 
const char DIR_CHARS_MOVE[4] = {'U', 'D', 'L', 'R'};   
const char PET_MOVE_CHARS[4] = {'U', 'D', 'L', 'R'}; 

struct PetInfo {
    Point pos;
    int type; 
    int id;
};

enum class HumanObjective { 
    BUILDING_WALLS, 
    GOING_TO_SAFE_SPOT, 
    STAYING_IN_SAFE_SPOT, 
    REPOSITIONING_STUCK
    // FLEEING_PET_IN_PEN removed, simplified objective setting
};

struct HumanInfo {
    Point pos;
    int id;
    
    int strip_r_start; 
    int strip_r_end;   
    
    Point inner_safe_ul;     
    Point inner_safe_br;     
    Point final_stand_pos;  

    std::vector<Point> assigned_wall_cells; 
    HumanObjective objective; 
    int turns_stuck_building = 0; 
};

// --- Game Grid and State ---
bool is_impassable_grid_static[GRID_SIZE + 1][GRID_SIZE + 1]; 
std::vector<PetInfo> pets_global_state;
std::vector<HumanInfo> humans_global_state;
int N_pets_global, M_humans_global;

Point bfs_parent_grid[GRID_SIZE + 1][GRID_SIZE + 1];
bool bfs_visited_grid[GRID_SIZE + 1][GRID_SIZE + 1];


// --- Utility Functions ---
bool is_valid_coord(int val) {
    return val >= 1 && val <= GRID_SIZE;
}

bool is_valid_point(Point p) {
    return is_valid_coord(p.r) && is_valid_coord(p.c);
}

int manhattan_distance(Point p1, Point p2) {
    if (!is_valid_point(p1) || !is_valid_point(p2)) return INF;
    return std::abs(p1.r - p2.r) + std::abs(p1.c - p2.c);
}

int count_adjacent_walls_or_boundaries(Point p) {
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        Point neighbor = {p.r + DIRS[i].r, p.c + DIRS[i].c};
        if (!is_valid_point(neighbor) || (is_valid_point(neighbor) && is_impassable_grid_static[neighbor.r][neighbor.c])) {
            count++;
        }
    }
    return count;
}

bool can_theoretically_build_at(Point wall_pos, int builder_human_id) {
    if (!is_valid_point(wall_pos)) return false;
    if (is_impassable_grid_static[wall_pos.r][wall_pos.c]) return false; 

    for (const auto& pet : pets_global_state) {
        if (pet.pos == wall_pos) return false; 
        if (manhattan_distance(wall_pos, pet.pos) == 1) return false; 
    }

    for (const auto& human : humans_global_state) { 
        if (human.id == builder_human_id) continue; // Builder themself can be adjacent
        if (human.pos == wall_pos) return false; // Other human on the wall_pos
    }
    return true;
}

char get_bfs_move_char(Point start_pos, Point target_pos, 
                       const std::vector<Point>& current_turn_tentative_walls) {
    if (start_pos == target_pos) return '.';
    
    std::queue<Point> q;
    q.push(start_pos);
    
    for(int r_bfs = 1; r_bfs <= GRID_SIZE; ++r_bfs) for(int c_bfs = 1; c_bfs <= GRID_SIZE; ++c_bfs) {
        bfs_visited_grid[r_bfs][c_bfs] = false;
        bfs_parent_grid[r_bfs][c_bfs] = INVALID_POINT;
    }
    if (!is_valid_point(start_pos)) return '.'; 
    bfs_visited_grid[start_pos.r][start_pos.c] = true;

    Point path_found_dest = INVALID_POINT; 

    while(!q.empty()){
        Point curr = q.front();
        q.pop();

        for(int i_dir=0; i_dir < 4; ++i_dir){ 
            Point next_p = {curr.r + DIRS[i_dir].r, curr.c + DIRS[i_dir].c};
            
            if(is_valid_point(next_p) && 
               !is_impassable_grid_static[next_p.r][next_p.c] && 
               !bfs_visited_grid[next_p.r][next_p.c]){
                
                bool is_tentative_wall_conflict = false;
                for(const auto& tw : current_turn_tentative_walls) {
                    if(next_p == tw) {
                        is_tentative_wall_conflict = true;
                        break;
                    }
                }
                if(is_tentative_wall_conflict) continue;

                bfs_visited_grid[next_p.r][next_p.c] = true;
                bfs_parent_grid[next_p.r][next_p.c] = curr;

                if (next_p == target_pos) { 
                    path_found_dest = next_p; 
                    goto bfs_done_label; 
                } 
                q.push(next_p);
            }
        }
    }

bfs_done_label:; 
    if (path_found_dest.r == -1) return '.'; 

    Point current_step_in_path = path_found_dest;
    while(!(bfs_parent_grid[current_step_in_path.r][current_step_in_path.c] == INVALID_POINT) &&
          !(bfs_parent_grid[current_step_in_path.r][current_step_in_path.c] == start_pos)) {
        current_step_in_path = bfs_parent_grid[current_step_in_path.r][current_step_in_path.c];
    }
    
    for(int i_dir = 0; i_dir < 4; ++i_dir){ 
        if(start_pos.r + DIRS[i_dir].r == current_step_in_path.r && 
           start_pos.c + DIRS[i_dir].c == current_step_in_path.c){
            return DIR_CHARS_MOVE[i_dir];
        }
    }
    return '.'; 
}


void initialize_game() {
    /** Initialize pets, people, and strip plans; only internal separators
        are assigned because the board boundary already acts as a wall. */
    std::cin >> N_pets_global;
    pets_global_state.resize(N_pets_global);
    for (int i = 0; i < N_pets_global; ++i) {
        pets_global_state[i].id = i;
        std::cin >> pets_global_state[i].pos.r >> pets_global_state[i].pos.c >> pets_global_state[i].type;
    }

    std::cin >> M_humans_global;
    humans_global_state.resize(M_humans_global);

    for(int r_grid=0; r_grid <= GRID_SIZE; ++r_grid) for(int c_grid=0; c_grid <= GRID_SIZE; ++c_grid) is_impassable_grid_static[r_grid][c_grid] = false;
    
    int base_strip_height = GRID_SIZE / M_humans_global;
    int remainder_heights = GRID_SIZE % M_humans_global;
    int current_r_start_coord = 1;

    for (int i = 0; i < M_humans_global; ++i) {
        HumanInfo& human = humans_global_state[i];
        human.id = i;
        std::cin >> human.pos.r >> human.pos.c;
        
        int strip_h_for_this_human = base_strip_height + (i < remainder_heights ? 1 : 0);
        human.strip_r_start = current_r_start_coord;
        human.strip_r_end = human.strip_r_start + strip_h_for_this_human - 1;
        human.strip_r_end = std::min(human.strip_r_end, GRID_SIZE); 
        
        int actual_strip_h = human.strip_r_end - human.strip_r_start + 1;
        int actual_strip_w = GRID_SIZE; 

        human.inner_safe_ul.r = human.strip_r_start + (actual_strip_h >= 3 ? 1 : 0);
        human.inner_safe_ul.c = 1 + (actual_strip_w >= 3 ? 1 : 0); 
        human.inner_safe_br.r = human.strip_r_end - (actual_strip_h >= 3 ? 1 : 0);
        human.inner_safe_br.c = GRID_SIZE - (actual_strip_w >= 3 ? 1 : 0);
        
        if (human.inner_safe_ul.r > human.inner_safe_br.r) human.inner_safe_br.r = human.inner_safe_ul.r;
        if (human.inner_safe_ul.c > human.inner_safe_br.c) human.inner_safe_br.c = human.inner_safe_ul.c;
        
        human.final_stand_pos = {
            human.inner_safe_ul.r + (human.inner_safe_br.r - human.inner_safe_ul.r) / 2,
            human.inner_safe_ul.c + (human.inner_safe_br.c - human.inner_safe_ul.c) / 2
        };
        human.final_stand_pos.r = std::max(human.inner_safe_ul.r, std::min(human.inner_safe_br.r, human.final_stand_pos.r));
        human.final_stand_pos.c = std::max(human.inner_safe_ul.c, std::min(human.inner_safe_br.c, human.final_stand_pos.c));
        if (!is_valid_point(human.final_stand_pos)) { 
            human.final_stand_pos = {human.strip_r_start, 1}; 
        }

        human.assigned_wall_cells.clear();
        int r_s = human.strip_r_start;
        int r_e = human.strip_r_end;

        // A separator is shared by the strips above and below it.  Split
        // each separator into two halves, so every cell is assigned exactly
        // once while adjacent people can construct different halves in
        // parallel.  No outer-edge cells are needed: the board boundary is
        // already impassable by the rules.
        if (i != 0) {
            for (int c_coord = GRID_SIZE / 2 + 1; c_coord <= GRID_SIZE; ++c_coord) {
                human.assigned_wall_cells.push_back({r_s, c_coord});
            }
        }
        if (i != M_humans_global - 1) {
            for (int c_coord = 1; c_coord <= GRID_SIZE / 2; ++c_coord) {
                human.assigned_wall_cells.push_back({r_e, c_coord});
            }
        }
        
        std::sort(human.assigned_wall_cells.begin(), human.assigned_wall_cells.end());
        human.assigned_wall_cells.erase(
            std::unique(human.assigned_wall_cells.begin(), human.assigned_wall_cells.end()),
            human.assigned_wall_cells.end()
        );
        current_r_start_coord = human.strip_r_end + 1; 
    }
}

std::string decide_human_actions() {
    std::string actions_str(M_humans_global, '.');
    std::vector<Point> tentative_walls_this_turn; 
    std::vector<Point> tentative_move_targets_this_turn(M_humans_global, INVALID_POINT); 

    for (int i = 0; i < M_humans_global; ++i) {
        HumanInfo& human = humans_global_state[i];

        int unbuilt_walls_count = 0;
        for (const auto& wall_cell : human.assigned_wall_cells) {
            if (is_valid_point(wall_cell) && !is_impassable_grid_static[wall_cell.r][wall_cell.c]) {
                unbuilt_walls_count++;
            }
        }

        if (unbuilt_walls_count == 0) { 
             human.objective = (human.pos == human.final_stand_pos) ? 
                              HumanObjective::STAYING_IN_SAFE_SPOT : 
                              HumanObjective::GOING_TO_SAFE_SPOT;
        } else { 
            human.objective = HumanObjective::BUILDING_WALLS;
        }
        
        if(human.objective == HumanObjective::BUILDING_WALLS && human.turns_stuck_building >= MAX_STUCK_TURNS) {
            human.objective = HumanObjective::REPOSITIONING_STUCK; 
        }

        char chosen_action_for_human_i = '.';
        if (human.objective == HumanObjective::STAYING_IN_SAFE_SPOT) {
            chosen_action_for_human_i = '.';
        } else if (human.objective == HumanObjective::GOING_TO_SAFE_SPOT || 
                   human.objective == HumanObjective::REPOSITIONING_STUCK) {
            if(human.objective == HumanObjective::REPOSITIONING_STUCK) human.turns_stuck_building = 0; 

            chosen_action_for_human_i = get_bfs_move_char(human.pos, human.final_stand_pos, tentative_walls_this_turn);
        
        } else if (human.objective == HumanObjective::BUILDING_WALLS) {
            Point best_wall_target = INVALID_POINT;
            Point best_stand_point = INVALID_POINT;
            int min_eval_score = INF;

            for (const auto& wall_coord : human.assigned_wall_cells) {
                if (!is_valid_point(wall_coord) || is_impassable_grid_static[wall_coord.r][wall_coord.c]) continue;
                if (!can_theoretically_build_at(wall_coord, human.id)) continue;

                int adj_wall_bonus_val = count_adjacent_walls_or_boundaries(wall_coord) * ADJACENT_WALL_PRIORITY_BONUS;
                int current_near_pet_penalty = 0; // NEAR_PET_PENALTY_POINTS_PER_PET is 0

                for (int k_dir_idx = 0; k_dir_idx < 4; ++k_dir_idx) { 
                    Point potential_stand_pos = {wall_coord.r + DIRS[k_dir_idx].r, 
                                                 wall_coord.c + DIRS[k_dir_idx].c};
                    
                    if (!is_valid_point(potential_stand_pos) || is_impassable_grid_static[potential_stand_pos.r][potential_stand_pos.c]) continue;
                    
                    bool conflict_with_tentative_wall_build_spot = false; 
                    for(const auto& tw : tentative_walls_this_turn) { if(potential_stand_pos == tw) { conflict_with_tentative_wall_build_spot = true; break; }}
                    if(conflict_with_tentative_wall_build_spot) continue;
                    
                    bool conflict_with_tentative_move_dest = false; 
                    for(int j=0; j < i; ++j) { 
                        if (tentative_move_targets_this_turn[j] == potential_stand_pos) { conflict_with_tentative_move_dest = true; break; }
                    }
                    if (conflict_with_tentative_move_dest) continue;

                    int current_dist_to_stand = manhattan_distance(human.pos, potential_stand_pos);
                    int current_eval_score = current_dist_to_stand - adj_wall_bonus_val + current_near_pet_penalty;

                    bool is_inside_inner_safe_region = 
                        (potential_stand_pos.r >= human.inner_safe_ul.r &&
                         potential_stand_pos.r <= human.inner_safe_br.r &&
                         potential_stand_pos.c >= human.inner_safe_ul.c &&
                         potential_stand_pos.c <= human.inner_safe_br.c);
                    
                    if (!is_inside_inner_safe_region) {
                        current_eval_score += STAND_OUTSIDE_INNER_SAFE_PENALTY; 
                    }

                    if (current_eval_score < min_eval_score) {
                        min_eval_score = current_eval_score;
                        best_wall_target = wall_coord;
                        best_stand_point = potential_stand_pos;
                    } else if (current_eval_score == min_eval_score) { 
                        if (best_wall_target.r == -1 || 
                            wall_coord < best_wall_target ||
                            (wall_coord == best_wall_target && potential_stand_pos < best_stand_point)) {
                            best_wall_target = wall_coord;
                            best_stand_point = potential_stand_pos;
                        }
                    }
                }
            }

            if (best_wall_target.r != -1) { 
                human.turns_stuck_building = 0; 
                if (human.pos == best_stand_point) { 
                    for(int k_dir=0; k_dir<4; ++k_dir){ 
                        if(human.pos.r + DIRS[k_dir].r == best_wall_target.r && 
                           human.pos.c + DIRS[k_dir].c == best_wall_target.c){
                            chosen_action_for_human_i = DIR_CHARS_BUILD[k_dir];
                            break;
                        }
                    }
                } else { 
                    chosen_action_for_human_i = get_bfs_move_char(human.pos, best_stand_point, tentative_walls_this_turn);
                }
            } else { 
                if (unbuilt_walls_count > 0) { 
                    human.turns_stuck_building++;
                }
                if (human.pos != human.final_stand_pos) {
                    chosen_action_for_human_i = get_bfs_move_char(human.pos, human.final_stand_pos, tentative_walls_this_turn);
                } else { 
                    chosen_action_for_human_i = '.'; 
                }
            }
        }
        
        actions_str[i] = chosen_action_for_human_i;
        
        if (chosen_action_for_human_i != '.' && (chosen_action_for_human_i == 'u' || chosen_action_for_human_i == 'd' || chosen_action_for_human_i == 'l' || chosen_action_for_human_i == 'r')) {
            for(int k_dir=0; k_dir<4; ++k_dir) {
                if (chosen_action_for_human_i == DIR_CHARS_BUILD[k_dir]) {
                    Point built_wall_pos = {human.pos.r + DIRS[k_dir].r, human.pos.c + DIRS[k_dir].c};
                    if (is_valid_point(built_wall_pos)) { 
                        tentative_walls_this_turn.push_back(built_wall_pos);
                    }
                    break;
                }
            }
        } else if (chosen_action_for_human_i != '.' && (chosen_action_for_human_i == 'U' || chosen_action_for_human_i == 'D' || chosen_action_for_human_i == 'L' || chosen_action_for_human_i == 'R')) {
            for(int k_dir=0; k_dir<4; ++k_dir) {
                if (chosen_action_for_human_i == DIR_CHARS_MOVE[k_dir]) {
                    Point target_pos = {human.pos.r + DIRS[k_dir].r, human.pos.c + DIRS[k_dir].c};
                     if (is_valid_point(target_pos)) { 
                        tentative_move_targets_this_turn[i] = target_pos;
                     } else {
                        actions_str[i] = '.'; 
                     }
                    break;
                }
            }
        }
    }

    for (int i = 0; i < M_humans_global; ++i) {
        if (actions_str[i] != '.' && (actions_str[i] == 'U' || actions_str[i] == 'D' || actions_str[i] == 'L' || actions_str[i] == 'R')) {
            Point target_move_sq = tentative_move_targets_this_turn[i];
            if (target_move_sq.r == -1) { 
                actions_str[i] = '.';
                continue;
            }

            bool conflict_with_wall = false;
            for (const auto& wall_being_built : tentative_walls_this_turn) { 
                if (target_move_sq == wall_being_built) {
                    conflict_with_wall = true;
                    break; 
                }
            }
            if (conflict_with_wall) {
                actions_str[i] = '.'; 
            } else { 
                for (int j = 0; j < i; ++j) { 
                    if (actions_str[j] != '.' && (actions_str[j] == 'U' || actions_str[j] == 'D' || actions_str[j] == 'L' || actions_str[j] == 'R') &&
                        tentative_move_targets_this_turn[j] == target_move_sq) {
                        actions_str[i] = '.'; 
                        break;
                    }
                }
            }
        }
    }
    return actions_str;
}

void apply_actions_and_update_state(const std::string& actions_str_final) {
    for (int i = 0; i < M_humans_global; ++i) {
        char action = actions_str_final[i];
        HumanInfo& human = humans_global_state[i]; 
        if (action != '.' && (action == 'u' || action == 'd' || action == 'l' || action == 'r')) {
            for(int k_dir=0; k_dir<4; ++k_dir){
                if (action == DIR_CHARS_BUILD[k_dir]) {
                    Point wall_pos = {human.pos.r + DIRS[k_dir].r, human.pos.c + DIRS[k_dir].c};
                    if (is_valid_point(wall_pos) && !is_impassable_grid_static[wall_pos.r][wall_pos.c]) { 
                        is_impassable_grid_static[wall_pos.r][wall_pos.c] = true;
                    }
                    break;
                }
            }
        }
    }

    for (int i = 0; i < M_humans_global; ++i) {
        char action = actions_str_final[i];
        HumanInfo& human = humans_global_state[i]; 
        if (action != '.' && (action == 'U' || action == 'D' || action == 'L' || action == 'R')) {
            for(int k_dir=0; k_dir<4; ++k_dir){
                 if (action == DIR_CHARS_MOVE[k_dir]) {
                    Point next_pos = {human.pos.r + DIRS[k_dir].r, human.pos.c + DIRS[k_dir].c};
                    if (is_valid_point(next_pos) && !is_impassable_grid_static[next_pos.r][next_pos.c]) {
                         human.pos = next_pos;
                    } 
                    break;
                }
            }
        }
    }

    for (int i = 0; i < N_pets_global; ++i) {
        std::string pet_moves_str;
        std::cin >> pet_moves_str;
        if (pet_moves_str == ".") continue;

        for (char move_char : pet_moves_str) {
            for(int k_dir=0; k_dir<4; ++k_dir){ 
                if(move_char == PET_MOVE_CHARS[k_dir]){ 
                    pets_global_state[i].pos.r += DIRS[k_dir].r;
                    pets_global_state[i].pos.c += DIRS[k_dir].c;
                    break; 
                }
            }
        }
    }
}

/**
 * Builds a shared 8x8 pet-free corner room.
 * Humans first gather in the selected corner, cooperatively build its two
 * internal border walls, leave one gate open, and close that gate only when
 * no pet is currently inside the room.
 */
/**
 * Selects the least initially occupied corner, gathers people in that corner,
 * constructs a 10x10 two-wall enclosure from its interior, then closes its
 * sole gate only when the room is pet-free and every person is already inside.
 */
void play_shared_corner_room() {
    constexpr int Q = 10;
    int best_corner = 0;
    int best_count = INF;

    auto to_global = [](int corner, int a, int b) -> Point {
        if (corner == 0) return {a, b};             // top-left
        if (corner == 1) return {a, 31 - b};        // top-right
        if (corner == 2) return {31 - a, b};        // bottom-left
        return {31 - a, 31 - b};                    // bottom-right
    };

    for (int corner = 0; corner < 4; ++corner) {
        int cnt = 0;
        for (const auto& pet : pets_global_state) {
            bool in_row = (corner < 2 ? pet.pos.r <= Q : pet.pos.r >= 31 - Q);
            bool in_col = ((corner == 0 || corner == 2) ? pet.pos.c <= Q : pet.pos.c >= 31 - Q);
            if (in_row && in_col) ++cnt;
        }
        if (cnt < best_count) {
            best_count = cnt;
            best_corner = corner;
        }
    }

    Point gathering_point = to_global(best_corner, 4, 4);
    Point gate = to_global(best_corner, Q + 1, Q / 2);

    std::vector<Point> wall_cells;
    for (int b = 1; b <= Q + 1; ++b) {
        Point p = to_global(best_corner, Q + 1, b);
        if (p != gate) wall_cells.push_back(p);
    }
    for (int a = 1; a <= Q; ++a) {
        wall_cells.push_back(to_global(best_corner, a, Q + 1));
    }

    auto is_inside_room = [&](Point p) {
        if (best_corner == 0) return p.r <= Q && p.c <= Q;
        if (best_corner == 1) return p.r <= Q && p.c >= 31 - Q;
        if (best_corner == 2) return p.r >= 31 - Q && p.c <= Q;
        return p.r >= 31 - Q && p.c >= 31 - Q;
    };

    auto can_build_now = [&](Point p) {
        if (!is_valid_point(p) || is_impassable_grid_static[p.r][p.c]) return false;
        for (const auto& h : humans_global_state) {
            if (h.pos == p) return false;
        }
        for (const auto& pet : pets_global_state) {
            if (pet.pos == p || manhattan_distance(p, pet.pos) == 1) return false;
        }
        return true;
    };

    for (int turn = 0; turn < NUM_TURNS; ++turn) {
        std::string actions(M_humans_global, '.');

        bool ordinary_walls_done = true;
        for (Point p : wall_cells) {
            if (!is_impassable_grid_static[p.r][p.c]) {
                ordinary_walls_done = false;
                break;
            }
        }

        int pets_inside = 0;
        for (const auto& pet : pets_global_state) {
            if (is_inside_room(pet.pos)) ++pets_inside;
        }

        // Closing before every person has crossed the gate can permanently
        // leave a person in the pet area, severely reducing the average score.
        bool all_humans_inside = true;
        for (const auto& human : humans_global_state) {
            if (!is_inside_room(human.pos)) {
                all_humans_inside = false;
                break;
            }
        }

        for (int i = 0; i < M_humans_global; ++i) {
            HumanInfo& h = humans_global_state[i];

            // Gather before constructing: this prevents humans being sealed out.
            if (turn < 45) {
                actions[i] = get_bfs_move_char(h.pos, gathering_point, {});
                continue;
            }

            Point target = INVALID_POINT;
            int best_dist = INF;

            // Each person owns a stable subset of non-gate wall cells.
            for (int k = i; k < (int)wall_cells.size(); k += M_humans_global) {
                Point wall = wall_cells[k];
                if (is_impassable_grid_static[wall.r][wall.c]) continue;

                // Always approach from the room side, so builders remain safe.
                for (int d = 0; d < 4; ++d) {
                    Point stand = {wall.r + DIRS[d].r, wall.c + DIRS[d].c};
                    if (!is_valid_point(stand) || is_impassable_grid_static[stand.r][stand.c]) continue;
                    if (!is_inside_room(stand)) continue;
                    int dist = manhattan_distance(h.pos, stand);
                    if (dist < best_dist) {
                        best_dist = dist;
                        target = wall;
                    }
                }
            }

            // After all ordinary wall cells are ready, close the gate only
            // during a pet-free moment.
            // One designated closer is enough.  In particular, avoiding many
            // identical simultaneous build requests keeps movement planning
            // around the narrow gate more stable.
            if (target == INVALID_POINT && i == 0 && ordinary_walls_done &&
                pets_inside == 0 && all_humans_inside &&
                !is_impassable_grid_static[gate.r][gate.c]) {
                target = gate;
            }

            if (target == INVALID_POINT) {
                actions[i] = get_bfs_move_char(h.pos, gathering_point, {});
                continue;
            }

            bool built = false;
            for (int d = 0; d < 4; ++d) {
                Point next = {h.pos.r + DIRS[d].r, h.pos.c + DIRS[d].c};
                if (next == target) {
                    if (can_build_now(target)) actions[i] = DIR_CHARS_BUILD[d];
                    built = true;
                    break;
                }
            }
            if (!built) {
                Point best_stand = INVALID_POINT;
                int stand_dist = INF;
                for (int d = 0; d < 4; ++d) {
                    Point stand = {target.r + DIRS[d].r, target.c + DIRS[d].c};
                    if (!is_valid_point(stand) || is_impassable_grid_static[stand.r][stand.c]) continue;
                    if (!is_inside_room(stand)) continue;
                    int dist = manhattan_distance(h.pos, stand);
                    if (dist < stand_dist) {
                        stand_dist = dist;
                        best_stand = stand;
                    }
                }
                actions[i] = get_bfs_move_char(h.pos, best_stand, {});
            }
        }

        std::cout << actions << std::endl;
        apply_actions_and_update_state(actions);
    }
}

/**
 * Runs the distributed horizontal-strip policy for all 300 turns.
 * Each person independently builds the boundary of its assigned strip and
 * finishes inside that component, yielding multiple pet-isolated areas.
 */
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    initialize_game();

    for (int turn_idx = 0; turn_idx < NUM_TURNS; ++turn_idx) {
        std::string actions_to_perform = decide_human_actions();
        std::cout << actions_to_perform << std::endl;
        apply_actions_and_update_state(actions_to_perform);
    }

    return 0;
}
# EVOLVE-BLOCK-END