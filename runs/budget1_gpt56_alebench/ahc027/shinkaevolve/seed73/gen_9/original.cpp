# EVOLVE-BLOCK-START
#pragma GCC optimize("O3,unroll-loops")

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip> 
#include <cmath>   
// #include <map> // Not used

// Global game data
int N_GRID_SIZE; 
std::vector<std::string> H_WALLS_INFO; 
std::vector<std::string> V_WALLS_INFO; 
int D_SUSC[40][40];

struct Pos {
    int16_t r, c;
    Pos() : r(0), c(0) {} 
    Pos(int16_t r_val, int16_t c_val) : r(r_val), c(c_val) {}

    bool operator==(const Pos& other) const { return r == other.r && c == other.c; }
    bool operator!=(const Pos& other) const { return !(*this == other); }
    bool operator<(const Pos& other) const { 
        if (r != other.r) return r < other.r;
        return c < other.c;
    }
};

constexpr int DR[] = {0, 1, 0, -1}; // R, D, L, U
constexpr int DC[] = {1, 0, -1, 0};
constexpr char DIR_CHARS[] = {'R', 'D', 'L', 'U'};
const int MAX_L_PATH = 100000; 
double MAX_L_PATH_HIGH_THRESHOLD_EFFECTIVE; 
double MIN_L_PATH_LOW_THRESHOLD_EFFECTIVE;  


std::mt19937 RND_ENGINE; 

Pos APSP_PARENT[40][40][40][40]; 
int APSP_DIST[40][40][40][40];   

bool is_valid_pos(int r, int c) {
    return r >= 0 && r < N_GRID_SIZE && c >= 0 && c < N_GRID_SIZE;
}

bool check_wall(Pos p_from, Pos p_to) {
    int dr = p_to.r - p_from.r;
    int dc = p_to.c - p_from.c;
    if (dr == 1) { // Down
        return H_WALLS_INFO[p_from.r][p_from.c] == '1';
    } else if (dr == -1) { // Up
        return H_WALLS_INFO[p_to.r][p_to.c] == '1';
    } else if (dc == 1) { // Right
        return V_WALLS_INFO[p_from.r][p_from.c] == '1';
    } else if (dc == -1) { // Left
        return V_WALLS_INFO[p_from.r][p_to.c] == '1';
    }
    return true; 
}

char get_move_char(Pos p_from, Pos p_to) {
    int dr = p_to.r - p_from.r;
    int dc = p_to.c - p_from.c;
    for(int i=0; i<4; ++i) if(DR[i] == dr && DC[i] == dc) return DIR_CHARS[i];
    return ' '; 
}

char invert_move(char move_char) {
    for(int i=0; i<4; ++i) if(DIR_CHARS[i] == move_char) return DIR_CHARS[(i+2)%4];
    return ' ';
}

void compute_apsp() {
    for (int sr = 0; sr < N_GRID_SIZE; ++sr) {
        for (int sc = 0; sc < N_GRID_SIZE; ++sc) {
            for (int tr = 0; tr < N_GRID_SIZE; ++tr) for (int tc = 0; tc < N_GRID_SIZE; ++tc) APSP_DIST[sr][sc][tr][tc] = -1; 
            
            std::vector<Pos> q; q.reserve(N_GRID_SIZE * N_GRID_SIZE);
            q.push_back(Pos{(int16_t)sr, (int16_t)sc});
            APSP_DIST[sr][sc][sr][sc] = 0;
            
            int head = 0;
            while(head < static_cast<int>(q.size())){
                Pos curr = q[head++];
                for(int i=0; i<4; ++i){
                    Pos next_candidate = Pos{(int16_t)(curr.r + DR[i]), (int16_t)(curr.c + DC[i])};
                    if(is_valid_pos(next_candidate.r, next_candidate.c) && !check_wall(curr, next_candidate) && APSP_DIST[sr][sc][next_candidate.r][next_candidate.c] == -1){
                        APSP_DIST[sr][sc][next_candidate.r][next_candidate.c] = APSP_DIST[sr][sc][curr.r][curr.c] + 1;
                        APSP_PARENT[sr][sc][next_candidate.r][next_candidate.c] = curr;
                        q.push_back(next_candidate);
                    }
                }
            }
        }
    }
}

bool get_apsp_moves(Pos p_from, Pos p_to, std::vector<char>& out_moves) {
    out_moves.clear();
    if (p_from == p_to) return true;
    if (APSP_DIST[p_from.r][p_from.c][p_to.r][p_to.c] == -1) return false; 
    
    out_moves.reserve(APSP_DIST[p_from.r][p_from.c][p_to.r][p_to.c]);
    Pos curr = p_to;
    while(curr != p_from) {
       Pos prev = APSP_PARENT[p_from.r][p_from.c][curr.r][curr.c];
       out_moves.push_back(get_move_char(prev, curr));
       curr = prev;
    }
    std::reverse(out_moves.begin(), out_moves.end());
    return true;
}

std::vector<std::vector<std::vector<int>>> CELL_VISIT_TIMES_GLOBAL_BUFFER; 

struct CellDirtInfo {
    long double weighted_dirt_contribution;
    Pos p;
    bool operator<(const CellDirtInfo& other) const {
        return weighted_dirt_contribution > other.weighted_dirt_contribution; 
    }
};
std::vector<CellDirtInfo> TMP_CELL_DIRT_INFOS_LIST_GLOBAL_BUFFER; 

struct PathData {
    std::vector<char> moves;
    std::vector<Pos> coords; 
    bool visited_flags[40][40]; 
    long double score_val;
    long double total_dirt_sum_numerator; 
    long double cell_dirt_term_sum[40][40]; 

    PathData() : score_val(1e18L), total_dirt_sum_numerator(0.0L) {
        for(int i=0; i<N_GRID_SIZE; ++i) for(int j=0; j<N_GRID_SIZE; ++j) { 
            visited_flags[i][j] = false;
            cell_dirt_term_sum[i][j] = 0.0L;
        }
    }
    
    PathData(const PathData& other) = default;
    PathData(PathData&& other) = default;
    PathData& operator=(const PathData& other) = default;
    PathData& operator=(PathData&& other) = default;
};

bool update_coords_and_visited_flags(PathData& pd) {
    pd.coords.assign(1, Pos{0,0}); 
    if (!pd.moves.empty()) { 
      pd.coords.reserve(pd.moves.size() + 1); 
    }

    for (int r = 0; r < N_GRID_SIZE; ++r) for (int c = 0; c < N_GRID_SIZE; ++c) pd.visited_flags[r][c] = false;
    pd.visited_flags[0][0] = true;

    Pos current_p = Pos{0,0};
    for (char move_char : pd.moves) {
        int dir_idx = -1;
        for (int i = 0; i < 4; ++i) if (DIR_CHARS[i] == move_char) dir_idx = i;
        
        if (dir_idx == -1) return false; 
        
        Pos next_p = Pos{(int16_t)(current_p.r + DR[dir_idx]), (int16_t)(current_p.c + DC[dir_idx])};
        if (!is_valid_pos(next_p.r, next_p.c) || check_wall(current_p, next_p)) return false; 

        current_p = next_p;
        pd.coords.push_back(current_p);
        pd.visited_flags[current_p.r][current_p.c] = true;
    }
    return true;
}

void calculate_score_full(PathData& pd) {
    if (!update_coords_and_visited_flags(pd)) {
        pd.score_val = 1e18L; 
        return;
    }

    if (pd.moves.size() > MAX_L_PATH) {
        pd.score_val = 1e18L; return;
    }
    if (!pd.moves.empty()){ 
        if (pd.coords.back() != Pos{0,0}) { pd.score_val = 1e18L; return;}
    } else { 
        if (N_GRID_SIZE > 1) { 
             pd.score_val = 1e18L; return;
        }
    }
    
    for (int r = 0; r < N_GRID_SIZE; ++r) for (int c = 0; c < N_GRID_SIZE; ++c) {
        if (!pd.visited_flags[r][c]) { pd.score_val = 1e18L; return; }
    }
    
    int L = pd.moves.size();

    if (L == 0) { 
       pd.score_val = (N_GRID_SIZE == 1) ? 0.0L : 1e18L; // N=1 case not in this contest
       pd.total_dirt_sum_numerator = 0;
       if (N_GRID_SIZE == 1) pd.cell_dirt_term_sum[0][0] = 0;
       return;
    }
    
    for(int r=0; r<N_GRID_SIZE; ++r) {
        for(int c=0; c<N_GRID_SIZE; ++c) {
            CELL_VISIT_TIMES_GLOBAL_BUFFER[r][c].clear(); 
        }
    }
    
    for (int t = 1; t <= L; ++t) { 
        Pos p = pd.coords[t]; 
        CELL_VISIT_TIMES_GLOBAL_BUFFER[p.r][p.c].push_back(t);
    }

    pd.total_dirt_sum_numerator = 0;

    for (int r_ = 0; r_ < N_GRID_SIZE; ++r_) {
        for (int c_ = 0; c_ < N_GRID_SIZE; ++c_) {
            const auto& specific_cell_visits = CELL_VISIT_TIMES_GLOBAL_BUFFER[r_][c_];
            long double current_cell_dirt_term = 0; 

            if (!specific_cell_visits.empty()){
                 int num_visits_in_cycle = specific_cell_visits.size();
                 for (int i = 0; i < num_visits_in_cycle; ++i) {
                    long long prev_visit_t = (i == 0) ? ((long long)specific_cell_visits[num_visits_in_cycle - 1] - L) : (long long)specific_cell_visits[i-1];
                    long long cur_visit_t = specific_cell_visits[i];
                    long long delta = cur_visit_t - prev_visit_t; 
                    current_cell_dirt_term += (long double)delta * (delta - 1) / 2.0L;
                }
            } else { 
                current_cell_dirt_term = (long double)L * (L - 1) / 2.0L;
            }
            pd.cell_dirt_term_sum[r_][c_] = current_cell_dirt_term;
            pd.total_dirt_sum_numerator += (long double)D_SUSC[r_][c_] * current_cell_dirt_term;
        }
    }
    pd.score_val = pd.total_dirt_sum_numerator / L;
}

bool initial_dfs_visited[40][40]; 
void generate_initial_dfs_path(int r, int c, PathData& pd) {
    initial_dfs_visited[r][c] = true;
    for (int dir_idx = 0; dir_idx < 4; ++dir_idx) {
        Pos current_p = Pos{(int16_t)r, (int16_t)c};
        Pos next_p = Pos{(int16_t)(r + DR[dir_idx]), (int16_t)(c + DC[dir_idx])};
        if (is_valid_pos(next_p.r, next_p.c) && !check_wall(current_p, next_p) && !initial_dfs_visited[next_p.r][next_p.c]) {
            pd.moves.push_back(DIR_CHARS[dir_idx]);
            generate_initial_dfs_path(next_p.r, next_p.c, pd);
            pd.moves.push_back(DIR_CHARS[(dir_idx + 2) % 4]); 
        }
    }
}

Pos select_target_cell_for_dirt_ops(const PathData& current_pd_obj, bool use_sqrt_N_sampling) {
    TMP_CELL_DIRT_INFOS_LIST_GLOBAL_BUFFER.clear(); 
    if (current_pd_obj.score_val > 1e17L) { 
        std::uniform_int_distribution<int> r_dist(0, N_GRID_SIZE-1);
        return Pos{(int16_t)r_dist(RND_ENGINE), (int16_t)r_dist(RND_ENGINE)};
    }

    for(int r=0; r<N_GRID_SIZE; ++r) {
        for(int c=0; c<N_GRID_SIZE; ++c) {
             TMP_CELL_DIRT_INFOS_LIST_GLOBAL_BUFFER.push_back({(long double)D_SUSC[r][c] * current_pd_obj.cell_dirt_term_sum[r][c], Pos{(int16_t)r,(int16_t)c}});
        }
    }
    std::sort(TMP_CELL_DIRT_INFOS_LIST_GLOBAL_BUFFER.begin(), TMP_CELL_DIRT_INFOS_LIST_GLOBAL_BUFFER.end());

    if (TMP_CELL_DIRT_INFOS_LIST_GLOBAL_BUFFER.empty()) { 
        std::uniform_int_distribution<int> r_dist(0, N_GRID_SIZE-1);
        return Pos{(int16_t)r_dist(RND_ENGINE), (int16_t)r_dist(RND_ENGINE)};
    }
    
    int K_select;
    if(use_sqrt_N_sampling){ 
        K_select = std::min((int)TMP_CELL_DIRT_INFOS_LIST_GLOBAL_BUFFER.size(), std::max(1, N_GRID_SIZE));
    } else { 
        K_select = std::min((int)TMP_CELL_DIRT_INFOS_LIST_GLOBAL_BUFFER.size(), std::max(10, N_GRID_SIZE * N_GRID_SIZE / 10));
    }

    if (K_select <= 0) { 
         std::uniform_int_distribution<int> r_dist(0, N_GRID_SIZE-1);
         return Pos{(int16_t)r_dist(RND_ENGINE), (int16_t)r_dist(RND_ENGINE)};
    }
    
    std::uniform_int_distribution<int> top_k_dist(0, K_select - 1);
    return TMP_CELL_DIRT_INFOS_LIST_GLOBAL_BUFFER[top_k_dist(RND_ENGINE)].p;
}

const int OP4_SAMPLE_POINTS = 20; 
const int OP5_MAX_SUBSEGMENT_LEN = 20; 

std::vector<char> GET_APSP_MOVES_BUFFER1;
std::vector<char> GET_APSP_MOVES_BUFFER2;


int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false); std::cin.tie(NULL);
    
    double time_limit_seconds = 1.95; 
    if (argc > 1) time_limit_seconds = std::stod(argv[1]);

    auto time_start_prog = std::chrono::high_resolution_clock::now();
    RND_ENGINE.seed(std::chrono::system_clock::now().time_since_epoch().count());

    std::cin >> N_GRID_SIZE;
    H_WALLS_INFO.resize(N_GRID_SIZE - 1); 
    V_WALLS_INFO.resize(N_GRID_SIZE);     

    for (int i = 0; i < N_GRID_SIZE - 1; ++i) std::cin >> H_WALLS_INFO[i];
    for (int i = 0; i < N_GRID_SIZE; ++i) std::cin >> V_WALLS_INFO[i]; 
    for (int i = 0; i < N_GRID_SIZE; ++i) for (int j = 0; j < N_GRID_SIZE; ++j) std::cin >> D_SUSC[i][j];
    
    MAX_L_PATH_HIGH_THRESHOLD_EFFECTIVE = MAX_L_PATH * 0.95; 
    MIN_L_PATH_LOW_THRESHOLD_EFFECTIVE = N_GRID_SIZE * N_GRID_SIZE; 

    compute_apsp();

    CELL_VISIT_TIMES_GLOBAL_BUFFER.resize(N_GRID_SIZE);
    TMP_CELL_DIRT_INFOS_LIST_GLOBAL_BUFFER.reserve(N_GRID_SIZE * N_GRID_SIZE);
    for(int r=0; r<N_GRID_SIZE; ++r) {
        CELL_VISIT_TIMES_GLOBAL_BUFFER[r].resize(N_GRID_SIZE);
        int reserve_size = std::max(2, MAX_L_PATH / (N_GRID_SIZE*N_GRID_SIZE) + 50);
         if (reserve_size > MAX_L_PATH / 2 && MAX_L_PATH > 2) reserve_size = MAX_L_PATH/2; // Cap reasonable max
        for(int c=0; c<N_GRID_SIZE; ++c) {
             CELL_VISIT_TIMES_GLOBAL_BUFFER[r][c].reserve(reserve_size);
        }
    }
    GET_APSP_MOVES_BUFFER1.reserve(N_GRID_SIZE*N_GRID_SIZE*2); 
    GET_APSP_MOVES_BUFFER2.reserve(N_GRID_SIZE*N_GRID_SIZE*2);


    PathData current_pd_obj;
    for(int i=0; i<N_GRID_SIZE; ++i) for(int j=0; j<N_GRID_SIZE; ++j) initial_dfs_visited[i][j] = false;
    generate_initial_dfs_path(0, 0, current_pd_obj);
    calculate_score_full(current_pd_obj); 

    PathData best_pd_obj = current_pd_obj; 

    double start_temp = 5000.0 * sqrt(N_GRID_SIZE); 
    double end_temp = 0.1;    
    
    int iterations_count = 0;
    PathData candidate_pd_obj; 
    std::uniform_real_distribution<double> accept_dist_01(0.0, 1.0);


    while(true) {
        iterations_count++;
        if(iterations_count % 100 == 0){ 
            auto now_time = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration<double>(now_time - time_start_prog).count();
            if (elapsed_seconds > time_limit_seconds) break;
        }
        
        int L_curr = current_pd_obj.moves.size();

        bool modified_successfully = false;
        for (int try_op = 0; try_op < 10; ++try_op) { 
            candidate_pd_obj = current_pd_obj; 
            
            std::uniform_int_distribution<int> op_dist(0, 99); 
            int op_choice_val = op_dist(RND_ENGINE);
            int operation_type = -1;

            if (op_choice_val < 15) operation_type = 0;      
            else if (op_choice_val < 30) operation_type = 1; 
            else if (op_choice_val < 60) operation_type = 2; 
            else if (op_choice_val < 70) operation_type = 3; 
            else if (op_choice_val < 85) operation_type = 4;                       
            else operation_type = 5;                         

            bool is_length_increasing_op = (operation_type == 0 || operation_type == 4); 
            // Op5 can increase or decrease length. Check its specific outcome for length control.
            bool is_length_decreasing_op = (operation_type == 1 || operation_type == 2); 

            if (is_length_increasing_op && L_curr > MAX_L_PATH_HIGH_THRESHOLD_EFFECTIVE) {
                if (accept_dist_01(RND_ENGINE) < 0.75) continue; 
            }
            if (is_length_decreasing_op && L_curr < MIN_L_PATH_LOW_THRESHOLD_EFFECTIVE) {
                 if (accept_dist_01(RND_ENGINE) < 0.75) continue; 
            }


            if (operation_type == 0) { 
                if (L_curr == 0 && N_GRID_SIZE > 1) continue; 
                if (candidate_pd_obj.moves.size() + 2 > MAX_L_PATH) continue; 
                if (current_pd_obj.coords.empty() && N_GRID_SIZE > 1) continue;

                std::uniform_int_distribution<int> k_idx_dist(0, L_curr); 
                int k_coord_idx = k_idx_dist(RND_ENGINE); 
                Pos p_k = current_pd_obj.coords[k_coord_idx]; 
                
                std::vector<int> possible_dirs; possible_dirs.reserve(4);
                for(int dir_i=0; dir_i<4; ++dir_i) {
                    Pos neighbor_p = Pos{(int16_t)(p_k.r + DR[dir_i]), (int16_t)(p_k.c + DC[dir_i])};
                    if (is_valid_pos(neighbor_p.r, neighbor_p.c) && !check_wall(p_k, neighbor_p)) {
                        possible_dirs.push_back(dir_i);
                    }
                }
                if (possible_dirs.empty()) continue;
                std::uniform_int_distribution<int> dir_choice_dist(0, possible_dirs.size()-1);
                int random_dir_idx = possible_dirs[dir_choice_dist(RND_ENGINE)];
                
                candidate_pd_obj.moves.insert(candidate_pd_obj.moves.begin() + k_coord_idx, 
                                             {DIR_CHARS[random_dir_idx], DIR_CHARS[(random_dir_idx+2)%4]});
            } else if (operation_type == 1) { 
                if (L_curr < 2) continue; 
                if (current_pd_obj.coords.size() < 3) continue; 

                std::vector<int> possible_indices; possible_indices.reserve(L_curr); 
                for(int k_m_idx = 0; k_m_idx <= L_curr - 2; ++k_m_idx) { 
                    if (current_pd_obj.coords[k_m_idx] == current_pd_obj.coords[k_m_idx+2]) {
                        possible_indices.push_back(k_m_idx);
                    }
                }
                if (possible_indices.empty()) continue;
                std::uniform_int_distribution<int> idx_choice_dist(0, possible_indices.size()-1);
                int k_move_idx_to_remove = possible_indices[idx_choice_dist(RND_ENGINE)];

                candidate_pd_obj.moves.erase(candidate_pd_obj.moves.begin() + k_move_idx_to_remove, 
                                             candidate_pd_obj.moves.begin() + k_move_idx_to_remove + 2);
            } else if (operation_type == 2) { 
                if (L_curr < 1) continue; 
                
                std::uniform_int_distribution<int> c_idx1_dist(0, L_curr > 0 ? L_curr - 1 : 0); 
                int c_idx1 = c_idx1_dist(RND_ENGINE); 
                std::uniform_int_distribution<int> c_idx2_dist(c_idx1 + 1, L_curr); 
                int c_idx2 = c_idx2_dist(RND_ENGINE);
                if (c_idx1 >= c_idx2 && L_curr > 0) continue; // Need valid subsegment
                if (L_curr == 0 && (c_idx1!=0 || c_idx2!=0)) continue; // L=0 means c_idx1=0, c_idx2=0 only
                                
                Pos p_A = current_pd_obj.coords[c_idx1]; Pos p_B = current_pd_obj.coords[c_idx2];
                
                if (!get_apsp_moves(p_A, p_B, GET_APSP_MOVES_BUFFER1)) continue;

                if (GET_APSP_MOVES_BUFFER1.size() >= (size_t)(c_idx2 - c_idx1) && L_curr > 0 ) continue; // APSP not shorter (allow if L_curr=0)

                if ( ( (long long)candidate_pd_obj.moves.size() - (c_idx2 - c_idx1) + GET_APSP_MOVES_BUFFER1.size()) > MAX_L_PATH) continue;
                
                if (c_idx1 < c_idx2) { // Ensure erase range is valid
                    candidate_pd_obj.moves.erase(candidate_pd_obj.moves.begin() + c_idx1, 
                                                candidate_pd_obj.moves.begin() + c_idx2);
                }
                candidate_pd_obj.moves.insert(candidate_pd_obj.moves.begin() + c_idx1, 
                                              GET_APSP_MOVES_BUFFER1.begin(), GET_APSP_MOVES_BUFFER1.end());
            } else if (operation_type == 3) { 
                if (L_curr < 1) continue; 
                
                std::uniform_int_distribution<int> move_idx_dist(0, L_curr -1);
                int move_idx1 = move_idx_dist(RND_ENGINE); 
                std::uniform_int_distribution<int> move_idx_dist2(move_idx1, L_curr -1); 
                int move_idx2_inclusive = move_idx_dist2(RND_ENGINE);
                
                int move_idx2_exclusive = move_idx2_inclusive + 1;
                
                std::reverse(candidate_pd_obj.moves.begin() + move_idx1, candidate_pd_obj.moves.begin() + move_idx2_exclusive);
                for(int i = move_idx1; i < move_idx2_exclusive; ++i) 
                    candidate_pd_obj.moves[i] = invert_move(candidate_pd_obj.moves[i]);

            } else if (operation_type == 4) { 
                if (L_curr == 0 && N_GRID_SIZE > 1) continue; 
                if (current_pd_obj.coords.empty() && N_GRID_SIZE > 1) continue; 
                
                Pos target_cell = select_target_cell_for_dirt_ops(current_pd_obj, false);
                
                int best_k_coord_idx = -1;
                long long min_detour_len_increase = (long long)MAX_L_PATH * 2 +1; // path len increase: 2 for wiggle, 2*dist for detour

                if (L_curr >= 0) { // Path can be empty (L_curr=0), then coords has 1 element (0,0)
                    int num_samples = (L_curr == 0) ? 1: OP4_SAMPLE_POINTS; // If L_curr=0, only one point to pick: coords[0]
                    for (int i=0; i < num_samples; ++i) {
                        std::uniform_int_distribution<int> k_idx_dist(0, L_curr); 
                        int k_coord_idx_sample = (L_curr == 0) ? 0 : k_idx_dist(RND_ENGINE);
                        Pos p_A_sample = current_pd_obj.coords[k_coord_idx_sample];
                        
                        long long current_detour_increase;
                        if (p_A_sample == target_cell) {
                             current_detour_increase = 2; // Wiggle cost
                        } else {
                            int dist_pa_target = APSP_DIST[p_A_sample.r][p_A_sample.c][target_cell.r][target_cell.c];
                            if (dist_pa_target != -1) {
                                current_detour_increase = (long long)dist_pa_target * 2;
                            } else {
                                current_detour_increase = (long long)MAX_L_PATH * 2 + 1; // effectively infinity
                            }
                        }
                        if (current_detour_increase < min_detour_len_increase) {
                            min_detour_len_increase = current_detour_increase;
                            best_k_coord_idx = k_coord_idx_sample;
                        }
                    }
                }


                if (best_k_coord_idx == -1 || min_detour_len_increase > MAX_L_PATH) continue; 
                
                Pos p_A = current_pd_obj.coords[best_k_coord_idx];

                if (candidate_pd_obj.moves.size() + min_detour_len_increase > MAX_L_PATH) continue;

                if (p_A == target_cell) { 
                     std::vector<int> possible_dirs; possible_dirs.reserve(4);
                     for(int dir_i=0; dir_i<4; ++dir_i) {
                         Pos neighbor_p = Pos{(int16_t)(p_A.r + DR[dir_i]), (int16_t)(p_A.c + DC[dir_i])};
                         if (is_valid_pos(neighbor_p.r, neighbor_p.c) && !check_wall(p_A, neighbor_p)) {
                             possible_dirs.push_back(dir_i);
                         }
                     }
                     if (possible_dirs.empty()) continue;
                     std::uniform_int_distribution<int> dir_choice_dist(0, possible_dirs.size()-1);
                     int random_dir_idx = possible_dirs[dir_choice_dist(RND_ENGINE)];

                     candidate_pd_obj.moves.insert(candidate_pd_obj.moves.begin() + best_k_coord_idx, 
                                                  {DIR_CHARS[random_dir_idx], DIR_CHARS[(random_dir_idx+2)%4]});
                } else { 
                    if (!get_apsp_moves(p_A, target_cell, GET_APSP_MOVES_BUFFER1)) continue;
                    if (!get_apsp_moves(target_cell, p_A, GET_APSP_MOVES_BUFFER2)) continue; 

                    candidate_pd_obj.moves.insert(candidate_pd_obj.moves.begin() + best_k_coord_idx, 
                                                  GET_APSP_MOVES_BUFFER2.begin(), GET_APSP_MOVES_BUFFER2.end()); 
                    candidate_pd_obj.moves.insert(candidate_pd_obj.moves.begin() + best_k_coord_idx, 
                                                  GET_APSP_MOVES_BUFFER1.begin(), GET_APSP_MOVES_BUFFER1.end()); 
                }
            } else { // operation_type == 5:
                Pos target_cell = select_target_cell_for_dirt_ops(current_pd_obj, true); 

                int c_idx1, c_idx2;
                if (L_curr == 0) {
                    c_idx1 = 0; c_idx2 = 0;
                } else {
                    std::uniform_int_distribution<int> c_idx1_dist_op5(0, L_curr -1 ); 
                    c_idx1 = c_idx1_dist_op5(RND_ENGINE);
                    std::uniform_int_distribution<int> c_idx2_dist_op5(c_idx1 + 1, std::min(L_curr, c_idx1 + OP5_MAX_SUBSEGMENT_LEN));
                    c_idx2 = c_idx2_dist_op5(RND_ENGINE);
                }
                if (c_idx1 > c_idx2) continue; // Should not happen with above logic for L_curr > 0

                Pos p_A = current_pd_obj.coords[c_idx1]; 
                Pos p_B = current_pd_obj.coords[c_idx2];
                                
                if (!get_apsp_moves(p_A, target_cell, GET_APSP_MOVES_BUFFER1)) continue;
                if (!get_apsp_moves(target_cell, p_B, GET_APSP_MOVES_BUFFER2)) continue;

                long long current_subsegment_len_moves = c_idx2 - c_idx1;
                long long new_subsegment_len_moves = GET_APSP_MOVES_BUFFER1.size() + GET_APSP_MOVES_BUFFER2.size();

                // Specific length control for Op5
                if (new_subsegment_len_moves > current_subsegment_len_moves && L_curr > MAX_L_PATH_HIGH_THRESHOLD_EFFECTIVE) {
                     if (accept_dist_01(RND_ENGINE) < 0.75) continue;
                }
                if (new_subsegment_len_moves < current_subsegment_len_moves && L_curr < MIN_L_PATH_LOW_THRESHOLD_EFFECTIVE) {
                     if (accept_dist_01(RND_ENGINE) < 0.75) continue;
                }


                if ( ( (long long)candidate_pd_obj.moves.size() - current_subsegment_len_moves + new_subsegment_len_moves) > MAX_L_PATH) continue;
                
                if (c_idx1 < c_idx2) { 
                    candidate_pd_obj.moves.erase(candidate_pd_obj.moves.begin() + c_idx1, 
                                                candidate_pd_obj.moves.begin() + c_idx2);
                }
                candidate_pd_obj.moves.insert(candidate_pd_obj.moves.begin() + c_idx1, 
                                            GET_APSP_MOVES_BUFFER2.begin(), GET_APSP_MOVES_BUFFER2.end());
                candidate_pd_obj.moves.insert(candidate_pd_obj.moves.begin() + c_idx1, 
                                            GET_APSP_MOVES_BUFFER1.begin(), GET_APSP_MOVES_BUFFER1.end());
            }

            modified_successfully = true; 
            break; 
        }

        if (!modified_successfully) continue;
        
        calculate_score_full(candidate_pd_obj); 
        
        bool candidate_is_invalid = candidate_pd_obj.score_val > 1e17L; 

        if (candidate_pd_obj.score_val < current_pd_obj.score_val) {
            current_pd_obj = std::move(candidate_pd_obj); 
            if (current_pd_obj.score_val < best_pd_obj.score_val) {
                best_pd_obj = current_pd_obj; 
            }
        } else if (!candidate_is_invalid) { 
             auto now_time_temp = std::chrono::high_resolution_clock::now();
             double elapsed_seconds_temp = std::chrono::duration<double>(now_time_temp - time_start_prog).count();
             double progress_ratio = elapsed_seconds_temp / time_limit_seconds; 
             progress_ratio = std::min(1.0, std::max(0.0, progress_ratio)); 
             
             double current_temp_val = end_temp; 
             if (start_temp > end_temp + 1e-9) { 
                current_temp_val = start_temp * std::pow(end_temp / start_temp, progress_ratio);
             } else if (start_temp > 1e-9) { 
                current_temp_val = start_temp;
             } else { 
                 current_temp_val = end_temp; 
             }
             if (current_temp_val < 1e-9 && end_temp >= 1e-9) current_temp_val = end_temp; 
             else if (current_temp_val < 1e-9) current_temp_val = 1e-9; 
             
             if (exp((current_pd_obj.score_val - candidate_pd_obj.score_val) / current_temp_val) > accept_dist_01(RND_ENGINE)) {
                 current_pd_obj = std::move(candidate_pd_obj); 
             }
        }
    }

    for (char move_char : best_pd_obj.moves) std::cout << move_char;
    std::cout << std::endl;
    
    return 0;
}
# EVOLVE-BLOCK-END