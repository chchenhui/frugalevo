# EVOLVE-BLOCK-START
#pragma GCC optimize("O3,unroll-loops")

#include <iostream>
#include <vector>
#include <map>       // For temp_adj_deltas_map_global
#include <queue>
#include <algorithm> // For std::min, std::max, std::sort, std::unique, std::shuffle
#include <random>    // For XorShift and std::shuffle
#include <chrono>
#include <utility>   // For std::pair
#include <cmath>     // For std::exp, std::pow
#include <climits>   // For UINT_MAX

// --- Globals ---
const int N_FIXED = 50;
const int M_FIXED = 100; // Max ward ID, problem states M=100

std::vector<std::vector<int>> current_grid_state(N_FIXED, std::vector<int>(N_FIXED));
std::vector<std::vector<int>> best_grid_state(N_FIXED, std::vector<int>(N_FIXED));
int best_score_val = -1; // Stores count of 0-cells for the best state

struct XorShift {
    unsigned int x, y, z, w;
    XorShift() { 
        // Using std::random_device for better seed initialization
        std::random_device rd;
        x = rd();
        y = rd();
        z = rd();
        w = rd();
        // Ensure no zero initial state for w, which is common if rd() produces same values or all are 0
        if (x == 0 && y == 0 && z == 0 && w == 0) w = 1; // Or any non-zero value
    }
    unsigned int next_uint() {
        unsigned int t = x;
        t ^= t << 11;
        t ^= t >> 8;
        x = y; y = z; z = w;
        w ^= w >> 19;
        w ^= t;
        return w;
    }
    double next_double() { // In [0,1)
        return (double)next_uint() / ((double)UINT_MAX + 1.0);
    }
    int next_int(int exclusive_max_val) { // In [0, exclusive_max_val - 1]
        if (exclusive_max_val <= 0) return 0; 
        return next_uint() % exclusive_max_val;
    }
    // For std::shuffle
    using result_type = unsigned int;
    static constexpr unsigned int min() { return 0; }
    static constexpr unsigned int max() { return UINT_MAX; }
    unsigned int operator()() { return next_uint(); }
};
XorShift rnd_gen; // Global instance
auto G_START_TIME = std::chrono::high_resolution_clock::now();

double time_elapsed_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - G_START_TIME).count();
}

struct AdjacencyInfo {
    bool matrix[M_FIXED + 1][M_FIXED + 1];
    AdjacencyInfo() {
        for (int i = 0; i <= M_FIXED; ++i) for (int j = 0; j <= M_FIXED; ++j) matrix[i][j] = false;
    }
    void set_adj(int c1, int c2) {
        if (c1 == c2) return;
        matrix[std::min(c1, c2)][std::max(c1, c2)] = true;
    }
    bool is_adj(int c1, int c2) const {
        if (c1 == c2) return false; 
        return matrix[std::min(c1, c2)][std::max(c1, c2)];
    }
};
AdjacencyInfo required_adjacencies; 
bool ward_has_any_req_adj[M_FIXED + 1]; 

struct BorderEdgeTracker {
    int counts_arr[M_FIXED + 1][M_FIXED + 1];
    BorderEdgeTracker() { clear(); }
    void add_edge(int c1, int c2) {
        if (c1 == c2) return;
        counts_arr[std::min(c1, c2)][std::max(c1, c2)]++;
    }
    void remove_edge(int c1, int c2) {
        if (c1 == c2) return;
        counts_arr[std::min(c1, c2)][std::max(c1, c2)]--;
    }
    int get_count(int c1, int c2) const {
        if (c1 == c2) return 0;
        return counts_arr[std::min(c1, c2)][std::max(c1, c2)];
    }
    void clear() {
        for (int i = 0; i <= M_FIXED; ++i) for (int j = 0; j <= M_FIXED; ++j) counts_arr[i][j] = 0;
    }
};
BorderEdgeTracker current_border_edges_tracker; 

std::vector<std::vector<std::pair<int, int>>> cells_by_color(M_FIXED + 1);
std::vector<std::vector<int>> pos_in_color_list(N_FIXED, std::vector<int>(N_FIXED));

unsigned int visited_marker_grid[N_FIXED][N_FIXED]; 
unsigned int current_visit_marker = 0; 

std::queue<std::pair<int, int>> q_bfs_global; 

const int DR[] = {-1, 1, 0, 0}; 
const int DC[] = {0, 0, -1, 1};

inline bool is_cell_on_grid(int r, int c) { return r >= 0 && r < N_FIXED && c >= 0 && c < N_FIXED; }

void increment_bfs_marker() {
    current_visit_marker++;
    if (current_visit_marker == 0) { 
        for (int i = 0; i < N_FIXED; ++i) {
            for (int j = 0; j < N_FIXED; ++j) {
                visited_marker_grid[i][j] = 0;
            }
        }
        current_visit_marker = 1; 
    }
}

void clear_global_bfs_queue() {
    std::queue<std::pair<int, int>> empty_queue;
    std::swap(q_bfs_global, empty_queue);
}

void add_cell_to_color_ds(int r, int c, int color) {
    cells_by_color[color].push_back({r,c});
    pos_in_color_list[r][c] = cells_by_color[color].size() - 1;
}

void remove_cell_from_color_ds(int r, int c, int color) {
    int idx_to_remove = pos_in_color_list[r][c];
    std::pair<int,int> last_cell = cells_by_color[color].back();
    
    cells_by_color[color][idx_to_remove] = last_cell;
    pos_in_color_list[last_cell.first][last_cell.second] = idx_to_remove;
    
    cells_by_color[color].pop_back();
}

void initialize_all_data_structures(const std::vector<std::vector<int>>& initial_grid) {
    required_adjacencies = AdjacencyInfo(); 
    current_border_edges_tracker.clear();
    for(int i=0; i <= M_FIXED; ++i) cells_by_color[i].clear();

    for (int i = 0; i < N_FIXED; ++i) {
        for (int j = 0; j < N_FIXED; ++j) {
            current_grid_state[i][j] = initial_grid[i][j]; 
            add_cell_to_color_ds(i, j, initial_grid[i][j]);
        }
    }
    
    for (int i = 0; i < N_FIXED; ++i) {
        for (int j = 0; j < N_FIXED; ++j) {
            int initial_color_val = initial_grid[i][j];
            if (i == 0 || i == N_FIXED - 1 || j == 0 || j == N_FIXED - 1) {
                required_adjacencies.set_adj(0, initial_color_val);
            }
            if (j + 1 < N_FIXED && initial_color_val != initial_grid[i][j+1]) {
                required_adjacencies.set_adj(initial_color_val, initial_grid[i][j+1]);
            }
            if (i + 1 < N_FIXED && initial_color_val != initial_grid[i+1][j]) {
                required_adjacencies.set_adj(initial_color_val, initial_grid[i+1][j]);
            }

            int current_color_val = current_grid_state[i][j]; 
            if (i == 0) current_border_edges_tracker.add_edge(0, current_color_val);
            if (i == N_FIXED - 1) current_border_edges_tracker.add_edge(0, current_color_val);
            if (j == 0) current_border_edges_tracker.add_edge(0, current_color_val);
            if (j == N_FIXED - 1) current_border_edges_tracker.add_edge(0, current_color_val);
            
            if (j + 1 < N_FIXED && current_color_val != current_grid_state[i][j+1]) {
                current_border_edges_tracker.add_edge(current_color_val, current_grid_state[i][j+1]);
            }
            if (i + 1 < N_FIXED && current_color_val != current_grid_state[i+1][j]) {
                current_border_edges_tracker.add_edge(current_color_val, current_grid_state[i+1][j]);
            }
        }
    }

    for (int c1 = 0; c1 <= M_FIXED; ++c1) {
        ward_has_any_req_adj[c1] = false; 
        for (int c2 = 0; c2 <= M_FIXED; ++c2) {
            if (c1 == c2) continue;
            if (required_adjacencies.is_adj(c1, c2)) {
                ward_has_any_req_adj[c1] = true;
                break;
            }
        }
    }

    best_grid_state = current_grid_state;
    best_score_val = cells_by_color[0].size();
}

bool check_region_connectivity_bfs(int target_color) {
    const auto& cells_of_target_color = cells_by_color[target_color]; 
    if (cells_of_target_color.empty()) return true; 
    
    increment_bfs_marker();
    clear_global_bfs_queue();

    q_bfs_global.push(cells_of_target_color[0]); 
    visited_marker_grid[cells_of_target_color[0].first][cells_of_target_color[0].second] = current_visit_marker;
    
    int count_visited_cells = 0;
    while (!q_bfs_global.empty()) {
        std::pair<int, int> curr = q_bfs_global.front();
        q_bfs_global.pop();
        count_visited_cells++;

        for (int k = 0; k < 4; ++k) {
            int nr = curr.first + DR[k];
            int nc = curr.second + DC[k];
            if (is_cell_on_grid(nr, nc) && 
                current_grid_state[nr][nc] == target_color && 
                visited_marker_grid[nr][nc] != current_visit_marker) {
                visited_marker_grid[nr][nc] = current_visit_marker;
                q_bfs_global.push({nr, nc});
            }
        }
    }
    return count_visited_cells == cells_of_target_color.size();
}

bool check_region_0_connectivity_full() {
    const auto& cells_c0 = cells_by_color[0];
    if (cells_c0.empty()) {
        return true; 
    }

    increment_bfs_marker();
    clear_global_bfs_queue();

    bool any_boundary_zero_cell_found = false;
    for (const auto& cell_coord : cells_c0) {
        int r = cell_coord.first;
        int c = cell_coord.second;
        if (r == 0 || r == N_FIXED - 1 || c == 0 || c == N_FIXED - 1) {
            if (visited_marker_grid[r][c] != current_visit_marker) { 
                 q_bfs_global.push(cell_coord);
                 visited_marker_grid[r][c] = current_visit_marker;
            }
            any_boundary_zero_cell_found = true;
        }
    }

    if (!any_boundary_zero_cell_found) {
        return false;
    }

    while (!q_bfs_global.empty()) {
        std::pair<int, int> curr = q_bfs_global.front();
        q_bfs_global.pop();

        for (int k_dir = 0; k_dir < 4; ++k_dir) {
            int nr = curr.first + DR[k_dir];
            int nc = curr.second + DC[k_dir];
            if (is_cell_on_grid(nr, nc) &&
                current_grid_state[nr][nc] == 0 && 
                visited_marker_grid[nr][nc] != current_visit_marker) { 
                visited_marker_grid[nr][nc] = current_visit_marker;
                q_bfs_global.push({nr, nc});
            }
        }
    }

    for (const auto& cell_coord : cells_c0) {
        if (visited_marker_grid[cell_coord.first][cell_coord.second] != current_visit_marker) {
            return false; 
        }
    }
    return true;
}

std::map<std::pair<int, int>, int> temp_adj_deltas_map_global;

bool attempt_change_cell_color_and_validate(int r, int c, int old_color, int new_color) {
    current_grid_state[r][c] = new_color;
    remove_cell_from_color_ds(r, c, old_color); 
    add_cell_to_color_ds(r, c, new_color);

    temp_adj_deltas_map_global.clear();
    for (int k_adj=0; k_adj<4; ++k_adj) {
        int nr = r + DR[k_adj];
        int nc = c + DC[k_adj];
        int neighbor_actual_color = is_cell_on_grid(nr,nc) ? current_grid_state[nr][nc] : 0; 
        
        if (old_color != neighbor_actual_color) { 
             temp_adj_deltas_map_global[{std::min(old_color, neighbor_actual_color), std::max(old_color, neighbor_actual_color)}]--;
        }
        if (new_color != neighbor_actual_color) { 
             temp_adj_deltas_map_global[{std::min(new_color, neighbor_actual_color), std::max(new_color, neighbor_actual_color)}]++;
        }
    }
    for(const auto& entry : temp_adj_deltas_map_global) {
        int c1 = entry.first.first; int c2 = entry.first.second; int delta = entry.second;
        if (delta > 0) for(int i=0; i<delta; ++i) current_border_edges_tracker.add_edge(c1,c2);
        else for(int i=0; i<-delta; ++i) current_border_edges_tracker.remove_edge(c1,c2);
    }

    bool is_change_valid = true;

    for(const auto& entry : temp_adj_deltas_map_global) {
        int c1 = entry.first.first; int c2 = entry.first.second;
        bool has_edge_now = current_border_edges_tracker.get_count(c1, c2) > 0;
        bool needs_edge = required_adjacencies.is_adj(c1, c2);
        if (has_edge_now != needs_edge) {
            is_change_valid = false; break;
        }
    }
    
    if (is_change_valid && old_color != 0 && cells_by_color[old_color].empty() && ward_has_any_req_adj[old_color]) {
        is_change_valid = false;
    }
    
    if (is_change_valid && old_color != 0 && !cells_by_color[old_color].empty()) {
        if (!check_region_connectivity_bfs(old_color)) is_change_valid = false;
    }

    if (is_change_valid && new_color != 0) { 
         if (!check_region_connectivity_bfs(new_color)) is_change_valid = false;
    }
    
    if (is_change_valid && (old_color == 0 || new_color == 0)) { 
        if (!cells_by_color[0].empty()) { 
            if (!check_region_0_connectivity_full()) is_change_valid = false;
        } else { 
            if (ward_has_any_req_adj[0]) { 
                 is_change_valid = false;
            }
        }
    }

    if (!is_change_valid) { 
        current_grid_state[r][c] = old_color; 
        remove_cell_from_color_ds(r, c, new_color); 
        add_cell_to_color_ds(r, c, old_color);      
        
        for(const auto& entry : temp_adj_deltas_map_global) {
            int c1_ = entry.first.first; int c2_ = entry.first.second; int delta = entry.second;
            if (delta > 0) for(int i=0; i<delta; ++i) current_border_edges_tracker.remove_edge(c1_,c2_);
            else for(int i=0; i<-delta; ++i) current_border_edges_tracker.add_edge(c1_,c2_);
        }
        return false; 
    }
    return true; 
}

/**
 * Rebuild all mutable bookkeeping from a candidate grid while preserving the
 * original adjacency graph, then accept the candidate only if every color is
 * connected and every required adjacency is represented exactly.
 */
bool rebuild_and_validate_candidate(
    const std::vector<std::vector<int>>& candidate,
    const AdjacencyInfo& original_required) {
    initialize_all_data_structures(candidate);
    required_adjacencies = original_required;

    for (int c1 = 0; c1 <= M_FIXED; ++c1) {
        ward_has_any_req_adj[c1] = false;
        for (int c2 = 0; c2 <= M_FIXED; ++c2) {
            if (c1 != c2 && required_adjacencies.is_adj(c1, c2)) {
                ward_has_any_req_adj[c1] = true;
                break;
            }
        }
    }

    for (int c = 1; c <= M_FIXED; ++c) {
        if (!check_region_connectivity_bfs(c)) return false;
    }
    if (!check_region_0_connectivity_full()) return false;

    for (int c1 = 0; c1 <= M_FIXED; ++c1) {
        for (int c2 = c1 + 1; c2 <= M_FIXED; ++c2) {
            bool present = current_border_edges_tracker.get_count(c1, c2) > 0;
            if (present != required_adjacencies.is_adj(c1, c2)) return false;
        }
    }
    return true;
}

/**
 * Relocate several low-degree wards. Each ward is reduced to the union of
 * shortest paths inside its old region joining witnesses for all required
 * neighboring colors and the exterior, while all unused old cells become zero.
 */
void whole_region_relocation_phase() {
    const double phase_start = time_elapsed_ms();
    const double phase_limit = 350.0;
    const AdjacencyInfo original_required = required_adjacencies;

    std::vector<int> order;
    for (int c = 1; c <= M_FIXED; ++c) {
        int degree = 0;
        for (int d = 0; d <= M_FIXED; ++d)
            if (c != d && required_adjacencies.is_adj(c, d)) ++degree;
        if (!cells_by_color[c].empty()) order.push_back(c);
    }

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int da = 0, db = 0;
        for (int d = 0; d <= M_FIXED; ++d) {
            if (a != d && required_adjacencies.is_adj(a, d)) ++da;
            if (b != d && required_adjacencies.is_adj(b, d)) ++db;
        }
        if (da != db) return da < db;
        return cells_by_color[a].size() > cells_by_color[b].size();
    });

    int attempts = 0;
    for (int color : order) {
        if (attempts++ >= 8 || time_elapsed_ms() - phase_start > phase_limit) break;
        if (cells_by_color[color].size() <= 2) continue;

        std::vector<std::vector<int>> candidate = current_grid_state;
        std::vector<std::pair<int,int>> region = cells_by_color[color];

        std::vector<std::pair<int,int>> seeds;
        for (int required_color = 0; required_color <= M_FIXED; ++required_color) {
            if (required_color == color ||
                !required_adjacencies.is_adj(color, required_color)) continue;

            std::pair<int,int> chosen = {-1, -1};
            for (auto p : region) {
                int r = p.first, c = p.second;
                bool witness = false;
                for (int k = 0; k < 4; ++k) {
                    int nr = r + DR[k], nc = c + DC[k];
                    int nc_color = is_cell_on_grid(nr, nc)
                        ? current_grid_state[nr][nc] : 0;
                    if (nc_color == required_color) {
                        witness = true;
                        break;
                    }
                }
                if (witness) {
                    chosen = p;
                    break;
                }
            }
            if (chosen.first >= 0) seeds.push_back(chosen);
        }

        if (seeds.empty()) continue;

        std::vector<std::vector<unsigned char>> keep(N_FIXED,
            std::vector<unsigned char>(N_FIXED, 0));
        std::vector<std::vector<unsigned char>> used(N_FIXED,
            std::vector<unsigned char>(N_FIXED, 0));

        for (auto source : seeds) {
            // Recompute reachability independently for every terminal so
            // previously constructed paths do not block later BFS searches.
            std::queue<std::pair<int,int>> q;
            std::vector<std::vector<unsigned char>> seen(
                N_FIXED, std::vector<unsigned char>(N_FIXED, 0));
            std::vector<std::vector<std::pair<int,int>>> parent(
                N_FIXED, std::vector<std::pair<int,int>>(
                    N_FIXED, std::make_pair(-1, -1)));

            const std::pair<int,int> target = seeds[0];
            q.push(source);
            seen[source.first][source.second] = 1;

            while (!q.empty()) {
                auto p = q.front();
                q.pop();
                if (p == target) break;

                for (int k = 0; k < 4; ++k) {
                    int nr = p.first + DR[k];
                    int nc = p.second + DC[k];
                    if (!is_cell_on_grid(nr, nc) || seen[nr][nc]) continue;
                    if (current_grid_state[nr][nc] != color) continue;

                    seen[nr][nc] = 1;
                    parent[nr][nc] = p;
                    q.push({nr, nc});
                }
            }

            if (!seen[target.first][target.second]) continue;

            auto p = target;
            while (true) {
                keep[p.first][p.second] = 1;
                if (p == source) break;
                p = parent[p.first][p.second];
                if (p.first < 0) break;
            }
        }

        for (auto p : region) {
            if (keep[p.first][p.second]) candidate[p.first][p.second] = color;
            else candidate[p.first][p.second] = 0;
        }

        std::vector<std::vector<int>> backup = current_grid_state;
        int backup_score = best_score_val;

        if (rebuild_and_validate_candidate(candidate, original_required)) {
            int score = static_cast<int>(cells_by_color[0].size());
            if (score > best_score_val) {
                best_score_val = score;
                best_grid_state = current_grid_state;
            }
        } else {
            rebuild_and_validate_candidate(backup, original_required);
            current_grid_state = backup;
            best_score_val = backup_score;
        }
    }
}

/**
 * Carve short zero channels through thin colored dams, then restore connectivity
 * with shortest compensating paths inside the original ward footprint.
 */
void zero_channel_tunneling_phase() {
    const double phase_start = time_elapsed_ms();
    const double phase_limit = 320.0;
    const AdjacencyInfo original_required = required_adjacencies;

    auto inside = [](int r, int c) {
        return r >= 0 && r < N_FIXED && c >= 0 && c < N_FIXED;
    };

    for (int color = 1; color <= M_FIXED; ++color) {
        if (time_elapsed_ms() - phase_start > phase_limit) break;
        if (cells_by_color[color].size() < 4) continue;

        std::vector<std::pair<int,int>> frontier;
        for (auto p : cells_by_color[color]) {
            bool exposed = false;
            for (int k = 0; k < 4; ++k) {
                int nr = p.first + DR[k], nc = p.second + DC[k];
                int qcolor = inside(nr, nc) ? current_grid_state[nr][nc] : 0;
                if (qcolor == 0) {
                    exposed = true;
                    break;
                }
            }
            if (exposed) frontier.push_back(p);
        }

        if (frontier.size() < 2) continue;

        int attempts = 0;
        for (int si = 0; si < (int)frontier.size() && attempts < 24; ++si) {
            if (time_elapsed_ms() - phase_start > phase_limit) break;

            std::vector<int> dist(N_FIXED * N_FIXED, -1);
            std::vector<int> parent(N_FIXED * N_FIXED, -1);
            std::queue<std::pair<int,int>> q;

            auto id = [](int r, int c) {
                return r * N_FIXED + c;
            };

            auto source = frontier[si];
            q.push(source);
            dist[id(source.first, source.second)] = 0;

            std::pair<int,int> target = {-1, -1};
            while (!q.empty()) {
                auto p = q.front();
                q.pop();

                int d = dist[id(p.first, p.second)];
                if (d >= 18) continue;

                for (int k = 0; k < 4; ++k) {
                    int nr = p.first + DR[k], nc = p.second + DC[k];
                    if (!inside(nr, nc)) continue;
                    if (current_grid_state[nr][nc] != color) continue;
                    int ni = id(nr, nc);
                    if (dist[ni] != -1) continue;

                    dist[ni] = d + 1;
                    parent[ni] = id(p.first, p.second);
                    q.push({nr, nc});

                    if (dist[ni] >= 2) {
                        bool is_frontier = false;
                        for (auto f : frontier) {
                            if (f.first == nr && f.second == nc) {
                                is_frontier = true;
                                break;
                            }
                        }
                        if (is_frontier) {
                            target = {nr, nc};
                            while (!q.empty()) q.pop();
                            break;
                        }
                    }
                }
                if (target.first >= 0) break;
            }

            if (target.first < 0) continue;

            std::vector<std::pair<int,int>> channel;
            int cur = id(target.first, target.second);
            while (cur != -1) {
                channel.push_back({cur / N_FIXED, cur % N_FIXED});
                if (cur == id(source.first, source.second)) break;
                cur = parent[cur];
            }
            if (channel.size() < 3 || channel.size() > 19) continue;

            std::vector<std::vector<unsigned char>> blocked(
                N_FIXED, std::vector<unsigned char>(N_FIXED, 0));
            for (auto p : channel) blocked[p.first][p.second] = 1;

            std::vector<std::vector<int>> candidate = current_grid_state;
            for (auto p : channel) candidate[p.first][p.second] = 0;

            /*
             * Identify components of the old ward after removing the proposed
             * channel, then reconnect every component to the first component
             * through old ward cells while never using channel cells.
             */
            std::vector<std::vector<unsigned char>> seen(
                N_FIXED, std::vector<unsigned char>(N_FIXED, 0));
            std::vector<std::pair<int,int>> representatives;

            for (auto start : cells_by_color[color]) {
                if (blocked[start.first][start.second] ||
                    seen[start.first][start.second]) continue;

                representatives.push_back(start);
                std::queue<std::pair<int,int>> cq;
                cq.push(start);
                seen[start.first][start.second] = 1;

                while (!cq.empty()) {
                    auto p = cq.front();
                    cq.pop();
                    for (int k = 0; k < 4; ++k) {
                        int nr = p.first + DR[k], nc = p.second + DC[k];
                        if (!inside(nr, nc) || blocked[nr][nc] ||
                            seen[nr][nc] ||
                            current_grid_state[nr][nc] != color) continue;
                        seen[nr][nc] = 1;
                        cq.push({nr, nc});
                    }
                }
            }

            if (representatives.empty()) continue;

            bool bridge_failed = false;
            auto root = representatives[0];

            for (int ri = 1; ri < (int)representatives.size(); ++ri) {
                std::vector<int> par(N_FIXED * N_FIXED, -1);
                std::queue<std::pair<int,int>> bq;
                bq.push(root);
                par[id(root.first, root.second)] =
                    id(root.first, root.second);

                while (!bq.empty() &&
                       par[id(representatives[ri].first,
                              representatives[ri].second)] == -1) {
                    auto p = bq.front();
                    bq.pop();

                    for (int k = 0; k < 4; ++k) {
                        int nr = p.first + DR[k], nc = p.second + DC[k];
                        if (!inside(nr, nc) || blocked[nr][nc] ||
                            current_grid_state[nr][nc] != color) continue;

                        int ni = id(nr, nc);
                        if (par[ni] != -1) continue;
                        par[ni] = id(p.first, p.second);
                        bq.push({nr, nc});
                    }
                }

                int finish = id(representatives[ri].first,
                                representatives[ri].second);
                if (par[finish] == -1) {
                    bridge_failed = true;
                    break;
                }

                while (true) {
                    int rr = finish / N_FIXED;
                    int cc = finish % N_FIXED;
                    candidate[rr][cc] = color;
                    if (finish == id(root.first, root.second)) break;
                    finish = par[finish];
                }
            }

            if (bridge_failed) continue;
            ++attempts;

            std::vector<std::vector<int>> backup = current_grid_state;
            int backup_score = best_score_val;

            if (rebuild_and_validate_candidate(candidate, original_required)) {
                int score = static_cast<int>(cells_by_color[0].size());
                if (score > best_score_val) {
                    best_score_val = score;
                    best_grid_state = current_grid_state;
                }
            } else {
                rebuild_and_validate_candidate(backup, original_required);
                current_grid_state = backup;
                best_score_val = backup_score;
            }
        }
    }
}

void solve_main_logic() {
    std::vector<std::vector<int>> initial_grid_from_input(N_FIXED, std::vector<int>(N_FIXED));
    for (int i = 0; i < N_FIXED; ++i) for (int j = 0; j < N_FIXED; ++j) std::cin >> initial_grid_from_input[i][j];
    
    initialize_all_data_structures(initial_grid_from_input);

    whole_region_relocation_phase();
    zero_channel_tunneling_phase();
    
    double sa_start_temp = 2.0;
    double sa_end_temp = 0.01; 
    const double TOTAL_COMPUTATION_TIME_MS = 1950.0; 
    
    double sa_start_abs_time = time_elapsed_ms();
    double sa_total_duration_ms = TOTAL_COMPUTATION_TIME_MS - sa_start_abs_time;
    if (sa_total_duration_ms <= 0) sa_total_duration_ms = 1.0; 
    
    int iter_count = 0;
    while(true) {
        iter_count++;
        if(iter_count % 256 == 0) { 
             if (time_elapsed_ms() >= TOTAL_COMPUTATION_TIME_MS) break;
        }

        double time_spent_in_sa = time_elapsed_ms() - sa_start_abs_time;
        double progress_ratio = (sa_total_duration_ms > 1e-9) ? (time_spent_in_sa / sa_total_duration_ms) : 1.0;
        progress_ratio = std::min(progress_ratio, 1.0); 
        
        double current_temperature = sa_start_temp * std::pow(sa_end_temp / sa_start_temp, progress_ratio);
        current_temperature = std::max(current_temperature, sa_end_temp); 
        
        int r_coord = rnd_gen.next_int(N_FIXED);
        int c_coord = rnd_gen.next_int(N_FIXED);
        int original_color_at_cell = current_grid_state[r_coord][c_coord];
        
        int candidate_new_colors[5]; 
        int num_candidate_options = 0;
        candidate_new_colors[num_candidate_options++] = 0; 
        for(int k_neighbor_idx=0; k_neighbor_idx<4; ++k_neighbor_idx) {
            int nr = r_coord + DR[k_neighbor_idx];
            int nc = c_coord + DC[k_neighbor_idx];
            if (is_cell_on_grid(nr,nc)) {
                candidate_new_colors[num_candidate_options++] = current_grid_state[nr][nc];
            } else { 
                candidate_new_colors[num_candidate_options++] = 0;
            }
        }
        int new_proposed_color = candidate_new_colors[rnd_gen.next_int(num_candidate_options)];

        if (original_color_at_cell == new_proposed_color) continue; 
        
        int delta_in_score_metric = 0; 
        if (new_proposed_color == 0 && original_color_at_cell != 0) delta_in_score_metric = 1;
        else if (new_proposed_color != 0 && original_color_at_cell == 0) delta_in_score_metric = -1;
        
        if (attempt_change_cell_color_and_validate(r_coord, c_coord, original_color_at_cell, new_proposed_color)) {
            bool accept_this_move = false;
            if (delta_in_score_metric >= 0) { 
                accept_this_move = true;
                if (cells_by_color[0].size() > best_score_val) { 
                    best_score_val = cells_by_color[0].size();
                    best_grid_state = current_grid_state; 
                }
            } else { 
                if (current_temperature > 1e-9 && rnd_gen.next_double() < std::exp((double)delta_in_score_metric / current_temperature)) {
                    accept_this_move = true;
                } else {
                    accept_this_move = false;
                }
            }

            if (!accept_this_move) { 
                current_grid_state[r_coord][c_coord] = original_color_at_cell; 
                remove_cell_from_color_ds(r_coord, c_coord, new_proposed_color); 
                add_cell_to_color_ds(r_coord, c_coord, original_color_at_cell);      
                
                for(const auto& entry : temp_adj_deltas_map_global) { 
                    int c1_ = entry.first.first; int c2_ = entry.first.second; int delta = entry.second;
                    if (delta > 0) for(int i=0; i<delta; ++i) current_border_edges_tracker.remove_edge(c1_,c2_);
                    else for(int i=0; i<-delta; ++i) current_border_edges_tracker.add_edge(c1_,c2_);
                }
            }
        } 
    }

    for (int i = 0; i < N_FIXED; ++i) {
        for (int j = 0; j < N_FIXED; ++j) {
            std::cout << best_grid_state[i][j] << (j == N_FIXED - 1 ? "" : " ");
        }
        std::cout << std::endl;
    }
}

int main() {
    std::ios_base::sync_with_stdio(false); std::cin.tie(NULL);
    G_START_TIME = std::chrono::high_resolution_clock::now();
    
    int n_in_dummy, m_in_dummy; 
    std::cin >> n_in_dummy >> m_in_dummy; 
    
    solve_main_logic();
    return 0;
}
# EVOLVE-BLOCK-END