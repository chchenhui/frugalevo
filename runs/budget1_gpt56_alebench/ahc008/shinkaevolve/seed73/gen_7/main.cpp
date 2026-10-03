# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <cstring>

constexpr int GRID_SIZE = 30;
constexpr int NUM_TURNS = 300;
constexpr int INF = 1 << 28;

struct Point {
    int r, c;

    bool operator==(const Point& other) const {
        return r == other.r && c == other.c;
    }
    bool operator<(const Point& other) const {
        return r != other.r ? r < other.r : c < other.c;
    }
};

const Point INVALID_POINT = {-1, -1};
const Point DIRS[4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
const char BUILD_CHARS[4] = {'u', 'd', 'l', 'r'};
const char MOVE_CHARS[4] = {'U', 'D', 'L', 'R'};

struct PetInfo {
    Point pos;
    int type;
};

struct HumanInfo {
    Point pos;
    int strip_start, strip_end;
    Point safe_ul, safe_br;
    Point final_pos;
    std::vector<Point> walls;
    int stuck_turns = 0;
};

int N, M;
bool wall_grid[31][31];
std::vector<PetInfo> pets;
std::vector<HumanInfo> humans;

bool valid(Point p) {
    return 1 <= p.r && p.r <= GRID_SIZE && 1 <= p.c && p.c <= GRID_SIZE;
}

int action_dir(char ch) {
    for (int d = 0; d < 4; ++d) {
        if (ch == BUILD_CHARS[d] || ch == MOVE_CHARS[d]) return d;
    }
    return -1;
}

bool has_tentative_wall(Point p, const std::vector<Point>& tentative) {
    for (const Point& q : tentative) {
        if (p == q) return true;
    }
    return false;
}

bool can_build(Point p, int builder_id) {
    if (!valid(p) || wall_grid[p.r][p.c]) return false;

    for (const auto& pet : pets) {
        int md = std::abs(p.r - pet.pos.r) + std::abs(p.c - pet.pos.c);
        if (md <= 1) return false;
    }

    for (int i = 0; i < M; ++i) {
        if (i != builder_id && humans[i].pos == p) return false;
    }
    return true;
}

void make_distance_map(
    Point start,
    const std::vector<Point>& tentative,
    int dist[31][31]
) {
    bool blocked[31][31] = {};
    for (const Point& p : tentative) {
        if (valid(p)) blocked[p.r][p.c] = true;
    }

    for (int r = 1; r <= GRID_SIZE; ++r) {
        for (int c = 1; c <= GRID_SIZE; ++c) {
            dist[r][c] = -1;
        }
    }

    if (!valid(start)) return;

    std::queue<Point> q;
    q.push(start);
    dist[start.r][start.c] = 0;

    while (!q.empty()) {
        Point cur = q.front();
        q.pop();

        for (int d = 0; d < 4; ++d) {
            Point nxt = {cur.r + DIRS[d].r, cur.c + DIRS[d].c};
            if (!valid(nxt) || wall_grid[nxt.r][nxt.c] || blocked[nxt.r][nxt.c]) continue;
            if (dist[nxt.r][nxt.c] != -1) continue;
            dist[nxt.r][nxt.c] = dist[cur.r][cur.c] + 1;
            q.push(nxt);
        }
    }
}

char bfs_move(Point start, Point target, const std::vector<Point>& tentative) {
    if (start == target) return '.';

    bool blocked[31][31] = {};
    bool visited[31][31] = {};
    Point parent[31][31];

    for (const Point& p : tentative) {
        if (valid(p)) blocked[p.r][p.c] = true;
    }

    if (!valid(start) || !valid(target) || blocked[target.r][target.c]) return '.';

    std::queue<Point> q;
    q.push(start);
    visited[start.r][start.c] = true;
    parent[start.r][start.c] = INVALID_POINT;

    bool found = false;
    while (!q.empty() && !found) {
        Point cur = q.front();
        q.pop();

        for (int d = 0; d < 4; ++d) {
            Point nxt = {cur.r + DIRS[d].r, cur.c + DIRS[d].c};
            if (!valid(nxt) || wall_grid[nxt.r][nxt.c] || blocked[nxt.r][nxt.c]) continue;
            if (visited[nxt.r][nxt.c]) continue;

            visited[nxt.r][nxt.c] = true;
            parent[nxt.r][nxt.c] = cur;

            if (nxt == target) {
                found = true;
                break;
            }
            q.push(nxt);
        }
    }

    if (!found) return '.';

    Point cur = target;
    while (!(parent[cur.r][cur.c] == start)) {
        cur = parent[cur.r][cur.c];
    }

    for (int d = 0; d < 4; ++d) {
        if (start.r + DIRS[d].r == cur.r && start.c + DIRS[d].c == cur.c) {
            return MOVE_CHARS[d];
        }
    }
    return '.';
}

void initialize() {
    std::cin >> N;
    pets.resize(N);

    for (int i = 0; i < N; ++i) {
        std::cin >> pets[i].pos.r >> pets[i].pos.c >> pets[i].type;
    }

    std::cin >> M;
    humans.resize(M);
    std::memset(wall_grid, 0, sizeof(wall_grid));

    int base_height = GRID_SIZE / M;
    int remainder = GRID_SIZE % M;
    int row_start = 1;

    for (int i = 0; i < M; ++i) {
        HumanInfo& h = humans[i];
        std::cin >> h.pos.r >> h.pos.c;

        int height = base_height + (i < remainder ? 1 : 0);
        h.strip_start = row_start;
        h.strip_end = row_start + height - 1;

        h.safe_ul = {h.strip_start + 1, 2};
        h.safe_br = {h.strip_end - 1, 29};

        if (h.safe_ul.r > h.safe_br.r) {
            h.safe_ul.r = h.safe_br.r = (h.strip_start + h.strip_end) / 2;
        }

        h.final_pos = {
            (h.safe_ul.r + h.safe_br.r) / 2,
            (h.safe_ul.c + h.safe_br.c) / 2
        };

        h.walls.clear();

        // Lower half of the partition above this strip.
        if (i > 0) {
            for (int c = 16; c <= 30; ++c) {
                h.walls.push_back({h.strip_start, c});
            }
        }

        // Upper half of the partition below this strip.
        if (i + 1 < M) {
            for (int c = 1; c <= 15; ++c) {
                h.walls.push_back({h.strip_end, c});
            }
        }

        row_start = h.strip_end + 1;
    }
}

std::string decide_actions() {
    std::string actions(M, '.');
    std::vector<Point> tentative_walls;
    tentative_walls.reserve(M);

    for (int i = 0; i < M; ++i) {
        HumanInfo& h = humans[i];

        int remaining = 0;
        for (const Point& p : h.walls) {
            if (!wall_grid[p.r][p.c]) ++remaining;
        }

        if (remaining == 0) {
            actions[i] = bfs_move(h.pos, h.final_pos, tentative_walls);
            continue;
        }

        if (h.stuck_turns >= 10) {
            h.stuck_turns = 0;
            actions[i] = bfs_move(h.pos, h.final_pos, tentative_walls);
            continue;
        }

        int dist[31][31];
        make_distance_map(h.pos, tentative_walls, dist);

        Point best_wall = INVALID_POINT;
        Point best_stand = INVALID_POINT;
        int best_score = INF;

        for (const Point& wall_pos : h.walls) {
            if (wall_grid[wall_pos.r][wall_pos.c]) continue;
            if (!can_build(wall_pos, i)) continue;
            if (has_tentative_wall(wall_pos, tentative_walls)) continue;

            for (int d = 0; d < 4; ++d) {
                Point stand = {wall_pos.r + DIRS[d].r, wall_pos.c + DIRS[d].c};

                if (!valid(stand) || wall_grid[stand.r][stand.c]) continue;
                if (has_tentative_wall(stand, tentative_walls)) continue;
                if (dist[stand.r][stand.c] < 0) continue;

                int score = dist[stand.r][stand.c];

                bool in_safe =
                    h.safe_ul.r <= stand.r && stand.r <= h.safe_br.r &&
                    h.safe_ul.c <= stand.c && stand.c <= h.safe_br.c;

                if (!in_safe) score += 1000;

                if (score < best_score ||
                    (score == best_score &&
                     (best_wall.r == -1 || wall_pos < best_wall ||
                      (wall_pos == best_wall && stand < best_stand)))) {
                    best_score = score;
                    best_wall = wall_pos;
                    best_stand = stand;
                }
            }
        }

        if (best_wall.r == -1) {
            ++h.stuck_turns;
            actions[i] = bfs_move(h.pos, h.final_pos, tentative_walls);
            continue;
        }

        h.stuck_turns = 0;

        if (h.pos == best_stand) {
            for (int d = 0; d < 4; ++d) {
                if (h.pos.r + DIRS[d].r == best_wall.r &&
                    h.pos.c + DIRS[d].c == best_wall.c) {
                    actions[i] = BUILD_CHARS[d];
                    tentative_walls.push_back(best_wall);
                    break;
                }
            }
        } else {
            actions[i] = bfs_move(h.pos, best_stand, tentative_walls);
        }
    }

    // A move cannot enter a square being made impassable this turn.
    for (int i = 0; i < M; ++i) {
        if (actions[i] < 'A' || actions[i] > 'Z') continue;

        int d = action_dir(actions[i]);
        Point target = {
            humans[i].pos.r + DIRS[d].r,
            humans[i].pos.c + DIRS[d].c
        };

        if (!valid(target) || wall_grid[target.r][target.c] ||
            has_tentative_wall(target, tentative_walls)) {
            actions[i] = '.';
        }
    }

    return actions;
}

void apply_and_read(const std::string& actions) {
    for (int i = 0; i < M; ++i) {
        char a = actions[i];
        if (a < 'a' || a > 'z') continue;

        int d = action_dir(a);
        if (d < 0) continue;

        Point p = {
            humans[i].pos.r + DIRS[d].r,
            humans[i].pos.c + DIRS[d].c
        };

        if (valid(p)) wall_grid[p.r][p.c] = true;
    }

    for (int i = 0; i < M; ++i) {
        char a = actions[i];
        if (a < 'A' || a > 'Z') continue;

        int d = action_dir(a);
        if (d < 0) continue;

        Point p = {
            humans[i].pos.r + DIRS[d].r,
            humans[i].pos.c + DIRS[d].c
        };

        if (valid(p) && !wall_grid[p.r][p.c]) {
            humans[i].pos = p;
        }
    }

    for (int i = 0; i < N; ++i) {
        std::string moves;
        std::cin >> moves;

        for (char ch : moves) {
            int d = -1;
            if (ch == 'U') d = 0;
            else if (ch == 'D') d = 1;
            else if (ch == 'L') d = 2;
            else if (ch == 'R') d = 3;

            if (d >= 0) {
                pets[i].pos.r += DIRS[d].r;
                pets[i].pos.c += DIRS[d].c;
            }
        }
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    initialize();

    for (int turn = 0; turn < NUM_TURNS; ++turn) {
        std::string actions = decide_actions();
        std::cout << actions << std::endl;
        apply_and_read(actions);
    }

    return 0;
}
# EVOLVE-BLOCK-END