# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

constexpr int S = 30;
constexpr int TURNS = 300;
constexpr int INF = 1e9;

struct Pos {
    int x, y;

    bool operator==(const Pos& o) const {
        return x == o.x && y == o.y;
    }
};

constexpr int DX[4] = {-1, 1, 0, 0};
constexpr int DY[4] = {0, 0, -1, 1};
constexpr char BUILD[4] = {'u', 'd', 'l', 'r'};
constexpr char MOVE[4] = {'U', 'D', 'L', 'R'};

struct Pet {
    Pos p;
    int type;
};

struct Human {
    Pos p;
    int top, bottom;
    Pos center;
    vector<Pos> walls;
};

struct Search {
    int dist[31][31];
    int first_dir[31][31];

    void run(const Pos& start,
             const bool blocked[31][31],
             const bool reserved_wall[31][31]) {
        for (int x = 1; x <= S; ++x) {
            for (int y = 1; y <= S; ++y) {
                dist[x][y] = -1;
                first_dir[x][y] = -1;
            }
        }

        queue<Pos> q;
        dist[start.x][start.y] = 0;
        q.push(start);

        while (!q.empty()) {
            Pos cur = q.front();
            q.pop();

            for (int d = 0; d < 4; ++d) {
                Pos nxt{cur.x + DX[d], cur.y + DY[d]};
                if (nxt.x < 1 || nxt.x > S || nxt.y < 1 || nxt.y > S) continue;
                if (blocked[nxt.x][nxt.y] || reserved_wall[nxt.x][nxt.y]) continue;
                if (dist[nxt.x][nxt.y] != -1) continue;

                dist[nxt.x][nxt.y] = dist[cur.x][cur.y] + 1;
                first_dir[nxt.x][nxt.y] = (dist[cur.x][cur.y] == 0 ? d : first_dir[cur.x][cur.y]);
                q.push(nxt);
            }
        }
    }

    char step_to(const Pos& target) const {
        if (target.x < 1 || target.x > S || target.y < 1 || target.y > S) return '.';
        int d = first_dir[target.x][target.y];
        return d < 0 ? '.' : MOVE[d];
    }
};

class Game {
    int n, m;
    vector<Pet> pets;
    vector<Human> humans;

    bool wall[31][31]{};
    bool pet_danger[31][31]{};
    bool human_here[31][31]{};

public:
    void read_initial() {
        cin >> n;
        pets.resize(n);
        for (auto& pet : pets) cin >> pet.p.x >> pet.p.y >> pet.type;

        cin >> m;
        humans.resize(m);
        for (auto& human : humans) cin >> human.p.x >> human.p.y;

        build_strip_layout();
    }

    void run() {
        for (int turn = 0; turn < TURNS; ++turn) {
            rebuild_dynamic_masks();

            string command = plan_turn();
            cout << command << endl;

            apply_our_actions(command);
            read_pet_actions();
        }
    }

private:
    static bool inside(const Pos& p) {
        return 1 <= p.x && p.x <= S && 1 <= p.y && p.y <= S;
    }

    void add_wall_task(Human& h, int x, int y) {
        if (inside({x, y})) h.walls.push_back({x, y});
    }

    void build_strip_layout() {
        int base = S / m;
        int rem = S % m;
        int row = 1;

        for (int i = 0; i < m; ++i) {
            Human& h = humans[i];
            int height = base + (i < rem ? 1 : 0);
            h.top = row;
            h.bottom = row + height - 1;
            row = h.bottom + 1;

            int inner_top = min(h.bottom, h.top + 1);
            int inner_bottom = max(inner_top, h.bottom - 1);
            h.center = {(inner_top + inner_bottom) / 2, 15};

            // Adjacent strips jointly construct their common border:
            // previous strip owns its left half, next strip owns its right half.
            if (i == 0) {
                for (int y = 1; y <= S; ++y) add_wall_task(h, h.top, y);
            } else {
                for (int y = 16; y <= S; ++y) add_wall_task(h, h.top, y);
            }

            if (i == m - 1) {
                for (int y = 1; y <= S; ++y) add_wall_task(h, h.bottom, y);
            } else {
                for (int y = 1; y <= 15; ++y) add_wall_task(h, h.bottom, y);
            }

            for (int x = h.top + 1; x <= h.bottom - 1; ++x) {
                add_wall_task(h, x, 1);
                add_wall_task(h, x, S);
            }

            sort(h.walls.begin(), h.walls.end(), [](const Pos& a, const Pos& b) {
                return a.x != b.x ? a.x < b.x : a.y < b.y;
            });
            h.walls.erase(unique(h.walls.begin(), h.walls.end(),
                                 [](const Pos& a, const Pos& b) {
                                     return a.x == b.x && a.y == b.y;
                                 }),
                          h.walls.end());
        }
    }

    void rebuild_dynamic_masks() {
        memset(pet_danger, 0, sizeof(pet_danger));
        memset(human_here, 0, sizeof(human_here));

        for (const Human& h : humans) {
            human_here[h.p.x][h.p.y] = true;
        }

        for (const Pet& pet : pets) {
            pet_danger[pet.p.x][pet.p.y] = true;
            for (int d = 0; d < 4; ++d) {
                Pos q{pet.p.x + DX[d], pet.p.y + DY[d]};
                if (inside(q)) pet_danger[q.x][q.y] = true;
            }
        }
    }

    bool is_inner_stand(const Human& h, const Pos& p) const {
        return h.top + 1 <= p.x && p.x <= h.bottom - 1 &&
               2 <= p.y && p.y <= 29;
    }

    int adjacent_solid_count(const Pos& p, const bool reserved_wall[31][31]) const {
        int result = 0;
        for (int d = 0; d < 4; ++d) {
            Pos q{p.x + DX[d], p.y + DY[d]};
            if (!inside(q) || wall[q.x][q.y] || reserved_wall[q.x][q.y]) ++result;
        }
        return result;
    }

    bool can_build(const Pos& target, int owner, const bool reserved_wall[31][31]) const {
        if (!inside(target)) return false;
        if (wall[target.x][target.y] || reserved_wall[target.x][target.y]) return false;
        if (pet_danger[target.x][target.y]) return false;

        for (int i = 0; i < m; ++i) {
            if (i != owner && humans[i].p == target) return false;
        }
        return true;
    }

    bool completed(const Human& h) const {
        for (const Pos& p : h.walls) {
            if (!wall[p.x][p.y]) return false;
        }
        return true;
    }

    string plan_turn() {
        bool reserved_wall[31][31]{};
        string actions(m, '.');

        for (int i = 0; i < m; ++i) {
            Human& h = humans[i];
            Search bfs;
            bfs.run(h.p, wall, reserved_wall);

            if (completed(h)) {
                actions[i] = bfs.step_to(h.center);
                continue;
            }

            Pos best_wall{-1, -1};
            Pos best_stand{-1, -1};
            int best_value = INF;

            for (const Pos& target : h.walls) {
                if (!can_build(target, i, reserved_wall)) continue;

                int join_bonus = adjacent_solid_count(target, reserved_wall);
                for (int d = 0; d < 4; ++d) {
                    Pos stand{target.x + DX[d], target.y + DY[d]};
                    if (!inside(stand)) continue;
                    if (wall[stand.x][stand.y] || reserved_wall[stand.x][stand.y]) continue;
                    if (bfs.dist[stand.x][stand.y] < 0) continue;

                    // Strongly prefer building boundaries from the protected side.
                    int value = bfs.dist[stand.x][stand.y] * 10 - join_bonus * 2;
                    if (!is_inner_stand(h, stand)) value += 10000;

                    if (value < best_value ||
                        (value == best_value &&
                         (target.x < best_wall.x ||
                          (target.x == best_wall.x && target.y < best_wall.y)))) {
                        best_value = value;
                        best_wall = target;
                        best_stand = stand;
                    }
                }
            }

            if (best_wall.x == -1) {
                // A pet may temporarily be blocking every nearby construction task.
                // Repositioning inside the designated strip avoids wasting the turn.
                actions[i] = bfs.step_to(h.center);
                continue;
            }

            if (h.p == best_stand) {
                for (int d = 0; d < 4; ++d) {
                    if (h.p.x + DX[d] == best_wall.x && h.p.y + DY[d] == best_wall.y) {
                        actions[i] = BUILD[d];
                        reserved_wall[best_wall.x][best_wall.y] = true;
                        break;
                    }
                }
            } else {
                actions[i] = bfs.step_to(best_stand);
            }
        }

        return actions;
    }

    int direction_of(char c) const {
        for (int d = 0; d < 4; ++d) {
            if (c == BUILD[d] || c == MOVE[d]) return d;
        }
        return -1;
    }

    void apply_our_actions(const string& actions) {
        // All constructions happen before all movements.
        for (int i = 0; i < m; ++i) {
            if (!islower(actions[i])) continue;
            int d = direction_of(actions[i]);
            Pos q{humans[i].p.x + DX[d], humans[i].p.y + DY[d]};
            if (inside(q)) wall[q.x][q.y] = true;
        }

        for (int i = 0; i < m; ++i) {
            if (!isupper(actions[i])) continue;
            int d = direction_of(actions[i]);
            Pos q{humans[i].p.x + DX[d], humans[i].p.y + DY[d]};
            if (inside(q) && !wall[q.x][q.y]) humans[i].p = q;
        }
    }

    void read_pet_actions() {
        for (int i = 0; i < n; ++i) {
            string s;
            cin >> s;
            for (char c : s) {
                if (c == '.') continue;
                int d = direction_of(c);
                if (d >= 0) {
                    pets[i].p.x += DX[d];
                    pets[i].p.y += DY[d];
                }
            }
        }
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Game game;
    game.read_initial();
    game.run();
    return 0;
}
# EVOLVE-BLOCK-END