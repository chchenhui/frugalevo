# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

constexpr int S = 30;
constexpr int TURNS = 300;
constexpr int INF = 1e9;

struct Pos {
    int x, y;
    bool operator==(const Pos& o) const { return x == o.x && y == o.y; }
    bool operator<(const Pos& o) const {
        return x != o.x ? x < o.x : y < o.y;
    }
};

const int DX[4] = {-1, 1, 0, 0};
const int DY[4] = {0, 0, -1, 1};
const char BUILD[4] = {'u', 'd', 'l', 'r'};
const char MOVE[4] = {'U', 'D', 'L', 'R'};

struct Pet {
    Pos p;
    int type;
};

struct Human {
    Pos p;
    int id;
    int rank;
    int top, bottom;
    Pos safe;
    vector<Pos> walls;
    int stalled = 0;
};

struct TurnPlan {
    bool reserved_wall[S + 1][S + 1]{};
    string actions;

    TurnPlan(int m) : actions(m, '.') {}
};

class Solver {
    int n, m;
    vector<Pet> pets;
    vector<Human> humans;
    bool wall[S + 1][S + 1]{};

    int dist[S + 1][S + 1];
    Pos parent[S + 1][S + 1];
    bool seen[S + 1][S + 1];

    bool inside(Pos p) const {
        return 1 <= p.x && p.x <= S && 1 <= p.y && p.y <= S;
    }

    int direction_of(char c) const {
        for (int d = 0; d < 4; ++d) {
            if (BUILD[d] == c || MOVE[d] == c) return d;
        }
        return -1;
    }

    bool pet_near(Pos p) const {
        for (const Pet& pet : pets) {
            int md = abs(p.x - pet.p.x) + abs(p.y - pet.p.y);
            if (md <= 1) return true;
        }
        return false;
    }

    bool occupied_by_other_human(Pos p, int self) const {
        for (const Human& h : humans) {
            if (h.id != self && h.p == p) return true;
        }
        return false;
    }

    bool can_build(Pos p, int self, const TurnPlan& plan) const {
        if (!inside(p) || wall[p.x][p.y] || plan.reserved_wall[p.x][p.y]) return false;
        if (occupied_by_other_human(p, self)) return false;
        return !pet_near(p);
    }

    void bfs(Pos start, const TurnPlan& plan) {
        for (int x = 1; x <= S; ++x) {
            for (int y = 1; y <= S; ++y) {
                dist[x][y] = -1;
                seen[x][y] = false;
            }
        }

        queue<Pos> q;
        q.push(start);
        dist[start.x][start.y] = 0;
        seen[start.x][start.y] = true;

        while (!q.empty()) {
            Pos cur = q.front();
            q.pop();

            for (int d = 0; d < 4; ++d) {
                Pos nx{cur.x + DX[d], cur.y + DY[d]};
                if (!inside(nx) || seen[nx.x][nx.y]) continue;
                if (wall[nx.x][nx.y] || plan.reserved_wall[nx.x][nx.y]) continue;
                seen[nx.x][nx.y] = true;
                dist[nx.x][nx.y] = dist[cur.x][cur.y] + 1;
                parent[nx.x][nx.y] = cur;
                q.push(nx);
            }
        }
    }

    char route_to(Pos start, Pos goal, const TurnPlan& plan) {
        if (start == goal) return '.';
        bfs(start, plan);
        if (!inside(goal) || dist[goal.x][goal.y] < 0) return '.';

        Pos cur = goal;
        while (!(parent[cur.x][cur.y] == start)) {
            cur = parent[cur.x][cur.y];
        }

        for (int d = 0; d < 4; ++d) {
            if (start.x + DX[d] == cur.x && start.y + DY[d] == cur.y) {
                return MOVE[d];
            }
        }
        return '.';
    }

    bool in_safe_strip(const Human& h, Pos p) const {
        return h.top < p.x && p.x < h.bottom && 1 < p.y && p.y < S;
    }

    int remaining_walls(const Human& h) const {
        int cnt = 0;
        for (Pos p : h.walls) {
            if (!wall[p.x][p.y]) ++cnt;
        }
        return cnt;
    }

    pair<Pos, Pos> choose_build_job(const Human& h, TurnPlan& plan) {
        bfs(h.p, plan);

        Pos best_wall{-1, -1};
        Pos best_stand{-1, -1};
        int best_score = INF;

        for (Pos target : h.walls) {
            if (!can_build(target, h.id, plan)) continue;

            for (int d = 0; d < 4; ++d) {
                Pos stand{target.x + DX[d], target.y + DY[d]};
                if (!inside(stand) || wall[stand.x][stand.y]) continue;
                if (plan.reserved_wall[stand.x][stand.y]) continue;
                if (dist[stand.x][stand.y] < 0) continue;

                int score = dist[stand.x][stand.y];

                // Prefer working from inside the assigned strip.  This avoids
                // leaving the human on the wrong side when a separator closes.
                if (!in_safe_strip(h, stand)) score += 10000;

                // Prefer extending already-made portions of a separator.
                int joined = 0;
                for (int e = 0; e < 4; ++e) {
                    Pos q{target.x + DX[e], target.y + DY[e]};
                    if (!inside(q) || wall[q.x][q.y]) ++joined;
                }
                score -= joined * 2;

                if (score < best_score ||
                    (score == best_score &&
                     (best_wall.x == -1 || target < best_wall ||
                      (target == best_wall && stand < best_stand)))) {
                    best_score = score;
                    best_wall = target;
                    best_stand = stand;
                }
            }
        }
        return {best_wall, best_stand};
    }

    char decide_for_human(Human& h, TurnPlan& plan) {
        int rem = remaining_walls(h);

        if (rem == 0) {
            h.stalled = 0;
            return route_to(h.p, h.safe, plan);
        }

        auto [target, stand] = choose_build_job(h, plan);
        if (target.x == -1) {
            ++h.stalled;
            // A pet temporarily blocks this separator cell.  Waiting in the
            // interior is safer than repeatedly trying an illegal action.
            if (h.stalled >= 12) {
                h.stalled = 0;
                return route_to(h.p, h.safe, plan);
            }
            return '.';
        }

        h.stalled = 0;
        if (h.p == stand) {
            for (int d = 0; d < 4; ++d) {
                if (h.p.x + DX[d] == target.x && h.p.y + DY[d] == target.y) {
                    plan.reserved_wall[target.x][target.y] = true;
                    return BUILD[d];
                }
            }
        }

        return route_to(h.p, stand, plan);
    }

    string make_turn() {
        TurnPlan plan(m);

        // Input indices determine output positions, but construction priority
        // follows strip order. This prevents a random input ordering from
        // systematically delaying top or bottom separator completion.
        vector<int> order(m);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            return humans[a].rank < humans[b].rank;
        });

        for (int id : order) {
            char action = decide_for_human(humans[id], plan);
            plan.actions[id] = action;

            // Reserve any wall generated by a direct build. Routes already
            // avoid all earlier reservations through the same TurnPlan.
            if ('a' <= action && action <= 'z') {
                int d = direction_of(action);
                if (d >= 0) {
                    Pos w{humans[id].p.x + DX[d], humans[id].p.y + DY[d]};
                    if (inside(w)) plan.reserved_wall[w.x][w.y] = true;
                }
            }
        }
        return plan.actions;
    }

    void apply_and_read(const string& actions) {
        // All constructions happen before movements.
        for (int i = 0; i < m; ++i) {
            if (!('a' <= actions[i] && actions[i] <= 'z')) continue;
            int d = direction_of(actions[i]);
            Pos p{humans[i].p.x + DX[d], humans[i].p.y + DY[d]};
            if (inside(p)) wall[p.x][p.y] = true;
        }

        for (int i = 0; i < m; ++i) {
            if (!('A' <= actions[i] && actions[i] <= 'Z')) continue;
            int d = direction_of(actions[i]);
            Pos p{humans[i].p.x + DX[d], humans[i].p.y + DY[d]};
            if (inside(p) && !wall[p.x][p.y]) humans[i].p = p;
        }

        for (int i = 0; i < n; ++i) {
            string s;
            cin >> s;
            for (char c : s) {
                int d = direction_of(c);
                if (d >= 0) {
                    pets[i].p.x += DX[d];
                    pets[i].p.y += DY[d];
                }
            }
        }
    }

    void build_strip_plan() {
        vector<int> order(m);
        iota(order.begin(), order.end(), 0);

        sort(order.begin(), order.end(), [&](int a, int b) {
            if (humans[a].p.x != humans[b].p.x) {
                return humans[a].p.x < humans[b].p.x;
            }
            return humans[a].p.y < humans[b].p.y;
        });

        int base = S / m;
        int rem = S % m;
        int row = 1;

        for (int rank = 0; rank < m; ++rank) {
            Human& h = humans[order[rank]];
            h.rank = rank;
            int height = base + (rank < rem ? 1 : 0);
            h.top = row;
            h.bottom = row + height - 1;
            row = h.bottom + 1;

            h.safe = {(h.top + h.bottom) / 2, S / 2};
            h.safe.x = max(h.top + 1, min(h.bottom - 1, h.safe.x));
            h.safe.y = max(2, min(S - 1, h.safe.y));

            h.walls.clear();

            // A separator is built in two independent halves. The upper
            // strip owns the left half of its bottom boundary; the lower
            // strip owns the right half of its top boundary.
            if (rank > 0) {
                for (int y = 16; y <= 30; ++y) h.walls.push_back({h.top, y});
            }
            if (rank + 1 < m) {
                for (int y = 1; y <= 15; ++y) h.walls.push_back({h.bottom, y});
            }
        }
    }

public:
    void run() {
        cin >> n;
        pets.resize(n);
        for (int i = 0; i < n; ++i) {
            cin >> pets[i].p.x >> pets[i].p.y >> pets[i].type;
        }

        cin >> m;
        humans.resize(m);
        for (int i = 0; i < m; ++i) {
            humans[i].id = i;
            cin >> humans[i].p.x >> humans[i].p.y;
        }

        build_strip_plan();

        for (int turn = 0; turn < TURNS; ++turn) {
            string actions = make_turn();
            cout << actions << endl;
            apply_and_read(actions);
        }
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Solver solver;
    solver.run();
    return 0;
}
# EVOLVE-BLOCK-END