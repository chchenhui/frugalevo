# EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <limits>

using namespace std;

constexpr int S = 30;
constexpr int TURNS = 300;
constexpr int INF = 1e9;
constexpr int STAND_OUTSIDE_INNER_SAFE_PENALTY = 1000;
constexpr int ADJACENT_WALL_PRIORITY_BONUS = 0;
constexpr int MAX_STUCK_TURNS = 10;

struct Point {
    int r, c;
    bool operator==(const Point& o) const { return r == o.r && c == o.c; }
    bool operator<(const Point& o) const {
        return r != o.r ? r < o.r : c < o.c;
    }
};

const Point D[4] = {{-1,0},{1,0},{0,-1},{0,1}};
const char BUILD[4] = {'u','d','l','r'};
const char MOVE[4] = {'U','D','L','R'};

struct Pet {
    Point p;
    int type;
};

struct Human {
    Point p;
    int rs, re;
    Point innerUL, innerBR, finalPos;
    vector<Point> walls;
    int stuck = 0;
};

int N, M;
Pet pets[20];
Human humans[10];
bool wallGrid[31][31];

int distGrid[31][31];
Point parentGrid[31][31];

inline bool valid(int r, int c) {
    return 1 <= r && r <= S && 1 <= c && c <= S;
}
inline bool valid(Point p) {
    return valid(p.r, p.c);
}
inline int manhattan(Point a, Point b) {
    return abs(a.r - b.r) + abs(a.c - b.c);
}

int actionDir(char ch) {
    for (int d = 0; d < 4; ++d) {
        if (ch == BUILD[d] || ch == MOVE[d]) return d;
    }
    return -1;
}

bool petNearOrOn(Point x) {
    for (int i = 0; i < N; ++i) {
        if (manhattan(x, pets[i].p) <= 1) return true;
    }
    return false;
}

bool humanOn(Point x, int except) {
    for (int i = 0; i < M; ++i) {
        if (i != except && humans[i].p == x) return true;
    }
    return false;
}

bool canBuild(Point x, int id) {
    return valid(x) && !wallGrid[x.r][x.c] && !petNearOrOn(x) && !humanOn(x, id);
}

void bfs(Point start, const bool blocked[31][31]) {
    for (int r = 1; r <= S; ++r) {
        for (int c = 1; c <= S; ++c) {
            distGrid[r][c] = -1;
            parentGrid[r][c] = {-1, -1};
        }
    }

    Point q[900];
    int head = 0, tail = 0;
    q[tail++] = start;
    distGrid[start.r][start.c] = 0;

    while (head < tail) {
        Point cur = q[head++];
        for (int d = 0; d < 4; ++d) {
            Point nx{cur.r + D[d].r, cur.c + D[d].c};
            if (!valid(nx) || wallGrid[nx.r][nx.c] || blocked[nx.r][nx.c]) continue;
            if (distGrid[nx.r][nx.c] != -1) continue;
            distGrid[nx.r][nx.c] = distGrid[cur.r][cur.c] + 1;
            parentGrid[nx.r][nx.c] = cur;
            q[tail++] = nx;
        }
    }
}

char firstMoveTo(Point start, Point target, const bool blocked[31][31]) {
    if (start == target) return '.';
    bfs(start, blocked);
    if (!valid(target) || distGrid[target.r][target.c] < 0) return '.';

    Point cur = target;
    while (!(parentGrid[cur.r][cur.c] == start)) {
        cur = parentGrid[cur.r][cur.c];
        if (cur.r < 0) return '.';
    }
    for (int d = 0; d < 4; ++d) {
        if (start.r + D[d].r == cur.r && start.c + D[d].c == cur.c) {
            return MOVE[d];
        }
    }
    return '.';
}

int adjacentWalls(Point p) {
    int res = 0;
    for (int d = 0; d < 4; ++d) {
        Point q{p.r + D[d].r, p.c + D[d].c};
        if (!valid(q) || wallGrid[q.r][q.c]) ++res;
    }
    return res;
}

void initialize() {
    cin >> N;
    for (int i = 0; i < N; ++i) {
        cin >> pets[i].p.r >> pets[i].p.c >> pets[i].type;
    }
    cin >> M;
    memset(wallGrid, 0, sizeof(wallGrid));

    int base = S / M, rem = S % M, rs = 1;
    for (int i = 0; i < M; ++i) {
        Human& h = humans[i];
        cin >> h.p.r >> h.p.c;

        int height = base + (i < rem);
        h.rs = rs;
        h.re = rs + height - 1;
        rs = h.re + 1;

        h.innerUL = {h.rs + 1, 2};
        h.innerBR = {h.re - 1, 29};
        h.finalPos = {(h.innerUL.r + h.innerBR.r) / 2, 15};

        h.walls.clear();
        if (i == 0) {
            for (int c = 1; c <= S; ++c) h.walls.push_back({h.rs, c});
        } else {
            for (int c = 16; c <= S; ++c) h.walls.push_back({h.rs, c});
        }
        if (i == M - 1) {
            for (int c = 1; c <= S; ++c) h.walls.push_back({h.re, c});
        } else {
            for (int c = 1; c <= 15; ++c) h.walls.push_back({h.re, c});
        }
        for (int r = h.rs + 1; r < h.re; ++r) {
            h.walls.push_back({r, 1});
            h.walls.push_back({r, S});
        }
        sort(h.walls.begin(), h.walls.end());
        h.walls.erase(unique(h.walls.begin(), h.walls.end()), h.walls.end());
    }
}

string decideActions() {
    string ans(M, '.');
    bool plannedWalls[31][31] = {};
    bool plannedMoves[31][31] = {};

    for (int i = 0; i < M; ++i) {
        Human& h = humans[i];
        int remaining = 0;
        for (const Point& x : h.walls) {
            if (!wallGrid[x.r][x.c]) ++remaining;
        }

        if (remaining == 0) {
            char a = firstMoveTo(h.p, h.finalPos, plannedWalls);
            ans[i] = a;
            if (a != '.') {
                int d = actionDir(a);
                Point to{h.p.r + D[d].r, h.p.c + D[d].c};
                plannedMoves[to.r][to.c] = true;
            }
            continue;
        }

        if (h.stuck >= MAX_STUCK_TURNS) {
            h.stuck = 0;
            char a = firstMoveTo(h.p, h.finalPos, plannedWalls);
            ans[i] = a;
            if (a != '.') {
                int d = actionDir(a);
                Point to{h.p.r + D[d].r, h.p.c + D[d].c};
                plannedMoves[to.r][to.c] = true;
            }
            continue;
        }

        // Actual shortest path distances, rather than Manhattan estimates.
        bfs(h.p, plannedWalls);

        Point bestWall{-1, -1}, bestStand{-1, -1};
        int bestScore = INF;

        for (const Point& w : h.walls) {
            if (wallGrid[w.r][w.c] || !canBuild(w, i)) continue;

            int bonus = adjacentWalls(w) * ADJACENT_WALL_PRIORITY_BONUS;
            for (int d = 0; d < 4; ++d) {
                Point stand{w.r + D[d].r, w.c + D[d].c};
                if (!valid(stand) || wallGrid[stand.r][stand.c]) continue;
                if (plannedWalls[stand.r][stand.c] || plannedMoves[stand.r][stand.c]) continue;
                if (distGrid[stand.r][stand.c] < 0) continue;

                int score = distGrid[stand.r][stand.c] - bonus;
                bool inside = h.innerUL.r <= stand.r && stand.r <= h.innerBR.r &&
                              h.innerUL.c <= stand.c && stand.c <= h.innerBR.c;
                if (!inside) score += STAND_OUTSIDE_INNER_SAFE_PENALTY;

                if (score < bestScore ||
                    (score == bestScore &&
                     (bestWall.r < 0 || w < bestWall || (w == bestWall && stand < bestStand)))) {
                    bestScore = score;
                    bestWall = w;
                    bestStand = stand;
                }
            }
        }

        if (bestWall.r < 0) {
            ++h.stuck;
            char a = firstMoveTo(h.p, h.finalPos, plannedWalls);
            ans[i] = a;
            if (a != '.') {
                int d = actionDir(a);
                Point to{h.p.r + D[d].r, h.p.c + D[d].c};
                plannedMoves[to.r][to.c] = true;
            }
        } else {
            h.stuck = 0;
            if (h.p == bestStand) {
                for (int d = 0; d < 4; ++d) {
                    if (h.p.r + D[d].r == bestWall.r && h.p.c + D[d].c == bestWall.c) {
                        ans[i] = BUILD[d];
                        plannedWalls[bestWall.r][bestWall.c] = true;
                        break;
                    }
                }
            } else {
                char a = firstMoveTo(h.p, bestStand, plannedWalls);
                ans[i] = a;
                if (a != '.') {
                    int d = actionDir(a);
                    Point to{h.p.r + D[d].r, h.p.c + D[d].c};
                    plannedMoves[to.r][to.c] = true;
                }
            }
        }
    }

    // A move into a square built by a later human this turn is not legal.
    for (int i = 0; i < M; ++i) {
        if (ans[i] >= 'A' && ans[i] <= 'Z') {
            int d = actionDir(ans[i]);
            Point to{humans[i].p.r + D[d].r, humans[i].p.c + D[d].c};
            if (plannedWalls[to.r][to.c]) ans[i] = '.';
        }
    }
    return ans;
}

void applyAndRead(const string& actions) {
    for (int i = 0; i < M; ++i) {
        if (actions[i] >= 'a' && actions[i] <= 'z') {
            int d = actionDir(actions[i]);
            Point w{humans[i].p.r + D[d].r, humans[i].p.c + D[d].c};
            if (valid(w)) wallGrid[w.r][w.c] = true;
        }
    }
    for (int i = 0; i < M; ++i) {
        if (actions[i] >= 'A' && actions[i] <= 'Z') {
            int d = actionDir(actions[i]);
            Point nx{humans[i].p.r + D[d].r, humans[i].p.c + D[d].c};
            if (valid(nx) && !wallGrid[nx.r][nx.c]) humans[i].p = nx;
        }
    }

    for (int i = 0; i < N; ++i) {
        string s;
        cin >> s;
        for (char ch : s) {
            for (int d = 0; d < 4; ++d) {
                if (ch == MOVE[d]) {
                    pets[i].p.r += D[d].r;
                    pets[i].p.c += D[d].c;
                    break;
                }
            }
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    initialize();
    for (int turn = 0; turn < TURNS; ++turn) {
        string actions = decideActions();
        cout << actions << '\n' << flush;
        applyAndRead(actions);
    }
    return 0;
}
# EVOLVE-BLOCK-END