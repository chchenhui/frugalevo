# EVOLVE-BLOCK-START
#pragma GCC optimize("O3,unroll-loops")

#include <bits/stdc++.h>
using namespace std;

static constexpr int MAXN = 40;
static constexpr int MAXV = 1600;
static constexpr int MAXL = 100000;
static constexpr int DR[4] = {0,1,0,-1};
static constexpr int DC[4] = {1,0,-1,0};
static constexpr char CH[4] = {'R','D','L','U'};

int N, V;
int dirt[MAXV];
string hwall[40], vwall[40];
vector<pair<int,char>> adj[MAXV];
short dista[MAXV][MAXV];
short parentv[MAXV][MAXV];
vector<int> visits[MAXV];
mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count());

inline int id(int r,int c){ return r*N+c; }
inline int row(int x){ return x/N; }
inline int col(int x){ return x%N; }
inline char inv(char c) {
    if(c=='R') return 'L';
    if(c=='L') return 'R';
    if(c=='U') return 'D';
    return 'U';
}

struct State {
    vector<char> mv;
    vector<int> pos;
    long double term[MAXV];
    long double score = 1e30L;
};

void build_graph() {
    V=N*N;
    for(int r=0;r<N;r++) for(int c=0;c<N;c++) {
        int a=id(r,c);
        for(int d=0;d<4;d++) {
            int nr=r+DR[d], nc=c+DC[d];
            if(nr<0||nr>=N||nc<0||nc>=N) continue;
            bool wall=false;
            if(d==0) wall=(vwall[r][c]=='1');
            if(d==2) wall=(vwall[r][c-1]=='1');
            if(d==1) wall=(hwall[r][c]=='1');
            if(d==3) wall=(hwall[r-1][c]=='1');
            if(!wall) adj[a].push_back({id(nr,nc),CH[d]});
        }
    }
}

void apsp() {
    static int q[MAXV];
    for(int s=0;s<V;s++) {
        for(int t=0;t<V;t++) dista[s][t]=-1;
        int head=0, tail=0;
        q[tail++]=s;
        dista[s][s]=0;
        while(head<tail) {
            int x=q[head++];
            for(auto [y,ch]:adj[x]) if(dista[s][y]<0) {
                dista[s][y]=dista[s][x]+1;
                parentv[s][y]=x;
                q[tail++]=y;
            }
        }
    }
}

void shortest_moves(int a,int b, vector<char>& out) {
    out.clear();
    while(a!=b) {
        int p=parentv[a][b];
        int pr=row(p), pc=col(p), br=row(b), bc=col(b);
        if(br==pr && bc==pc+1) out.push_back('R');
        else if(br==pr && bc==pc-1) out.push_back('L');
        else if(br==pr+1) out.push_back('D');
        else out.push_back('U');
        b=p;
    }
    reverse(out.begin(),out.end());
}

bool score(State& s) {
    int L=(int)s.mv.size();
    if(L<=0 || L>MAXL) return false;
    s.pos.clear();
    s.pos.reserve(L+1);
    s.pos.push_back(0);
    int cur=0;
    bool seen[MAXV]={};
    seen[0]=true;
    for(char ch:s.mv) {
        int nx=-1;
        for(auto [to,c]:adj[cur]) if(c==ch) { nx=to; break; }
        if(nx<0) return false;
        cur=nx;
        seen[cur]=true;
        s.pos.push_back(cur);
    }
    if(cur!=0) return false;
    for(int i=0;i<V;i++) if(!seen[i]) return false;

    for(int i=0;i<V;i++) visits[i].clear();
    for(int t=1;t<=L;t++) visits[s.pos[t]].push_back(t);

    long double total=0;
    for(int x=0;x<V;x++) {
        long double z=0;
        const auto& a=visits[x];
        if(a.empty()) z=(long double)L*(L-1)/2;
        else {
            int m=(int)a.size();
            for(int i=0;i<m;i++) {
                long long prev=(i? a[i-1] : a.back()-L);
                long long gap=(long long)a[i]-prev;
                z+=(long double)gap*(gap-1)/2;
            }
        }
        s.term[x]=z;
        total+=z*dirt[x];
    }
    s.score=total/L;
    return true;
}

bool useddfs[MAXV];
void dfs_init(int x, vector<char>& out) {
    useddfs[x]=true;
    for(auto [y,ch]:adj[x]) if(!useddfs[y]) {
        out.push_back(ch);
        dfs_init(y,out);
        out.push_back(inv(ch));
    }
}

int choose_target(const State& s, bool narrow) {
    vector<pair<long double,int>> a;
    a.reserve(V);
    for(int i=0;i<V;i++) a.push_back({(long double)dirt[i]*s.term[i],i});
    int k=narrow ? min(V,N) : min(V,max(10,V/10));
    nth_element(a.begin(),a.begin()+k,a.end(),
        [](const auto& x,const auto& y){ return x.first>y.first; });
    uniform_int_distribution<int> pick(0,k-1);
    return a[pick(rng)].second;
}

int main(int argc,char**argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    double limit=1.94;
    if(argc>1) limit=stod(argv[1]);
    auto start=chrono::steady_clock::now();

    cin>>N;
    for(int i=0;i<N-1;i++) cin>>hwall[i];
    for(int i=0;i<N;i++) cin>>vwall[i];
    for(int r=0;r<N;r++) for(int c=0;c<N;c++) cin>>dirt[id(r,c)];

    build_graph();
    apsp();

    State cur;
    dfs_init(0,cur.mv);
    score(cur);
    State best=cur;

    vector<char> p1,p2;
    uniform_real_distribution<double> real01(0.0,1.0);
    int it=0;

    while(true) {
        if((++it&127)==0) {
            double e=chrono::duration<double>(chrono::steady_clock::now()-start).count();
            if(e>=limit) break;
        }

        int L=(int)cur.mv.size();
        State cand=cur;
        int op=(int)(rng()%100);
        bool changed=false;

        if(op<15 && L+2<=MAXL) {
            int k=rng()%(L+1), x=cur.pos[k];
            auto &e=adj[x];
            auto [y,ch]=e[rng()%e.size()];
            cand.mv.insert(cand.mv.begin()+k,{ch,inv(ch)});
            changed=true;
        } else if(op<30 && L>=2) {
            vector<int> can;
            for(int i=0;i+2<=L;i++) if(cur.pos[i]==cur.pos[i+2]) can.push_back(i);
            if(!can.empty()) {
                int k=can[rng()%can.size()];
                cand.mv.erase(cand.mv.begin()+k,cand.mv.begin()+k+2);
                changed=true;
            }
        } else if(op<60 && L>=2) {
            int a=rng()%L;
            int b=a+1+rng()%(L-a);
            shortest_moves(cur.pos[a],cur.pos[b],p1);
            if((int)p1.size()<b-a) {
                cand.mv.erase(cand.mv.begin()+a,cand.mv.begin()+b);
                cand.mv.insert(cand.mv.begin()+a,p1.begin(),p1.end());
                changed=true;
            }
        } else if(op<70 && L>0) {
            int a=rng()%L;
            int b=a+rng()%(L-a);
            reverse(cand.mv.begin()+a,cand.mv.begin()+b+1);
            for(int i=a;i<=b;i++) cand.mv[i]=inv(cand.mv[i]);
            changed=true;
        } else if(op<85 && L<MAXL-100) {
            int target=choose_target(cur,false);
            int midpoint=0;
            if(!visits[target].empty()) {
                int bg=-1, bp=0, bc=0;
                const auto& vv=visits[target];
                for(int i=0;i<(int)vv.size();i++) {
                    int p=i ? vv[i-1] : vv.back()-L;
                    int g=vv[i]-p;
                    if(g>bg) bg=g,bp=p,bc=vv[i];
                }
                midpoint=(bp+bc)/2;
                if(midpoint<0) midpoint+=L;
                if(midpoint>L) midpoint-=L;
            }

            int bestk=midpoint;
            int bestd=dista[cur.pos[bestk]][target];
            for(int z=0;z<20;z++) {
                int k=rng()%(L+1);
                int d=dista[cur.pos[k]][target];
                if(d<bestd) bestd=d,bestk=k;
            }
            if(bestd>=0 && L+2*bestd<=MAXL) {
                int x=cur.pos[bestk];
                shortest_moves(x,target,p1);
                shortest_moves(target,x,p2);
                cand.mv.insert(cand.mv.begin()+bestk,p2.begin(),p2.end());
                cand.mv.insert(cand.mv.begin()+bestk,p1.begin(),p1.end());
                changed=true;
            }
        } else if(L>0) {
            int target=choose_target(cur,true);
            int a=rng()%L;
            int b=a+1+rng()%(min(20,L-a));
            shortest_moves(cur.pos[a],target,p1);
            shortest_moves(target,cur.pos[b],p2);
            int nl=L-(b-a)+(int)p1.size()+(int)p2.size();
            if(nl<=MAXL) {
                cand.mv.erase(cand.mv.begin()+a,cand.mv.begin()+b);
                cand.mv.insert(cand.mv.begin()+a,p2.begin(),p2.end());
                cand.mv.insert(cand.mv.begin()+a,p1.begin(),p1.end());
                changed=true;
            }
        }

        if(!changed || !score(cand)) continue;

        double elapsed=chrono::duration<double>(chrono::steady_clock::now()-start).count();
        double ratio=min(1.0,max(0.0,elapsed/limit));
        double temp=5000.0*sqrt((double)N)*pow(0.1/(5000.0*sqrt((double)N)),ratio);

        if(cand.score<cur.score ||
           exp((double)((cur.score-cand.score)/temp))>real01(rng)) {
            cur=move(cand);
            if(cur.score<best.score) best=cur;
        }
    }

    for(char c:best.mv) cout<<c;
    cout<<'\n';
    return 0;
}
# EVOLVE-BLOCK-END