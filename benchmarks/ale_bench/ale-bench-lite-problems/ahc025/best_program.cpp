# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

/* Approach overview:
- Sort k pivots by 1v1 balance queries (merge sort), k is a power of two under budget.
- Classify non-pivots by binary search among pivots.
- Assign surrogate weights from ranks (spread quadratically for separation).
- Greedy LPT packing, then a tiny deterministic refinement (move/swap) to reduce sum of squares.
- Use remaining queries (if any) for light interactive refinement between groups, then pad. */

// Global parameters and query counter
int N_, D_, Q_;
int q_used = 0;

// Cache for 1v1 comparisons (symmetry-aware)
/* cmp(a,b):
   - Returns '<' if a is lighter than b, '>' if heavier, '=' if equal.
   - Uses a 1 item per pan query and memoizes symmetric results. */
static unsigned char memo_cmp[128][128]; // 0=unknown; else '<','>','='

// Perform actual query
/* ask(L,R): prints the sets on the pan and reads the judge's response.
   Requirements: both sides non-empty and disjoint; used for both 1v1 and set vs set. */
char ask(const vector<int>& L, const vector<int>& R){ /* Query balance between sets L and R; returns '<','>','='. Flush each call. */
    ++q_used;
    cout << (int)L.size() << " " << (int)R.size();
    for(int x: L) cout << " " << x;
    for(int x: R) cout << " " << x;
    cout << endl;
    char c; cin >> c;
    return c;
}

// Compare two single items with caching
/* cmp(a,b) as comparator:
   - If we ran out of queries, conservatively returns '=' to avoid TLE/illegal ops. */
char cmp(int a, int b){ /* Compare singletons a vs b with memoization; returns '<','>','='. */
    if(a==b) return '=';
    if(memo_cmp[a][b]) return (char)memo_cmp[a][b];
    if(q_used>=Q_) return '=';
    char r = ask({a},{b});
    memo_cmp[a][b] = (unsigned char)r;
    memo_cmp[b][a] = (unsigned char)(r=='<'?'>':(r=='>'?'<':'='));
    return r;
}

// Merge sort using cmp as comparator on item indices
/* merge_sort(ids,l,r):
   - Sorts ids[l..r] in non-decreasing order of weight using only cmp(). */
void merge_sort(vector<int>& ids, int l, int r){ /* Stable merge sort using cmp() as comparator over indices. */
    if(l>=r) return;
    int m=(l+r)>>1;
    merge_sort(ids,l,m);
    merge_sort(ids,m+1,r);
    vector<int> tmp; tmp.reserve(r-l+1);
    int i=l,j=m+1;
    while(i<=m && j<=r){
        char c = cmp(ids[i], ids[j]);
        // Treat '=' as <= to keep order deterministic
        if(c=='<' || c=='=') tmp.push_back(ids[i++]);
        else tmp.push_back(ids[j++]);
    }
    while(i<=m) tmp.push_back(ids[i++]);
    while(j<=r) tmp.push_back(ids[j++]);
    for(int k=0;k<(int)tmp.size();++k) ids[l+k]=tmp[k];
}

int main(){ /* Orchestrates: budgeted ranking via 1v1, surrogate weights from exp order stats, LPT pack, local+interactive refinement. */
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> N_ >> D_ >> Q_;
    // Choose pivots with a reserved budget for later set-based refinement.
    // With k=2^e, mergesort on pivots costs ~k*e and classifying others costs ~(N-k)*e => ~N*e.
    int log2N = 0; while((1<<(log2N+1))<=max(1,N_)) ++log2N;
    int reserve = max(D_, Q_/4);
    int e = min(log2N, max(1, (Q_ - reserve - 2)/max(1,N_)));
    int k = min(N_, 1<<e);
    int e_full = 0; while((1<<e_full) < max(1,N_)) ++e_full;
    long long cost_full = 1LL * N_ * e_full;
    if(Q_ - reserve - 2 >= cost_full) k = N_;
    if(k<1) k=1;

    // Pick first k items as pivots and sort them by weight
    vector<int> piv(k);
    iota(piv.begin(), piv.end(), 0);
    merge_sort(piv, 0, k-1);

    vector<char> is_pivot(N_, 0);
    for(int x: piv) is_pivot[x]=1;

    // Surrogate weights via expected order statistics of exponential distribution.
    // Map rank quantile q in (0,1] to w_hat ∝ H_N - H_{N - round(q*N)}, then scale to integers.
    vector<long double> H(N_+1, 0.0L);
    for(int i=1;i<=N_;++i) H[i]=H[i-1]+1.0L/i;
    long double HN = H[N_];
    auto q_to_w = [&](long double q)->long long{
        if(q<=0) return 1;
        int idx = (int)llround(q * (long double)N_);
        if(idx<1) idx=1; if(idx>N_) idx=N_;
        long double v = HN - H[N_-idx];
        long long w = (long long)llround(v * 1000000.0L);
        if(w<1) w=1;
        return w;
    };
    vector<long long> w_hat(N_, 1);
    for(int i=0;i<k;i++){
        long double q = ((long double)i + 0.5L) / (long double)k;
        w_hat[piv[i]] = q_to_w(q);
    }

    // Classify non-pivot items by binary searching among pivots
    for(int id=0; id<N_; ++id){
        if(is_pivot[id]) continue;
        int lo=0, hi=k;
        while(lo<hi){
            int mid=(lo+hi)/2;
            char r = cmp(id, piv[mid]);
            if(r=='<' || r=='=') hi=mid; else lo=mid+1;
        }
        long double q;
        if(lo==0) q = 0.25L / max(1, k); // slightly above 0
        else if(lo==k) q = ((long double)k - 0.25L) / (long double)k; // slightly below 1
        else q = ((long double)lo) / (long double)k;
        w_hat[id] = q_to_w(q);
    }

    // Greedy largest-first with tie-breaker by size
    vector<pair<long long,int>> ord; ord.reserve(N_);
    for(int i=0;i<N_;++i) ord.push_back({-w_hat[i], i});
    sort(ord.begin(), ord.end());
    vector<long long> sum(D_, 0);
    vector<vector<int>> grp(D_);
    vector<int> ans(N_, 0);
    for(auto [negw, id] : ord){
        int best = 0;
        for(int g=1; g<D_; ++g){
            if(sum[g] < sum[best] || (sum[g]==sum[best] && grp[g].size() < grp[best].size())) best=g;
        }
        ans[id] = best;
        sum[best] += -negw;
        grp[best].push_back(id);
    }

    // Deterministic local refinement: move or swap between heaviest and lightest if it improves sum of squares
    {
        int iter_limit = max(100, N_);
        for(int it=0; it<iter_limit; ++it){
            int gH=0,gL=0;
            for(int g=1; g<D_; ++g){ if(sum[g]>sum[gH]) gH=g; if(sum[g]<sum[gL]) gL=g; }
            if(gH==gL) break;
            long long SA=sum[gH], SB=sum[gL], diff=SA-SB;
            if(diff<=0) break;

            // Try a single-item move from gH to gL
            int bestId=-1; long long bestGap=(1LL<<62);
            for(int id: grp[gH]){
                long long w=w_hat[id];
                if(w<diff){
                    long long gap = llabs((long long)(diff/2) - w);
                    if(gap<bestGap){ bestGap=gap; bestId=id; }
                }
            }
            bool improved=false;
            if(bestId!=-1){
                long long w=w_hat[bestId];
                long long nSA=SA-w, nSB=SB+w;
                long long old2=SA*SA + SB*SB, new2=nSA*nSA + nSB*nSB;
                if(new2<old2){
                    sum[gH]=nSA; sum[gL]=nSB;
                    auto &A=grp[gH]; auto &B=grp[gL];
                    for(int i=0;i<(int)A.size();++i) if(A[i]==bestId){ A[i]=A.back(); A.pop_back(); break; }
                    B.push_back(bestId);
                    improved=true;
                }
            }
            if(improved) continue;

            // Try swapping one item between gH and gL
            int bestA=-1, bestB=-1; bestGap=(1LL<<62);
            for(int ia: grp[gH]){
                long long wA=w_hat[ia];
                for(int ib: grp[gL]){
                    long long wB=w_hat[ib];
                    long long delta = wA - wB;
                    if(delta<=0 || delta>=diff) continue;
                    long long gap = llabs((long long)(diff/2) - delta);
                    if(gap<bestGap){ bestGap=gap; bestA=ia; bestB=ib; }
                }
            }
            if(bestA!=-1){
                long long wA=w_hat[bestA], wB=w_hat[bestB];
                long long nSA=SA - wA + wB, nSB=SB - wB + wA;
                long long old2=SA*SA + SB*SB, new2=nSA*nSA + nSB*nSB;
                if(new2<old2){
                    sum[gH]=nSA; sum[gL]=nSB;
                    auto &A=grp[gH]; auto &B=grp[gL];
                    for(int i=0;i<(int)A.size();++i) if(A[i]==bestA){ A[i]=bestB; break; }
                    for(int i=0;i<(int)B.size();++i) if(B[i]==bestB){ B[i]=bestA; break; }
                    continue;
                }
            }
            break; // no improving move or swap
        }
    }

    // Rebuild final assignment from groups (keeps consistency after refinement)
    fill(ans.begin(), ans.end(), 0);
    for(int g=0; g<D_; ++g) for(int id: grp[g]) ans[id]=g;

    // Set-based interactive refinement using remaining queries: move single items guided by balance
    {
        if(q_used < Q_){
            mt19937 rng(712367);
            vector<pair<int,int>> pairs;
            pairs.reserve(D_*(D_-1)/2);
            for(int a=0;a<D_;++a) for(int b=a+1;b<D_;++b) pairs.emplace_back(a,b);
            int rem = Q_ - q_used;
            int passes = min(24, 2 + rem / max(1, D_));
            int K = min(12, 2 + rem / max(1, D_));
            for(int pass=0; pass<passes && q_used < Q_; ++pass){
                shuffle(pairs.begin(), pairs.end(), rng);
                bool any=false;
                for(auto pr: pairs){
                    int a=pr.first, b=pr.second;
                    if(q_used >= Q_) break;
                    if(grp[a].empty() || grp[b].empty()) continue;
                    char r = ask(grp[a], grp[b]);
                    if(r=='=') continue;
                    int H = (r=='>')?a:b;
                    int L = (H==a)?b:a;
                    if((int)grp[H].size()<=1) continue;
                    vector<int> idx(grp[H].size()); iota(idx.begin(), idx.end(), 0);
                    shuffle(idx.begin(), idx.end(), rng);
                    int tries = min(K, (int)idx.size());
                    for(int t=0; t<tries && q_used < Q_; ++t){
                        int pos = idx[t];
                        int id = grp[H][pos];
                        // Build H\{id} as left set
                        vector<int> Left; Left.reserve(grp[H].size()-1);
                        for(int x: grp[H]) if(x!=id) Left.push_back(x);
                        if(Left.empty() || grp[L].empty()) continue;
                        char rr = ask(Left, grp[L]);
                        if(rr=='>'){
                            // apply move id: H -> L; keep surrogate sums in sync
                            grp[L].push_back(id);
                            grp[H][pos] = grp[H].back(); grp[H].pop_back();
                            sum[H] -= w_hat[id];
                            sum[L] += w_hat[id];
                            any=true;
                            break;
                        }
                    }
                }
                if(!any) break;
            }
        }
    }

    // Rebuild final assignment from groups after interactive refinement
    fill(ans.begin(), ans.end(), 0);
    for(int g=0; g<D_; ++g) for(int id: grp[g]) ans[id]=g;

    // Consume any remaining queries with safe 1v1 dummies
    if(N_>=2){
        int a=0, b=1;
        while(q_used < Q_){
            ask({a},{b});
            b = (b+1)%N_;
            if(b==a) b=(b+1)%N_;
        }
    }

    for(int i=0;i<N_;++i){
        if(i) cout << ' ';
        cout << ans[i];
    }
    cout << '\n';
    return 0;
}
# EVOLVE-BLOCK-END