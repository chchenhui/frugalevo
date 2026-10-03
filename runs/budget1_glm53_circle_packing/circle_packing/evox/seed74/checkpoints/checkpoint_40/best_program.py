# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Hexagonal lattice packing: 6 staggered rows with widths 5,4,5,4,5,3 (=26).
    # All circles share radius r. Vertical constraint is binding:
    #   2r + 5*(sqrt(3)*r) = 1  =>  r = 1/(2 + 5*sqrt(3)) ~ 0.09381
    # Adjacent rows are staggered horizontally by r so that neighbor
    # center-to-center distances are exactly 2r (hexagonal contacts).
    # Sum of radii = 26 * r ~ 2.439.
    # Approach: seed with a hexagonal-lattice layout (guaranteed feasible,
    # sum = 26*r ~ 2.439), then improve by hill-climbing the centers.
    # For fixed centers, the optimal radii come from a linear program:
    #   maximize sum(r) s.t. r_i + r_j <= dist(i,j), 0 <= r_i <= wall cap_i.
    # Random single-center perturbations are accepted only when the LP
    # objective improves, so the result is always valid and never worse
    # than the seed. This escapes the equal-radius hexagonal optimum by
    # letting circles grow unevenly into wall gaps.
    n = 26

    def hex_seed(row_widths):
        """Staggered hexagonal-lattice seed: rows of the given widths with
        uniform radius r = 1/(2 + (rows-1)*sqrt(3)) so it fits vertically;
        adjacent rows are staggered by r for hexagonal contacts."""
        C = np.zeros((n, 2))
        nr = len(row_widths)
        r = 1.0 / (2.0 + (nr - 1) * np.sqrt(3.0))
        dy = np.sqrt(3.0) * r
        idx = 0
        for row, w in enumerate(row_widths):
            y = r + row * dy
            # offset parity keeps each row centered and staggered
            c = w - 1 if (w - 1) % 2 == row % 2 else w
            for k in range(w):
                C[idx] = [0.5 + (2 * k - c) * r, y]
                idx += 1
        return C, r

    def corner_seed(a, inner_rows):
        """Seed with 4 large corner circles of radius a plus an interior
        staggered hex pattern of the given row widths. Corner circles sit
        at (a,a), (1-a,a), (a,1-a), (1-a,1-a); the interior block is
        centered in the remaining square of side 1-2a and staggered so
        adjacent rows touch hexagonally. LP-based growth then lets the
        corner circles dominate, matching the structure of good n=26
        packings that uniform hex seeds cannot reach."""
        C = np.zeros((n, 2))
        C[0] = [a, a]
        C[1] = [1.0 - a, a]
        C[2] = [a, 1.0 - a]
        C[3] = [1.0 - a, 1.0 - a]
        m = n - 4
        w = max(inner_rows)
        nr = len(inner_rows)
        # Interior radius fits vertically in (1-2a) and horizontally in
        # (1-2a) for the widest row: use the smaller of the two limits.
        r_v = (1.0 - 2.0 * a) / (2.0 + (nr - 1) * np.sqrt(3.0))
        r_h = (1.0 - 2.0 * a) / (2.0 * w)
        r = min(r_v, r_h)
        dy = np.sqrt(3.0) * r
        idx = 4
        y0 = 0.5 - (nr - 1) * dy / 2.0
        for row, ww in enumerate(inner_rows):
            y = y0 + row * dy
            c = ww - 1 if (ww - 1) % 2 == row % 2 else ww
            for k in range(ww):
                C[idx] = [0.5 + (2 * k - c) * r, y]
                idx += 1
        return C

    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix
    except Exception:
        C, r = hex_seed([5, 4, 5, 4, 5, 3])
        return C, np.full(n, r), float(n * r)

    # Constant upper-triangular pair index lists.
    ii, jj = np.triu_indices(n, 1)

    def solve_radii(C):
        """LP: max sum(r) s.t. r_i + r_j <= d_ij, 0 <= r_i <= wall cap_i.
        Pairs with d_ij >= cap_i + cap_j can never bind (the bounds already
        imply r_i + r_j <= d_ij), so they are dropped exactly -- the LP stays
        tiny and fast. r_i = 0 is always feasible."""
        d = np.hypot(C[ii, 0] - C[jj, 0], C[ii, 1] - C[jj, 1])
        caps = np.minimum(np.minimum(C[:, 0], C[:, 1]),
                          np.minimum(1.0 - C[:, 0], 1.0 - C[:, 1]))
        keep = d < (caps[ii] + caps[jj])
        ik, jk, dk = ii[keep], jj[keep], d[keep]
        m = len(dk)
        if m == 0:
            return caps, float(caps.sum())
        rows = np.concatenate([np.arange(m), np.arange(m)])
        cols = np.concatenate([ik, jk])
        A = csr_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
        res = linprog(c=-np.ones(n), A_ub=A, b_ub=dk,
                      bounds=np.stack([np.zeros(n), caps], axis=1),
                      method="highs")
        if not res.success:
            return None, -np.inf
        return res.x, -res.fun

    import time
    t0 = time.time()
    deadline = t0 + 330.0  # wall-clock budget, inside the 360s limit
    rng = np.random.default_rng(12345)

    def climb(C, sub_deadline):
        """Monotone hill-climb of centers under the LP objective: accept a
        perturbed configuration only if the optimal-radii sum improves.
        On stagnation, reheat the step size and shake 3 circles at once."""
        rad, s = solve_radii(C)
        if rad is None:
            return -np.inf, C, None
        cur_sum, cur_rad = s, rad
        scale = 0.03
        it = 0
        stuck = 0
        while time.time() < sub_deadline:
            it += 1
            if it % 150 == 0:
                scale = max(scale * 0.9, 0.004)
            if stuck > 250:
                # Reheat + multi-circle shake to escape local optima.
                stuck = 0
                scale = min(scale * 2.5, 0.03)
                C2 = C.copy()
                for i in rng.choice(n, size=3, replace=False):
                    C2[i] += rng.normal(0.0, scale, 2)
                    C2[i] = np.clip(C2[i], 0.02, 0.98)
            else:
                C2 = C.copy()
                i = rng.integers(n)
                C2[i] += rng.normal(0.0, scale, 2)
                C2[i] = np.clip(C2[i], 0.02, 0.98)
            # Every 8th move: contact-relief move. Find circle pairs whose
            # constraint r_i + r_j <= d_ij is tight (binding) and translate
            # the smaller-radius circle directly away from its tightest
            # contact. This is the productive direction for growing radii
            # and is still validated through the LP, so feasibility and
            # monotone improvement are preserved.
            if it % 8 == 0 and cur_rad is not None:
                C2 = C.copy()
                d = np.hypot(C[ii, 0] - C[jj, 0], C[ii, 1] - C[jj, 1])
                slack = d - (cur_rad[ii] + cur_rad[jj])
                k = int(np.argmin(slack))
                i, j = int(ii[k]), int(jj[k])
                mv = (cur_rad[i] < cur_rad[j]) and i or j
                other = j if mv == i else i
                dirv = C[mv] - C[other]
                nrm = np.hypot(dirv[0], dirv[1])
                if nrm > 1e-12:
                    C2[mv] += (dirv / nrm) * scale
                    C2[mv] = np.clip(C2[mv], 0.02, 0.98)
                rad2, s2 = solve_radii(C2)
                if rad2 is not None and s2 > cur_sum + 1e-9:
                    C, cur_sum, cur_rad = C2, s2, rad2
                    stuck = 0
                else:
                    stuck += 1
                continue
            rad2, s2 = solve_radii(C2)
            if rad2 is not None and s2 > cur_sum + 1e-9:
                C, cur_sum, cur_rad = C2, s2, rad2
                stuck = 0
            else:
                stuck += 1
        return cur_sum, C, cur_rad

    # Multi-start over hexagonal row patterns (each sums to 26) and
    # corner-circle seeds (4 big corner circles + 22 interior hex circles),
    # which give the climb access to the corner-dominated basin that the
    # best known n=26 packings live in.
    hex_seeds = [[5, 4, 5, 4, 5, 3],
                 [4, 5, 4, 5, 4, 4],
                 [5, 5, 4, 4, 5, 3],
                 [3, 5, 5, 4, 5, 4],
                 [5, 4, 4, 5, 4, 4],
                 [4, 4, 4, 4, 4, 3, 3]]
    seed_fns = [(lambda sw=sw: hex_seed(sw)[0]) for sw in hex_seeds]
    for a in (0.13, 0.15, 0.17):
        for rows in ([4, 5, 4, 5, 4], [5, 4, 5, 4, 4]):
            if sum(rows) == n - 4:
                seed_fns.append(lambda a=a, rows=rows: corner_seed(a, rows))
    n_seeds = len(seed_fns)
    best_sum, best_C, best_rad = -np.inf, None, None
    for si, fn in enumerate(seed_fns):
        sub_dl = min(t0 + (si + 1) * (deadline - t0) / n_seeds, deadline)
        if time.time() >= sub_dl:
            break
        C0 = fn()
        s, C, rad = climb(C0, sub_dl)
        if s > best_sum:
            best_sum, best_C, best_rad = s, C, rad

    # Final polish phase from the best configuration found.
    if best_C is not None:
        s, C, rad = climb(best_C, deadline)
        if s > best_sum:
            best_sum, best_C, best_rad = s, C, rad

    # Extra random restarts from the best packing (shake all centers) to
    # escape the local optimum, using any remaining time.
    while time.time() < deadline - 2.0 and best_C is not None:
        C0 = best_C + rng.normal(0.0, 0.05, (n, 2))
        C0 = np.clip(C0, 0.02, 0.98)
        s, C, rad = climb(C0, deadline)
        if s > best_sum:
            best_sum, best_C, best_rad = s, C, rad

    if best_C is None:
        C, r = hex_seed([5, 4, 5, 4, 5, 3])
        return C, np.full(n, r), float(n * r)

    return best_C, best_rad, float(best_sum)





# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
