# Circle Packing Strategies for n=26 in a Unit Square

## Key Insight
Naive geometric placement (rings, grids) gives sum_radii ~ 1.0.
Using numerical optimization (scipy.optimize) with proper constraint formulation
can push sum_radii above 2.5.

## Why Optimization Works Better Than Manual Placement

Manual placement fixes circle positions, then computes maximum radii.
This leaves gaps because positions aren't optimized for the radii they produce.

**Joint optimization** treats both positions (x,y for each circle) AND radii
as decision variables, optimizing them simultaneously. This is the key insight.

Decision vector: [x0, y0, x1, y1, ..., x25, y25, r0, r1, ..., r25]
Total variables: 26*2 + 26 = 78

## Constraint Formulation

1. **Non-overlap**: For every pair (i,j): distance(center_i, center_j) >= r_i + r_j
2. **Boundary**: For every circle i: x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
3. **Positive radii**: r_i > 0 for all i (use bounds, not constraints)

## Recommended Solver

scipy.optimize.minimize with method="SLSQP":
- Handles inequality constraints natively
- Works with bounds on variables
- Good for smooth, continuous problems like circle packing
- Sensitive to initial guess — use multiple starts or a good heuristic

## Initial Guess Strategy

A hexagonal grid initial guess works well:
- Place circles on offset rows (hex pattern)
- Start with equal small radii (e.g., 0.05)
- Let the optimizer adjust both positions and radii

## Performance Tips

- Set maxiter=1000 or higher for 26 circles
- Use ftol=1e-8 or smaller for precise solutions
- Radii bounds: (0.01, 0.2) is a reasonable range for n=26
- The objective is -sum(radii) (minimize negative to maximize)
