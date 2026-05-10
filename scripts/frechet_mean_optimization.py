#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar, lsq_linear
from scipy.interpolate import BSpline, PPoly
import cvxopt
from cvxopt import matrix, solvers, spmatrix
from logzero import logger

# Suppress cvxopt output
solvers.options['show_progress'] = False

# ===
# SEIR Model & Simulation
# ===
def seir_model(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]

def simulate_SEIR(beta, sigma, gamma, N, E0, I0, T, t_eval):
    S0 = N - E0 - I0
    R0 = 0
    y0 = [S0, E0, I0, R0]
    sol = solve_ivp(seir_model, [0, T], y0, args=(beta, sigma, gamma, N), t_eval=t_eval)
    return sol.t, sol.y

# ===
# Trajectory Utilities
# ===
def shift_trajectory(t, y, delta):
    t_shift = t + delta
    return np.interp(t_shift, t, y, left=y[0], right=y[-1])

# ===
# B-spline Basis Construction
# ===
def build_bspline_basis(t, K, degree=3):
    t0, t_end = t[0], t[-1]
    num_internal = K - degree - 1
    if num_internal > 0:
        internal_knots = np.linspace(t0, t_end, num_internal + 2)[1:-1]
    else:
        internal_knots = np.array([])
    knots = np.concatenate(
        (np.repeat(t0, degree + 1), internal_knots, np.repeat(t_end, degree + 1))
    )
    return knots

def get_design_matrices(t_eval, knots, K, degree=3):
    n = len(t_eval)
    B = np.zeros((n, K))
    B_prime = np.zeros((n, K))
    Phi = np.zeros((n, K))
    
    t0 = t_eval[0]
    for k in range(K):
        coeff = np.zeros(K)
        coeff[k] = 1.0
        spline = BSpline(knots, coeff, degree, extrapolate=False)
        B[:, k] = spline(t_eval)
        B_prime[:, k] = spline.derivative(1)(t_eval)
        F = spline.antiderivative(1)
        Phi[:, k] = F(t_eval) - F(t0)
        
    return B, B_prime, Phi

def integrate_ppoly_product(pp1, pp2):
    x = pp1.x
    if not np.array_equal(x, pp2.x):
        return 0.0 
    c1, c2 = pp1.c, pp2.c
    val = 0.0
    for i in range(c1.shape[1]):
        h = x[i + 1] - x[i]
        if h == 0: continue
        a1 = c1[:, i][::-1]
        a2 = c2[:, i][::-1]
        prod = np.convolve(a1, a2)
        r = np.arange(prod.size)
        val += np.sum(prod * h ** (r + 1) / (r + 1))
    return float(val)

def compute_gram_matrix_H1(knots, K, degree=3, rho=1):
    pp = []
    ppd = []
    for k in range(K):
        coeff = np.zeros(K)
        coeff[k] = 1.0
        spline = BSpline(knots, coeff, degree, extrapolate=False)
        ppk = PPoly.from_spline((spline.t, spline.c, spline.k))
        pp.append(ppk)
        ppd.append(ppk.derivative(1))
    
    G = np.zeros((K, K))
    for k in range(K):
        for l in range(k, K):
            val = integrate_ppoly_product(pp[k], pp[l]) + rho * integrate_ppoly_product(ppd[k], ppd[l])
            G[k, l] = val
            G[l, k] = val
    return G

def estimate_coefficients(y, B):
    c, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
    return c

# ===
# Joint Optimization: Two-Level Algorithm
# ===
class JointOptimizer:
    def __init__(self, t_eval, N, K, degree, knots, G_H1):
        self.t_eval = t_eval
        self.N = N
        self.K = K
        self.M = len(t_eval)
        
        # Precompute Matrices
        self.B, self.B_prime, self.Phi = get_design_matrices(t_eval, knots, K, degree)
        self.G_H1 = G_H1
        
        # Joint Gram Matrix G^Y
        self.G_Y = np.block([
            [self.G_H1, np.zeros((K, K))],
            [np.zeros((K, K)), self.G_H1]
        ])
        
        # Cholesky decomposition of G^Y for SOCP
        try:
            self.L_mat = np.linalg.cholesky(self.G_Y).T
        except np.linalg.LinAlgError:
            self.L_mat = np.linalg.cholesky(self.G_Y + 1e-9 * np.eye(2*K)).T

    def _solve_projection(self, G_np, h_np, A_np, b_np, sample_coeffs, q, irls_iter=10, irls_tol=1e-3):
        """
        Generic solver for minimizing sum D(c, c_j)^q under linear constraints.
        Supports QP (q=2), and IRLS-QP (others).
        """
        J = len(sample_coeffs)
        c_mean = np.mean(sample_coeffs, axis=0)
        
        G_cvx = matrix(G_np)
        h_cvx = matrix(h_np)
        if A_np is not None and A_np.shape[0] > 0:
            A_cvx = matrix(A_np)
            b_cvx = matrix(b_np)
        else:
            A_cvx = None
            b_cvx = None

        c_sol = None
        duals = {}
        
        try:
            # --- QP (q=2) ---
            if q == 2:
                P = matrix(2 * self.G_Y)
                q_vec = matrix(-2 * self.G_Y @ c_mean)
                if A_cvx is not None:
                    res = solvers.qp(P, q_vec, G_cvx, h_cvx, A_cvx, b_cvx)
                else:
                    res = solvers.qp(P, q_vec, G_cvx, h_cvx)
                
                c_sol = np.array(res['x']).flatten()
                
                if 'y' in res and res['y'] is not None:
                    nu = np.array(res['y']).flatten()
                else:
                    nu = np.array([])
                
                z_all = np.array(res['z']).flatten() 
                # Assuming standard constraint ordering: E>=0, I>=0, Pop>=0
                # Each block size M_ineq
                M_ineq = len(h_np) // 3
                if len(z_all) >= 3 * M_ineq:
                    lambda_pop = z_all[2*M_ineq : 3*M_ineq]
                else:
                    lambda_pop = np.zeros(M_ineq)
                    
                duals = {'nu': nu, 'lambda': lambda_pop}

            # --- IRLS-QP (q != 2) ---
            else:
                c_curr = c_mean
                c_prev = c_mean
                epsilon = 1e-6
                for _ in range(irls_iter):
                    weights = []
                    for j in range(J):
                        diff = c_curr - sample_coeffs[j]
                        dist2 = diff.T @ self.G_Y @ diff
                        w_j = (dist2 + epsilon**2)**(q/2.0 - 1.0)
                        weights.append(w_j)
                    sum_w = sum(weights)
                    
                    weighted_sum_c = np.zeros_like(c_curr)
                    for j in range(J):
                        weighted_sum_c += weights[j] * sample_coeffs[j]
                        
                    P = matrix(2 * sum_w * self.G_Y)
                    q_vec = matrix(-2 * self.G_Y @ weighted_sum_c)
                    
                    if A_cvx is not None:
                        res = solvers.qp(P, q_vec, G_cvx, h_cvx, A_cvx, b_cvx)
                    else:
                        res = solvers.qp(P, q_vec, G_cvx, h_cvx)
                        
                    c_curr = np.array(res['x']).flatten()
                    
                    diff_norm = np.linalg.norm(c_curr - c_prev)
                    norm_prev = np.linalg.norm(c_prev)
                    rel_error = diff_norm / (norm_prev + 1e-9)

                    if rel_error < irls_tol:
                        break
                
                c_sol = c_curr
                if 'y' in res and res['y'] is not None:
                    nu = np.array(res['y']).flatten()
                else:
                    nu = np.array([])

                z_all = np.array(res['z']).flatten()
                M_ineq = len(h_np) // 3
                if len(z_all) >= 3 * M_ineq:
                    lambda_pop = z_all[2*M_ineq : 3*M_ineq]
                else:
                    lambda_pop = np.zeros(M_ineq)
                    
                duals = {'nu': nu, 'lambda': lambda_pop}
                
        except Exception as e:
            logger.error(f"Solver Error: {e}")
            raise e

        return c_sol, duals

    def _build_full_constraints(self, sigma, gamma):
        """
        Constructs constraints for the full problem including ODE equalities.
        """
        K = self.K
        M = self.M
        N = self.N
        
        # --- 1. Equality Constraints (ODE) ---
        step_eq = max(1, int(M / (K * 0.8))) 
        idx_eq = np.arange(0, M, step_eq)
        if len(idx_eq) >= 2 * K:
            idx_eq = np.linspace(0, M-1, int(1.5*K), dtype=int)
        M_eq = len(idx_eq)
        
        B_eq = self.B[idx_eq]
        B_prime_eq = self.B_prime[idx_eq]
        
        # -sigma B c_E + (B' + gamma B) c_I = 0
        row_eq = np.hstack([-sigma * B_eq, B_prime_eq + gamma * B_eq])
        A_np = row_eq
        b_np = np.zeros(M_eq)
        
        # --- 2. Inequality Constraints ---
        step_ineq = 2
        idx_ineq = np.arange(0, M, step_ineq)
        M_ineq = len(idx_ineq)
        
        B_ineq = self.B[idx_ineq]
        Phi_ineq = self.Phi[idx_ineq]
        zeros_ineq_K = np.zeros((M_ineq, K))
        
        G_ineq_list = []
        h_ineq_list = []
        
        # (1) -E <= 0
        G_ineq_list.append(np.hstack([-B_ineq, zeros_ineq_K]))
        h_ineq_list.append(np.zeros(M_ineq))
        
        # (2) -I <= 0
        G_ineq_list.append(np.hstack([zeros_ineq_K, -B_ineq]))
        h_ineq_list.append(np.zeros(M_ineq))
        
        # (3) N - E - I - gamma * Phi * I >= 0  => E + I + gamma * Phi * I <= N
        # Matrix: [B, B + gamma*Phi]
        row3 = np.hstack([B_ineq, B_ineq + gamma * Phi_ineq])
        G_ineq_list.append(row3)
        h_ineq_list.append(np.full(M_ineq, N))
        
        G_np = np.vstack(G_ineq_list)
        h_np = np.hstack(h_ineq_list)
        
        return G_np, h_np, A_np, b_np, idx_eq

    def _build_reduced_constraints(self):
        """
        Constructs constraints for the initialization step (No ODE, simplified Population).
        Constraints:
          E >= 0
          I >= 0
          N - E - I >= 0
        """
        K = self.K
        M = self.M
        N = self.N
        
        # No Equality Constraints
        A_np = np.zeros((0, 2*K))
        b_np = np.zeros(0)
        
        # Inequality Constraints
        step_ineq = 2
        idx_ineq = np.arange(0, M, step_ineq)
        M_ineq = len(idx_ineq)
        
        B_ineq = self.B[idx_ineq]
        zeros_ineq_K = np.zeros((M_ineq, K))
        
        G_ineq_list = []
        h_ineq_list = []
        
        # (1) -E <= 0
        G_ineq_list.append(np.hstack([-B_ineq, zeros_ineq_K]))
        h_ineq_list.append(np.zeros(M_ineq))
        
        # (2) -I <= 0
        G_ineq_list.append(np.hstack([zeros_ineq_K, -B_ineq]))
        h_ineq_list.append(np.zeros(M_ineq))
        
        # (3) E + I <= N
        row3 = np.hstack([B_ineq, B_ineq])
        G_ineq_list.append(row3)
        h_ineq_list.append(np.full(M_ineq, N))
        
        G_np = np.vstack(G_ineq_list)
        h_np = np.hstack(h_ineq_list)
        
        return G_np, h_np, A_np, b_np

    def solve_inner(self, sigma, gamma, sample_coeffs, q, irls_iter=10):
        G_np, h_np, A_np, b_np, idx_eq = self._build_full_constraints(sigma, gamma)
        c_sol, duals = self._solve_projection(G_np, h_np, A_np, b_np, sample_coeffs, q, irls_iter)
        duals['idx_eq'] = idx_eq
        return c_sol, duals

    def solve_reduced_init(self, sample_coeffs, q):
        """
        Solves the reduced optimization problem for initialization.
        """
        G_np, h_np, A_np, b_np = self._build_reduced_constraints()
        c_sol, _ = self._solve_projection(G_np, h_np, A_np, b_np, sample_coeffs, q)
        return c_sol

    def estimate_ode_params(self, c_init):
        """
        Estimates sigma and gamma using least squares on the differential equation:
        dI/dt = sigma * E - gamma * I
        Subject to sigma > 0, gamma > 0.
        Minimizes integral over [0, T] of squared residual.
        """
        c_E = c_init[:self.K]
        c_I = c_init[self.K:]
        
        # Evaluate basis vectors on the full grid
        E_vec = self.B @ c_E
        I_vec = self.B @ c_I
        dI_vec = self.B_prime @ c_I
        
        # We want to minimize || dI - (sigma * E - gamma * I) ||^2
        # Target: dI
        # Feature 1: E (coeff sigma)
        # Feature 2: -I (coeff gamma)
        
        X = np.vstack([E_vec, -I_vec]).T
        y = dI_vec
        
        # Constrained least squares: sigma > 0, gamma > 0
        res = lsq_linear(X, y, bounds=(0, np.inf))
        
        sigma_est, gamma_est = res.x
        
        # Ensure non-zero to avoid division errors later
        sigma_est = max(sigma_est, 1e-4)
        gamma_est = max(gamma_est, 1e-4)
        
        return sigma_est, gamma_est

    def profile_loss(self, params, sample_coeffs, q):
        sigma, gamma = params
        if sigma <= 0 or gamma <= 0: return 1e6, np.zeros(2)

        try:
            c_hat, duals = self.solve_inner(sigma, gamma, sample_coeffs, q)
        except Exception:
            return 1e6, np.zeros(2)
        
        J = len(sample_coeffs)
        loss = 0.0
        for j in range(J):
            diff = c_hat - sample_coeffs[j]
            d2 = diff.T @ self.G_Y @ diff
            if q == 2:
                loss += d2
            elif q == 1:
                loss += np.sqrt(d2 + 1e-12)
            else:
                loss += (d2 + 1e-12)**(q/2.0)
        loss /= J
        
        nu = duals['nu']
        lam = duals['lambda']
        idx_eq = duals['idx_eq']
        
        B_eq = self.B[idx_eq]
        
        step_ineq = 2
        idx_ineq = np.arange(0, self.M, step_ineq)
        Phi_ineq = self.Phi[idx_ineq]
        
        c_E = c_hat[:self.K]
        c_I = c_hat[self.K:]
        
        grad_sigma = - nu @ (B_eq @ c_E)
        grad_gamma = nu @ (B_eq @ c_I) + lam @ (Phi_ineq @ c_I)
        
        return loss, np.array([grad_sigma, grad_gamma])


# ===
# Shift Optimization (1D)
# ===
def shift_objective(delta, c_ref, traj, t_eval, B, G_Y, q):
    if "E_orig" in traj:
        E_raw = traj["E_orig"]
        I_raw = traj["I_orig"]
    else:
        E_raw = traj["E_aligned"]
        I_raw = traj["I_aligned"]
    
    E_shift = shift_trajectory(t_eval, E_raw, delta)
    I_shift = shift_trajectory(t_eval, I_raw, delta)
    c_E = estimate_coefficients(E_shift, B)
    c_I = estimate_coefficients(I_shift, B)
    c_samp = np.concatenate([c_E, c_I])
    
    diff = c_ref - c_samp
    d2 = diff.T @ G_Y @ diff
    
    if q == 1:
        return np.sqrt(d2 + 1e-12)
    elif q == 2:
        return d2
    else:
        return (d2 + 1e-12)**(q/2.0)

def update_shifts(c_ref, trajectories, t_eval, B, G_Y, q, current_deltas, delta_max):
    from scipy.optimize import brentq

    tilde_deltas = np.zeros_like(current_deltas)
    for j, traj in enumerate(trajectories):
        res = minimize_scalar(
            lambda d: shift_objective(d, c_ref, traj, t_eval, B, G_Y, q),
            bounds=(current_deltas[j] - 0.5, current_deltas[j] + 0.5),
            method='bounded'
        )
        tilde_deltas[j] = res.x

    def projection_residual(mu):
        return np.sum(np.clip(tilde_deltas - mu, -delta_max, delta_max))

    lower_bound = np.min(tilde_deltas) - delta_max - 1.0
    upper_bound = np.max(tilde_deltas) + delta_max + 1.0

    mu_star = brentq(projection_residual, lower_bound, upper_bound)

    new_deltas = np.clip(tilde_deltas - mu_star, -delta_max, delta_max)
    return new_deltas

# ===
# Helper: PPoly Multiplication
# ===
def multiply_ppoly(pp1, pp2):
    """
    Multiplies two PPoly objects derived from the same breakpoints.
    Returns a new PPoly object representing the product.
    """
    # Check compatibility of breakpoints
    if pp1.x.shape != pp2.x.shape or not np.allclose(pp1.x, pp2.x):
        raise ValueError("PPoly objects must have the same breakpoints to be multiplied directly.")

    # pp.c has shape (order, n_intervals). order = degree + 1.
    k1, m = pp1.c.shape
    k2 = pp2.c.shape[0]
    
    # The product of polynomials of degree d1 and d2 has degree d1+d2.
    # New order = (k1 - 1) + (k2 - 1) + 1 = k1 + k2 - 1
    new_k = k1 + k2 - 1
    new_c = np.zeros((new_k, m))
    
    for i in range(m):
        # Convolution of coefficients corresponds to polynomial multiplication
        # (Scipy stores coeffs in descending order of power, consistent with np.convolve)
        new_c[:, i] = np.convolve(pp1.c[:, i], pp2.c[:, i])
        
    return PPoly(new_c, pp1.x)


# ===
# Main Pipeline
# ===
def run_SEIR_frechet_pipeline(
    trajectories,
    t_eval,
    N,
    K=30,
    degree=3,
    q_values=[1, 1.5, 2],
    w_E=1.0,
    w_I=1.0,
    max_outer_iter=20,
    rho=1.0,
    delta_max=None,
    lambda_shift=0.0,
    tol=1e-3,
):
    results = {}
    knots = build_bspline_basis(t_eval, K, degree)
    B, _, _ = get_design_matrices(t_eval, knots, K, degree)
    G_H1 = compute_gram_matrix_H1(knots, K, degree, rho)
    
    if delta_max is None:
        delta_max = 0.2 * (t_eval[-1] - t_eval[0])

    optimizer = JointOptimizer(t_eval, N, K, degree, knots, G_H1)

    for i in range(len(q_values)):
        q = q_values[i]
        if q == 1:
            q += 1e-4
        logger.info(f"Starting optimization for q={q_values[i]}")
        
        # Initial Alignment by Peak
        I_list = []
        for traj in trajectories:
            I_source = traj["I_orig"] if "I_orig" in traj else traj["I_aligned"]
            I_list.append(I_source)
        peak_times = np.array([t_eval[np.argmax(I)] for I in I_list])
        deltas = peak_times - np.median(peak_times)
        deltas -= np.mean(deltas)
        deltas = np.clip(deltas, -delta_max, delta_max)

        # === Initialization Step (Reduced Optimization) ===
        # 1. Compute basis coefficients for initially shifted trajectories
        sample_coeffs_init = []
        for j, traj in enumerate(trajectories):
            if "E_orig" in traj:
                E_d = shift_trajectory(t_eval, traj["E_orig"], deltas[j])
                I_d = shift_trajectory(t_eval, traj["I_orig"], deltas[j])
            else:
                E_d = shift_trajectory(t_eval, traj["E_aligned"], deltas[j])
                I_d = shift_trajectory(t_eval, traj["I_aligned"], deltas[j])
            
            c_E = estimate_coefficients(E_d, B)
            c_I = estimate_coefficients(I_d, B)
            sample_coeffs_init.append(np.concatenate([c_E, c_I]))
        
        # 2. Solve Reduced Problem for c_hat (No ODE constraints)
        logger.info("Solving reduced optimization for initialization...")
        c_ref_init = optimizer.solve_reduced_init(sample_coeffs_init, q)
        
        # 3. Estimate sigma, gamma from c_hat via Least Squares
        sigma_curr, gamma_curr = optimizer.estimate_ode_params(c_ref_init)
        c_ref = c_ref_init
        
        logger.info(f"Initialized params: sigma={sigma_curr:.4f}, gamma={gamma_curr:.4f}")

        # === Main Alternating Minimization Loop ===
        for it in range(max_outer_iter):
            # 1. Update Coefficients with current shifts
            sample_coeffs = []
            for j, traj in enumerate(trajectories):
                if "E_orig" in traj:
                    E_d = shift_trajectory(t_eval, traj["E_orig"], deltas[j])
                    I_d = shift_trajectory(t_eval, traj["I_orig"], deltas[j])
                else:
                    E_d = shift_trajectory(t_eval, traj["E_aligned"], deltas[j])
                    I_d = shift_trajectory(t_eval, traj["I_aligned"], deltas[j])
                
                c_E = estimate_coefficients(E_d, B)
                c_I = estimate_coefficients(I_d, B)
                sample_coeffs.append(np.concatenate([c_E, c_I]))

            # 2. Outer Profile Optimization
            res_outer = minimize(
                optimizer.profile_loss,
                x0=[sigma_curr, gamma_curr],
                args=(sample_coeffs, q),
                method='L-BFGS-B',
                jac=True,
                bounds=[(1e-3, 5.0), (1e-3, 5.0)]
            )
            
            sigma_new, gamma_new = res_outer.x
            
            c_ref_new, _ = optimizer.solve_inner(sigma_new, gamma_new, sample_coeffs, q)
            
            # 3. Update Shifts
            deltas_new = update_shifts(c_ref_new, trajectories, t_eval, B, optimizer.G_Y, q, deltas, delta_max)
            
            diff_param = (np.abs(sigma_new - sigma_curr) / (np.abs(sigma_curr) + 1e-12) + 
                          np.abs(gamma_new - gamma_curr) / (np.abs(gamma_curr) + 1e-12))
            diff_delta = np.max(np.abs(deltas_new - deltas) / (np.abs(deltas) + 1e-12))
    
            sigma_curr, gamma_curr = sigma_new, gamma_new
            c_ref = c_ref_new
            deltas = deltas_new
            
            logger.info(f"Iter {it}: sigma={sigma_curr:.4f}, gamma={gamma_curr:.4f}, diff={max(diff_param, diff_delta):.6f}")
            
            if max(diff_param, diff_delta) < tol:
                break

        # Reconstruction
        c_E_est = c_ref[:K]
        c_I_est = c_ref[K:]
        E_est = B @ c_E_est
        I_est = B @ c_I_est
        Phi_I = optimizer.Phi @ c_I_est
        R_est = gamma_curr * Phi_I
        S_est = N - E_est - I_est - R_est
        
        # Estimate Beta        
        # 1. Reconstruct PPoly objects from estimated coefficients
        spl_E = BSpline(knots, c_E_est, degree, extrapolate=False)
        spl_I = BSpline(knots, c_I_est, degree, extrapolate=False)
        pp_E = PPoly.from_spline(spl_E)
        pp_I = PPoly.from_spline(spl_I)
        
        # 2. Prepare Integration Helper
        def get_integral_0_t(pp_poly):
            """Computes definite integral int_0^t P(tau) dtau analytically."""
            anti = pp_poly.antiderivative()
            return anti(t_eval) - anti(t_eval[0])

        # 3. Calculate RHS Integral: int_0^t (S*I)/N dtau
        # S(tau) = N - E(tau) - I(tau) - R(tau)
        # R(tau) = gamma * (int_0^tau I(s) ds)
        
        # Calculate R's PPoly (Antiderivative of I)
        pp_Int_I = pp_I.antiderivative()
        val_Int_I_0 = pp_Int_I(t_eval[0]) # Constant offset for R
        
        # Expand integrand: S*I = N*I - E*I - I^2 - gamma*(Int_I * I) + gamma*const*I
        
        # Term 1: N * int(I)
        term1 = N * get_integral_0_t(pp_I)
        
        # Term 2: int(E * I) -> Use helper
        pp_ExI = multiply_ppoly(pp_E, pp_I)
        term2 = get_integral_0_t(pp_ExI)
        
        # Term 3: int(I * I) -> Use helper
        pp_IxI = multiply_ppoly(pp_I, pp_I)
        term3 = get_integral_0_t(pp_IxI)
        
        # Term 4: int(R * I)
        # R(t) = gamma * ( pp_Int_I(t) - val_Int_I_0 )
        # R * I = gamma * (pp_Int_I * I) - gamma * val_Int_I_0 * I
        
        pp_Int_IxI = multiply_ppoly(pp_Int_I, pp_I)
        term4_a = gamma_curr * get_integral_0_t(pp_Int_IxI)
        
        # For the constant term, we just integrate I and multiply by scalar
        term4_b = gamma_curr * val_Int_I_0 * get_integral_0_t(pp_I)
        
        term4 = term4_a - term4_b
        
        # Combine all terms
        rhs_integral = (term1 - term2 - term3 - term4) / N
        
        # 4. Calculate LHS: E(t) - E(0) + sigma * int_0^t E(tau) dtau
        # int_0^t E is exactly Phi @ c_E
        Phi_E = optimizer.Phi @ c_E_est
        lhs = (E_est - E_est[0]) + sigma_curr * Phi_E
        
        # 5. Solve for Beta
        numerator = np.dot(rhs_integral, lhs)
        denominator = np.dot(rhs_integral, rhs_integral)
        
        if denominator < 1e-12:
            beta_est = 0.0
        else:
            beta_est = max(0.0, numerator / denominator)

        results[q_values[i]] = {
            "E": E_est, "I": I_est, "S": S_est, "R": R_est,
            "sigma": sigma_curr, "gamma": gamma_curr, "beta": beta_est,
            "x": np.concatenate([c_ref, [sigma_curr, gamma_curr]]),
            "delta": deltas,
            "diffs": [] 
        }

    return results
