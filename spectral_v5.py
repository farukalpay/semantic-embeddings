#!/usr/bin/env python
"""
Spectral-Chain Laboratory (v5.0 – Unified Projectors + Advanced Novelty)
=========================================================================

This version merges the powerful spectral projector framework of v4.0 with the
advanced novelty chain algorithm from v4.2, creating a single, comprehensive tool.

KEY FEATURES
────────────
1.  **Advanced Novelty Chain Algorithm (from v4.2):**
    *   **γ-Redundancy Penalty:** Promotes diversity by penalizing candidates
        that are too similar to existing chain members. Tunable with `--gamma`.
    *   **Adaptive ε-stop:** The early-exit threshold tightens over time
        (ε₀/√t) to curb late-stage noise and improve chain quality.
    *   **Geometric Candidate Pool:** Uses a sharper, similarity-based
        candidate pool (`--k_pool`, `--theta_pool`) for more relevant fallbacks.
    *   **Stronger Connectivity Default:** Uses a more effective default for
        `--lambda_conn` (0.30).

2.  **Comprehensive Spectral Projectors (from v4.0):**
    *   A full suite of methods to transform the embedding space before
        generating chains, including `smooth`, `iso`, `poly`, `whiten`, and `combo`.
    *   These projectors allow fine-grained control over the semantic properties
        of the vector space.

3.  **Unified Framework:**
    *   Full support for Johnson-Lindenstrauss (JL) dimension reduction with
        a fast, cached FJLT implementation.
    *   Includes legacy chain strategies (`maxvol`, `greedy`) for comparison.
    *   A rich command-line interface and an interactive console for
        experimentation.
"""
from __future__ import annotations

import argparse, sys, math, os
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, Sequence, Callable, Union

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial import chebyshev as C, polynomial as Pm
from gensim.models import KeyedVectors, Word2Vec
from scipy.linalg import hadamard, eigh

# ──────────────────────────────────────────────────────────────────────────────
# Type Aliases
# ──────────────────────────────────────────────────────────────────────────────
Vector    = NDArray[np.float64]
Matrix    = NDArray[np.float64]
Transform = Union[Callable[[Vector], Vector], Matrix, None]
PRNG      = np.random.Generator

# ──────────────────────────────────────────────────────────────────────────────
# 0. Johnson–Lindenstrauss Utilities
# ──────────────────────────────────────────────────────────────────────────────
def compute_epsilon_min(d: int, delta: float) -> float:
    """ε_min(d,δ) = sqrt(4 ln(d/δ) / d)."""
    return math.sqrt(4.0 * math.log(d / delta) / d)

def compute_jl_dim(d: int, eps: float, delta: float | None = None) -> int:
    """Return the capped target dimension m⋆ = min{d, ceil[4 ln(d/δ) / ε²]}."""
    if not (0. < eps < 1.):
        raise ValueError("eps ∉ (0,1)")
    delta = 1. / (d * d) if delta is None else delta
    eps_min = compute_epsilon_min(d, delta)
    if eps < eps_min:
        return d
    m_needed = math.ceil(4. * math.log(d / delta) / (eps ** 2))
    return min(d, m_needed)

def make_fjlt_transform(m: int, d: int, rng: PRNG) -> Callable[[Matrix], Matrix]:
    """Fast JL transform ℝᵈ→ℝᵐ with O(d log d) mat-vecs."""
    d_pad = 1 << ((d - 1).bit_length())
    H = hadamard(d_pad).astype(np.float32)
    D = rng.choice([-1., 1.], d_pad).astype(np.float32)
    Π_idx = rng.permutation(d_pad)
    P_idx = rng.choice(d_pad, size=m, replace=False)
    scale = math.sqrt(d_pad / m)

    def fjlt(X: Matrix) -> Matrix:
        if X.ndim == 1: X = X[None, :]
        n_vec, d_in = X.shape
        if d_in != d: raise ValueError(f"Expected dim {d}, got {d_in}")
        X_pad = np.zeros((n_vec, d_pad), dtype=np.float32)
        X_pad[:, :d] = X.astype(np.float32)
        X_pad = X_pad[:, Π_idx]
        X_pad *= D
        X_pad = (H @ X_pad.T) / math.sqrt(d_pad)
        return (scale * X_pad[P_idx, :]).T
    return fjlt

# ──────────────────────────────────────────────────────────────────────────────
# 1. Sparse Chebyshev Helpers
# ──────────────────────────────────────────────────────────────────────────────
def estimate_lambda_max(A: Matrix, it: int = 100) -> float:
    """Power-iteration upper bound on λ_max(A)."""
    v = np.random.default_rng().standard_normal(A.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(it):
        v = A @ v
        n = np.linalg.norm(v)
        v /= n
    return float(n)

def cheb_poly_coeffs(f, deg: int, lam_max: float, oversample: int = 501) -> Vector:
    xs = np.linspace(0., lam_max, oversample)
    ys = f(xs)
    cheb = C.Chebyshev.fit(xs, ys, deg, domain=[0., lam_max])
    poly = cheb.convert(kind=Pm.Polynomial)
    return poly.coef

def sparsify_coeffs(coeffs: Vector, d: int, eps: float) -> Vector:
    k_max = int(np.ceil(d * math.log2(1. / eps)))
    if k_max >= len(coeffs): return coeffs
    keep = np.argsort(np.abs(coeffs))[::-1][:k_max]
    mask = np.zeros_like(coeffs, dtype=bool); mask[keep] = True
    out = coeffs.copy(); out[~mask] = 0.
    return out

def polynomial_matrix(A: Matrix, coeffs: Vector) -> Matrix:
    P = coeffs[-1] * np.eye(A.shape[0])
    for c in reversed(coeffs[:-1]):
        P = A @ P + c * np.eye(A.shape[0])
    return P

# ──────────────────────────────────────────────────────────────────────────────
# 2. Model Loading/Training
# ──────────────────────────────────────────────────────────────────────────────
def load_or_train_kv(*, pretrained_path: str | None = None,
                     corpus_iter=None, **w2v_kwargs) -> KeyedVectors:
    if pretrained_path:
        print(f"[INFO] Loading vectors from {Path(pretrained_path).name!r}")
        kv = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        print(f"[INFO] Loaded {len(kv):,} embeddings (d={kv.vector_size})")
        return kv
    if corpus_iter is None: raise ValueError("Either --bin or a corpus iterator is required")
    print("[INFO] Training Word2Vec model...")
    w2v = Word2Vec(corpus_iter, **w2v_kwargs)
    return w2v.wv

# ──────────────────────────────────────────────────────────────────────────────
# 3. Eigendecomposition & Projectors
# ──────────────────────────────────────────────────────────────────────────────
def eig_from_kv(kv: KeyedVectors):
    E = kv.vectors.astype(np.float64)
    mu = E.mean(0)
    E_ = E - mu
    Sigma = np.cov(E_, rowvar=False)
    lam, U = eigh(Sigma)
    lam, U = lam[::-1], U[:, ::-1]
    Sigma_sqrt = (U * np.sqrt(np.maximum(lam, 0))) @ U.T
    Sigma_inv_sqrt = (U * (np.maximum(lam, 1e-12) ** -0.5)) @ U.T
    return mu, lam, U, (Sigma_sqrt, Sigma_inv_sqrt)

def smooth_projector(lam: Vector, U: Matrix, rho: float, p: float):
    def frac(theta): return ((1 - np.exp(-(lam / theta) ** p)) * lam).sum() / lam.sum()
    lo, hi = 1e-12, lam.max()
    while frac(hi) > rho: hi *= 2
    for _ in range(100):
        mid = .5 * (lo + hi)
        if abs(frac(mid) - rho) < 1e-6: break
        lo, hi = (mid, hi) if frac(mid) > rho else (lo, mid)
    theta = .5 * (lo + hi)
    w = np.exp(-(lam / theta) ** p)
    P = (U * w) @ U.T
    kept = 1 - frac(theta)
    print(f"[INFO] Smooth projector ρ={rho:.3f} → θ={theta:.4f}, kept {kept*100:.1f}% variance")
    return P, theta

def iso_projector(lam: Vector, U: Matrix, rho: float):
    def frac(beta): return ((1 - lam / (lam + beta)) * lam).sum() / lam.sum()
    lo, hi = 1e-12, lam.max()
    while frac(hi) < rho: hi *= 2
    for _ in range(100):
        mid = .5 * (lo + hi)
        if abs(frac(mid) - rho) < 1e-6: break
        lo, hi = ((mid, hi) if frac(mid) < rho else (lo, mid))
    beta = .5 * (lo + hi)
    w = lam / (lam + beta)
    P = (U * w) @ U.T
    kept = 1 - frac(beta)
    print(f"[INFO] Isotropic projector ρ={rho:.3f} → β={beta:.4f}, kept {kept*100:.1f}% variance")
    return P, beta

def clip_vector(v: Vector, mu: Vector, Sigma_sqrt: Matrix, Sigma_inv_sqrt: Matrix, alpha: float) -> Vector:
    z = (v - mu) @ Sigma_inv_sqrt
    r = np.linalg.norm(z)
    if r <= alpha: return v
    z *= alpha / r
    return mu + z @ Sigma_sqrt

def cumulant_projector(E_: Matrix, U: Matrix, lam: Vector, rho2: float):
    Z = (E_ @ U) / np.sqrt(np.maximum(lam, 1e-12))
    kappa = (Z ** 4).mean(0) - 3
    power = kappa ** 2
    order = power.argsort()[::-1]
    cum_pow = np.cumsum(power[order]) / power.sum()
    k_cut = int(np.searchsorted(cum_pow, rho2)) + 1
    V = U[:, order[:k_cut]]
    Q = np.eye(U.shape[0]) - V @ V.T
    print(f"[INFO] Cumulant projector ρ2={rho2:.2f} – removed {k_cut} axes")
    return Q

def poly_projector(Sigma: Matrix, mask: str, lam_max: float, *, theta: float | None = None, beta: float | None = None,
                   p: float = 2., deg: int = 24, eps: float = 1e-3):
    if mask == "smooth":
        if theta is None: raise ValueError("--theta required for poly smooth mask")
        f = lambda lam: np.exp(-(lam / theta) ** p)
    elif mask == "iso":
        if beta is None: raise ValueError("--beta_iso required for poly iso mask")
        f = lambda lam: lam / (lam + beta)
    else:
        raise ValueError(f"Unknown mask '{mask}'")
    coeffs = cheb_poly_coeffs(f, deg, lam_max)
    coeffs = sparsify_coeffs(coeffs, Sigma.shape[0], eps)
    P = polynomial_matrix(Sigma, coeffs)
    kept = np.count_nonzero(coeffs)
    print(f"[INFO] Polynomial projector deg={deg}, kept {kept}/{deg+1} terms (ε={eps:g})")
    return P, coeffs

# ──────────────────────────────────────────────────────────────────────────────
# 4. Vector Transform + Normalisation Utilities
# ──────────────────────────────────────────────────────────────────────────────
def apply_transform_and_normalize(vecs: Matrix, mu: Vector, T: Transform) -> Matrix:
    if T is None: Y = vecs - mu
    elif callable(T): Y = np.stack([(T(v) - mu) for v in vecs])
    else: Y = (vecs - mu) @ T
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    return Y / np.maximum(norms, 1e-12)

def orthonormalise_add(B: Matrix, v: Vector, tol: float = 1e-12) -> Tuple[Matrix, float]:
    proj = B.T @ v if B.size else np.array([])
    v_r = v - (B @ proj if B.size else 0)
    n = np.linalg.norm(v_r)
    if n < tol: return B, 0.
    v_r /= n
    return np.column_stack((B, v_r)), n

# ──────────────────────────────────────────────────────────────────────────────
# 5. Advanced JL-Accelerated "Novelty" Chain Generator (from v4.2)
# ──────────────────────────────────────────────────────────────────────────────
def generate_chain_novelty_jl(
    seeds: List[int],
    y_norm: Matrix,
    kv: KeyedVectors,
    k_neigh: int,
    eps0: float,
    max_steps: int,
    lambda_conn: float,
    jl_eps: float,
    sanity_prob: float,
    theta_pool: float,
    k_pool: int,
    gamma: float,
    raw_norm: Matrix,
) -> List[str]:
    """Greedy novelty chain with JL, redundancy penalty, and adaptive ε."""
    rng = np.random.default_rng()
    d_low = y_norm.shape[1]
    if not seeds: return []

    S_low = np.empty((d_low, 0))
    for s in seeds:
        S_low, _ = orthonormalise_add(S_low, y_norm[s].copy())

    neigh_cache: Dict[int, List[int]] = {}
    def neigh(u: int) -> List[int]:
        if u in neigh_cache: return neigh_cache[u]
        items = kv.similar_by_vector(kv.vectors[u], topn=k_neigh)
        neigh_cache[u] = [kv.key_to_index[t] for t, _ in items]
        return neigh_cache[u]

    pool_cache: Dict[int, List[int]] = {}
    def pool_neigh(u: int) -> List[int]:
        if u in pool_cache: return pool_cache[u]
        items = kv.similar_by_vector(kv.vectors[u], topn=k_pool)
        pool_cache[u] = [kv.key_to_index[t] for t, sim in items if sim >= theta_pool]
        return pool_cache[u]

    chain = [kv.index_to_key[i] for i in seeds]
    current = list(dict.fromkeys(seeds))
    vec_current: List[int] = current[:]

    for step in range(len(current), max_steps):
        eps_stop = eps0 / math.sqrt(step + 1)  # Adaptive ε_t

        C_layer = {
            v for u in current
            for v in neigh(u)
            if v not in current and len(set(neigh(v)).intersection(current)) >= 2
        }

        if not C_layer:
            pool: Set[int] = set()
            for u in current: pool.update(pool_neigh(u))
            pool.difference_update(current)
            C_layer = pool

        if not C_layer: break
        C = list(C_layer)

        # Redundancy penalty: γ · max_cos(candidate, current_chain)
        cur_mat = raw_norm[vec_current]
        sims = raw_norm[C] @ cur_mat.T
        redundancy_penalty = np.max(sims, axis=1)

        # Residue (novelty)
        if S_low.size:
            proj = S_low.T @ y_norm[C].T
            r = np.sqrt(np.maximum(0., 1. - np.sum(proj ** 2, axis=0)))
        else:
            r = np.ones(len(C))

        r_new = np.maximum(0., r - gamma * redundancy_penalty)

        # Connectivity
        conn = np.fromiter((len(set(neigh(v)).intersection(current)) for v in C), dtype=np.int32)
        conn_factor = np.where(conn > 0, conn.astype(np.float64) ** lambda_conn, 0.)
        
        score = r_new * conn_factor
        if np.all(score == 0): score = r_new  # Fallback to residue only

        best_loc = int(np.argmax(score))
        best, best_score = C[best_loc], score[best_loc]

        if best_score < eps_stop + jl_eps:
            print(f"[INFO] Stop: score {best_score:.4f} < {eps_stop + jl_eps:.4f} (ε_t + ε_jl)")
            break

        if sanity_prob > 0 and rng.random() < sanity_prob:
            B_full, _ = orthonormalise_add(np.empty((raw_norm.shape[1], 0)), raw_norm[current[0]].copy())
            for j in current[1:]:
                B_full, _ = orthonormalise_add(B_full, raw_norm[j].copy())
            proj_full = B_full.T @ raw_norm[best]
            r_full = math.sqrt(max(0., 1. - np.sum(proj_full ** 2)))
            if abs(r_full - r[best_loc]) > 2 * jl_eps:
                raise RuntimeError(f"Sanity mismatch: |r_full - r_jl| = {abs(r_full - r[best_loc]):.4f} > 2*ε_jl")

        S_low, _ = orthonormalise_add(S_low, y_norm[best].copy())
        current.append(best)
        vec_current.append(best)
        chain.append(kv.index_to_key[best])

        if len(current) >= d_low: break

    return chain

# ──────────────────────────────────────────────────────────────────────────────
# 6. Legacy Strategies: maxvol & greedy
# ──────────────────────────────────────────────────────────────────────────────
def _chain_maxvol(seed: int, norm: Matrix, tau: float, max_steps: int) -> List[int]:
    S = [seed]
    B = norm[seed][:, None]
    for _ in range(max_steps):
        resid = norm.T - B @ (B.T @ norm.T)
        n = np.linalg.norm(resid, axis=0)
        n[S] = -np.inf
        nxt = int(np.argmax(n))
        if n[nxt] <= tau: break
        B, _ = orthonormalise_add(B, norm[nxt].copy())
        S.append(nxt)
    return S

def _chain_greedy(seed: int, norm: Matrix, lamb: float, tau: float, max_steps: int) -> List[int]:
    S = [seed]; x = seed
    for _ in range(max_steps):
        prox = norm @ norm[x]
        red = (norm @ norm[S].T).max(1)
        s = prox - lamb * red
        s[S] = -np.inf
        nxt = int(np.argmax(s))
        if prox[nxt] <= tau or s[nxt] == -np.inf: break
        S.append(nxt); x = nxt
    return S

# ──────────────────────────────────────────────────────────────────────────────
# 7. Generate Chain (Dispatcher)
# ──────────────────────────────────────────────────────────────────────────────
def generate_chain(
    tokens: Sequence[str], kv: KeyedVectors, y_norm: Matrix, *,
    strategy: str, lamb: float, tau: float, k: int, eps_stop: float, max_steps: int,
    lambda_conn: float, jl_eps: float, sanity_prob: float,
    theta_pool: float, k_pool: int, gamma: float, raw_norm: Matrix
) -> List[str]:
    if strategy == "novelty":
        seeds = [kv.key_to_index[t] for t in tokens if t in kv]
        return generate_chain_novelty_jl(
            seeds, y_norm, kv, k_neigh=k, eps0=eps_stop, max_steps=max_steps,
            lambda_conn=lambda_conn, jl_eps=jl_eps, sanity_prob=sanity_prob,
            theta_pool=theta_pool, k_pool=k_pool, gamma=gamma, raw_norm=raw_norm
        )
    
    out = []
    for tok in tokens:
        if tok not in kv: continue
        idx = kv.key_to_index[tok]
        if strategy == "maxvol":
            chain_idx = _chain_maxvol(idx, raw_norm, tau, max_steps)
        elif strategy == "greedy":
            chain_idx = _chain_greedy(idx, raw_norm, lamb, tau, max_steps)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")
        out.extend(kv.index_to_key[i] for i in chain_idx)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 8. Interactive Console
# ──────────────────────────────────────────────────────────────────────────────
def console(
    kv: KeyedVectors, y_norm: Matrix, raw_norm: Matrix, *,
    strategy: str, lamb: float, tau: float, k: int, eps_stop: float, max_steps: int,
    lambda_conn: float, jl_eps: float, sanity_prob: float,
    theta_pool: float, k_pool: int, gamma: float
):
    print("\n" + "=" * 60)
    print(" Spectral-Chain Laboratory – Interactive Console")
    print("=" * 60)
    print(" /chain <seeds>   – Build a chain using the selected strategy")
    print(" <text>           – Find 5 nearest neighbors for the given text")
    print(" quit / exit      – Leave the application\n")

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if q.lower() in {"quit", "exit"}: break
        if not q: continue

        if q.startswith("/chain"):
            seeds = q[len("/chain"):].split()
            if not seeds:
                print("[WARN] No seed words provided for /chain.")
                continue
            print(f"[INFO] Chain seeds: {seeds}")
            chain = generate_chain(
                seeds, kv, y_norm, strategy=strategy, lamb=lamb, tau=tau, k=k,
                eps_stop=eps_stop, max_steps=max_steps, lambda_conn=lambda_conn,
                jl_eps=jl_eps, sanity_prob=sanity_prob, theta_pool=theta_pool,
                k_pool=k_pool, gamma=gamma, raw_norm=raw_norm
            )
            print("  " + " -> ".join(chain) + "\n")
        else:
            toks = [t for t in q.split() if t in kv]
            if not toks:
                print("[WARN] No in-vocabulary tokens found in input."); continue
            vec = kv[toks[0]] if len(toks) == 1 else np.mean([kv[t] for t in toks], axis=0)
            for i, (tok, sim) in enumerate(kv.similar_by_vector(vec, topn=5), 1):
                print(f"  . {tok:20s} (cos={sim:.3f})")
            print()

# ──────────────────────────────────────────────────────────────────────────────
# 9. CLI Entry-Point
# ──────────────────────────────────────────────────────────────────────────────
def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(
        description="Spectral-Chain Lab (v5.0 - Unified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    g_model = ap.add_argument_group("Model and Projector")
    g_model.add_argument("--bin", required=True, help="Path to pre-trained word2vec .bin file")
    g_model.add_argument("--mode", choices=["smooth", "iso", "poly", "combo", "whiten", "none"], default="none",
                         help="Type of spectral transformation to apply. 'none' for no transform.")
    g_model.add_argument("--rho", type=float, default=0.95, help="Variance to preserve for smooth/iso projectors")
    g_model.add_argument("--p", type=float, default=2.0, help="Exponent for smooth projector")
    g_model.add_argument("--mask", choices=["smooth", "iso"], default="smooth", help="Mask for polynomial projector")
    g_model.add_argument("--theta", type=float, help="Parameter for smooth/poly-smooth projector")
    g_model.add_argument("--beta_iso", type=float, help="Parameter for iso/poly-iso projector")
    g_model.add_argument("--poly_degree", type=int, default=24, help="Degree for polynomial projector")
    g_model.add_argument("--poly_eps", type=float, default=1e-3, help="Sparsification epsilon for polynomial projector")

    g_chain = ap.add_argument_group("Chain Generation")
    g_chain.add_argument("--chain_strategy", choices=["novelty", "maxvol", "greedy"], default="novelty")
    g_chain.add_argument("--k_neighbors", type=int, default=10, help="Neighborhood size for connectivity")
    g_chain.add_argument("--k_pool", type=int, default=50, help="[Novelty] Size of fallback candidate pool")
    g_chain.add_argument("--theta_pool", type=float, default=0.4, help="[Novelty] Min similarity for fallback pool")
    g_chain.add_argument("--epsilon_stop", type=float, default=0.05, help="Initial stopping threshold ε₀")
    g_chain.add_argument("--max_steps", type=int, default=60, help="Maximum length of the generated chain")
    g_chain.add_argument("--lambda_conn", type=float, default=0.30, help="[Novelty] Connectivity weight λ")
    g_chain.add_argument("--gamma", type=float, default=0.15, help="[Novelty] Redundancy penalty weight γ")
    g_chain.add_argument("--lambda_red", type=float, default=0.5, help="[Greedy] Redundancy penalty")
    g_chain.add_argument("--tau", type=float, default=0.7, help="[Greedy/Maxvol] Stopping threshold")

    g_jl = ap.add_argument_group("Johnson-Lindenstrauss")
    g_jl.add_argument("--jl_eps", type=float, default=0.02, help="JL distortion parameter ε")
    g_jl.add_argument("--jl_delta", type=float, help="JL failure probability δ (default: 1/d²)")
    g_jl.add_argument("--jl_cache", type=str, default=".jl_cache", help="Directory to cache JL projections")
    g_jl.add_argument("--sanity_prob", type=float, default=0.0, help="Probability of running full-dim sanity check")

    args = ap.parse_args(argv)

    kv = load_or_train_kv(pretrained_path=args.bin)
    raw = kv.vectors.astype(np.float64)
    d = raw.shape[1]
    mu = raw.mean(0)
    T: Transform = None

    if args.mode != "none":
        print(f"[INFO] Building '{args.mode}' projector...")
        mu_eig, lam, U, (Sigma_sqrt, Sigma_inv_sqrt) = eig_from_kv(kv)
        if args.mode == "smooth": T, _ = smooth_projector(lam, U, args.rho, args.p)
        elif args.mode == "iso": T, _ = iso_projector(lam, U, args.rho)
        elif args.mode == "whiten": T = lambda v: clip_vector(v, mu, Sigma_sqrt, Sigma_inv_sqrt, alpha=1.0)
        elif args.mode == "combo":
            P, _ = smooth_projector(lam, U, args.rho, args.p)
            Q = cumulant_projector(raw - mu, U, lam, 0.8)
            T = Q @ P
        elif args.mode == "poly":
            Sigma = np.cov(raw - mu, rowvar=False)
            lam_max = estimate_lambda_max(Sigma)
            T, _ = poly_projector(Sigma, args.mask, lam_max, theta=args.theta, beta=args.beta_iso,
                                  p=args.p, deg=args.poly_degree, eps=args.poly_eps)

    print("[INFO] Pre-computing transformed & normalized full-dim vectors...")
    raw_norm = apply_transform_and_normalize(raw, mu, T)

    delta = args.jl_delta if args.jl_delta is not None else 1 / d ** 2
    m_star = compute_jl_dim(d, args.jl_eps, delta)
    print(f"[INFO] Ambient d={d}, JL m*={m_star} (ε={args.jl_eps}, δ={delta:.2g})")

    if m_star >= d:
        print("[INFO] ε is below critical threshold -> skipping JL (using full-dim vectors)")
        y_norm = raw_norm
    else:
        Path(args.jl_cache).mkdir(exist_ok=True)
        cache_file = Path(args.jl_cache) / f"fjlt_d{d}_m{m_star}_eps{args.jl_eps}.npz"
        if cache_file.exists():
            print(f"[INFO] Loading JL cache from {cache_file.name}")
            y_norm = np.load(cache_file)["y_norm"]
        else:
            print(f"[INFO] Generating {m_star}×{d} FJLT...")
            rng = np.random.default_rng(42)
            J = make_fjlt_transform(m_star, d, rng)
            print("[INFO] Projecting all vectors...")
            Y = J(raw)
            mu_proj = J(mu[None, :])[0]
            Y -= mu_proj
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            y_norm = Y
            np.savez_compressed(cache_file, y_norm=y_norm)

    console(
        kv, y_norm, raw_norm, strategy=args.chain_strategy,
        lamb=args.lambda_red, tau=args.tau, k=args.k_neighbors,
        eps_stop=args.epsilon_stop, max_steps=args.max_steps,
        lambda_conn=args.lambda_conn, jl_eps=args.jl_eps,
        sanity_prob=args.sanity_prob, theta_pool=args.theta_pool,
        k_pool=args.k_pool, gamma=args.gamma
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
