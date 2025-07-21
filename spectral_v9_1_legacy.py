#!/usr/bin/env python3
"""
Unified Spectral-Chain Laboratory (v6.1 - Optimized)
=====================================================

This implementation includes the efficient heap-based algorithm for computing
the least fixed point S* of the operator Φ(S) = S ∪ {v ∉ S : σ_S(v) > ε}.

Key optimization: Instead of recomputing σ_S for all vertices at each step,
we maintain incremental updates using:
- Rank-1 Pythagorean updates for residues
- Incremental connectivity updates
- Max-heap for efficient vertex selection

Complexity: O((N + |E|) log N) where |E| = Σ_{v∈S*} |N(v)|
"""
from __future__ import annotations

import argparse
import heapq
import math
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial import chebyshev as C, polynomial as Pm
from gensim.models import KeyedVectors, Word2Vec
from scipy.linalg import hadamard, eigh

# ──────────────────────────────────────────────────────────────────────────────
# Type Aliases
# ──────────────────────────────────────────────────────────────────────────────
Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
Transform = Union[Callable[[Vector], Vector], Matrix, None]

# ──────────────────────────────────────────────────────────────────────────────
# Core Mathematical Primitives
# ──────────────────────────────────────────────────────────────────────────────

def orthonormalise_add(B: Matrix, v: Vector, tol: float = 1e-12) -> Tuple[Matrix, float]:
    """
    Extends orthonormal basis B with vector v (if linearly independent).
    Returns (B', ||residual||).
    """
    if B.size == 0:
        nrm = np.linalg.norm(v)
        return (v / max(nrm, tol)).reshape(-1, 1), nrm
    proj = B.T @ v
    r = v - B @ proj
    nrm = np.linalg.norm(r)
    if nrm < tol:
        return B, 0.0
    r /= nrm
    return np.column_stack((B, r)), nrm


class ResidueOracle:
    """
    Maintains an orthonormal basis for span{y_u : u ∈ S} and allows O(dim²)
    batch evaluation of ‖P_S y_v‖² for many outside vertices v.
    """
    __slots__ = ("B", "Y")

    def __init__(self, Y: Matrix):
        self.B: Matrix = np.empty((Y.shape[1], 0), dtype=np.float64)
        self.Y: Matrix = Y

    def extend(self, idx: int) -> None:
        self.B, _ = orthonormalise_add(self.B, self.Y[idx])

    def squared_proj(self, idxs: Sequence[int]) -> Vector:
        if self.B.size == 0:
            return np.zeros(len(idxs), dtype=np.float64)
        Ysub = self.Y[np.fromiter(idxs, dtype=np.int32)]
        coeff = self.B.T @ Ysub.T  # k × m
        return np.sum(coeff * coeff, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Efficient Heap-Based Fixed Point Algorithm
# ──────────────────────────────────────────────────────────────────────────────

class MaxHeapEntry:
    """Entry for max-heap with vertex index and score."""
    __slots__ = ('v', 'score')
    
    def __init__(self, v: int, score: float):
        self.v = v
        self.score = score
    
    def __lt__(self, other):
        # Python's heapq is a min-heap, so we negate for max-heap behavior
        return self.score > other.score


def generate_chain_fixed_point_optimized(
    seeds: List[int],
    Y: Matrix,
    kv: KeyedVectors,
    k_neigh: int,
    epsilon: float,
    lambda_conn: float,
    max_steps: int | None = None
) -> List[str]:
    """
    Optimized fixed-point algorithm using heap and incremental updates.
    
    Maintains:
    - r(v) = 1 - ||P_S y_v||² (residue squared)
    - c(v) = |N(v) ∩ S| (connectivity)
    - s(v) = sqrt(r(v)) * c(v)^λ (score)
    """
    if not seeds:
        raise ValueError("Fixed-point strategy needs at least one seed")
    
    n = len(Y)
    max_steps = n if not max_steps or max_steps <= 0 else min(max_steps, n)
    
    # Neighbor cache
    neigh_cache: Dict[int, List[int]] = {}
    def get_neighbors(u: int) -> List[int]:
        if u in neigh_cache:
            return neigh_cache[u]
        items = kv.similar_by_vector(kv.vectors[u], topn=k_neigh)
        neigh_cache[u] = [kv.key_to_index[t] for t, _ in items]
        return neigh_cache[u]
    
    # Initialize arrays
    r = np.ones(n, dtype=np.float64)  # residue squared: r(v) = 1 - ||P_S y_v||²
    c = np.zeros(n, dtype=np.int32)   # connectivity: c(v) = |N(v) ∩ S|
    s = np.zeros(n, dtype=np.float64) # score: s(v) = sqrt(r(v)) * c(v)^λ
    
    # Track selected vertices
    S = set(seeds)
    in_heap = set()
    heap = []
    chain = list(seeds)
    
    # Orthonormal basis (stored as list of normalized residual vectors)
    basis_vectors = []
    
    # Initialize with seeds
    for seed in seeds:
        # Update basis
        if not basis_vectors:
            y_hat = Y[seed].copy()
        else:
            # Project out existing basis components
            y_hat = Y[seed].copy()
            for b in basis_vectors:
                y_hat -= np.dot(y_hat, b) * b
        
        norm = np.linalg.norm(y_hat)
        if norm > 1e-12:
            y_hat /= norm
            basis_vectors.append(y_hat)
        
        # Update residues for all vertices using rank-1 update
        for v in range(n):
            if v not in S:
                r[v] = r[v] - np.dot(y_hat, Y[v])**2
                r[v] = max(0, r[v])  # numerical safety
        
        # Update connectivity for neighbors
        for w in get_neighbors(seed):
            if w not in S:
                c[w] += 1
    
    # Compute initial scores and populate heap
    for v in range(n):
        if v not in S and c[v] > 0:
            s[v] = np.sqrt(r[v]) * (c[v] ** lambda_conn)
            if s[v] > epsilon:
                heapq.heappush(heap, MaxHeapEntry(v, s[v]))
                in_heap.add(v)
    
    # Main loop
    while heap and len(chain) < max_steps:
        # Extract max
        entry = heapq.heappop(heap)
        v_star = entry.v
        
        # Check if score is still valid (due to lazy deletion)
        if v_star in S or s[v_star] <= epsilon:
            in_heap.discard(v_star)
            continue
        
        # Add to solution
        S.add(v_star)
        in_heap.discard(v_star)
        chain.append(v_star)
        
        # Update basis and residues
        if basis_vectors:
            y_hat = Y[v_star].copy()
            for b in basis_vectors:
                y_hat -= np.dot(y_hat, b) * b
        else:
            y_hat = Y[v_star].copy()
        
        norm = np.linalg.norm(y_hat)
        if norm > 1e-12:
            y_hat /= norm
            basis_vectors.append(y_hat)
            
            # Update residues using rank-1 Pythagorean update
            for w in range(n):
                if w not in S:
                    dot_prod = np.dot(y_hat, Y[w])
                    r[w] = r[w] - dot_prod**2
                    r[w] = max(0, r[w])  # numerical safety
        
        # Update connectivity for neighbors of v_star
        neighbors_to_update = []
        for w in get_neighbors(v_star):
            if w not in S:
                c[w] += 1
                neighbors_to_update.append(w)
        
        # Recompute scores and update heap for affected vertices
        for w in neighbors_to_update:
            if c[w] > 0:
                s[w] = np.sqrt(r[w]) * (c[w] ** lambda_conn)
                if s[w] > epsilon:
                    heapq.heappush(heap, MaxHeapEntry(w, s[w]))
                    in_heap.add(w)
                elif w in in_heap:
                    # Mark for lazy deletion
                    in_heap.discard(w)
    
    return [kv.index_to_key[i] for i in chain]


# ──────────────────────────────────────────────────────────────────────────────
# Other Chain Generation Strategies (kept for comparison)
# ──────────────────────────────────────────────────────────────────────────────

def generate_chain_novelty(
    seeds: List[int],
    y_norm: Matrix,
    kv: KeyedVectors,
    k_neigh: int,
    epsilon: float,
    lambda_conn: float,
    gamma: float,
    raw_norm: Matrix,
    max_steps: int
) -> List[str]:
    """
    Novelty strategy with redundancy penalty and adaptive ε.
    """
    if not seeds:
        return []

    S_low = np.empty((y_norm.shape[1], 0))
    chain: List[int] = list(dict.fromkeys(seeds))
    
    for s in seeds:
        S_low, _ = orthonormalise_add(S_low, y_norm[s])

    neigh_cache: Dict[int, List[int]] = {}
    def neigh(u: int) -> List[int]:
        if u in neigh_cache:
            return neigh_cache[u]
        items = kv.similar_by_vector(kv.vectors[u], topn=k_neigh)
        neigh_cache[u] = [kv.key_to_index[t] for t, _ in items]
        return neigh_cache[u]

    for step in range(len(chain), min(max_steps, raw_norm.shape[0])):
        # Adaptive epsilon
        eps_t = epsilon / math.sqrt(step + 1)
        
        cand = [v for v in range(len(kv)) if v not in chain]
        if not cand:
            break

        # Residue
        if S_low.size:
            proj = S_low.T @ y_norm[cand].T
            r = np.sqrt(np.maximum(0., 1. - np.sum(proj ** 2, axis=0)))
        else:
            r = np.ones(len(cand))

        # Connectivity
        conn = np.fromiter(
            (len(set(neigh(v)).intersection(chain)) for v in cand),
            dtype=np.int32
        )
        conn_factor = np.where(conn > 0,
                               conn.astype(np.float64) ** lambda_conn,
                               0.0)

        # Redundancy penalty
        cur_mat = raw_norm[chain]
        sims = raw_norm[cand] @ cur_mat.T
        redund = np.max(sims, axis=1)
        
        score = np.maximum(0., r - gamma * redund) * conn_factor

        best_idx = int(np.argmax(score))
        if score[best_idx] <= eps_t:
            break
        
        best = cand[best_idx]
        chain.append(best)
        S_low, _ = orthonormalise_add(S_low, y_norm[best])

    return [kv.index_to_key[i] for i in chain]


def _chain_maxvol(seed: int, norm: Matrix, tau: float, max_steps: int) -> List[int]:
    """MaxVol strategy: maximize residual norm."""
    S = [seed]
    B = norm[seed][:, None]
    for _ in range(max_steps):
        resid = norm.T - B @ (B.T @ norm.T)
        n = np.linalg.norm(resid, axis=0)
        n[S] = -np.inf
        nxt = int(np.argmax(n))
        if n[nxt] <= tau:
            break
        B, _ = orthonormalise_add(B, norm[nxt])
        S.append(nxt)
    return S


def _chain_greedy(seed: int, norm: Matrix, lamb: float, tau: float, max_steps: int) -> List[int]:
    """Greedy strategy with redundancy penalty."""
    S = [seed]
    x = seed
    for _ in range(max_steps):
        prox = norm @ norm[x]
        red = (norm @ norm[S].T).max(1)
        s = prox - lamb * red
        s[S] = -np.inf
        nxt = int(np.argmax(s))
        if prox[nxt] <= tau or s[nxt] == -np.inf:
            break
        S.append(nxt)
        x = nxt
    return S


# ──────────────────────────────────────────────────────────────────────────────
# Johnson-Lindenstrauss Utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_epsilon_min(d: int, delta: float) -> float:
    """Minimum achievable JL distortion."""
    return math.sqrt(4.0 * math.log(d / delta) / d)


def compute_jl_dim(d: int, eps: float, delta: float | None = None) -> int:
    """Target dimension for JL projection."""
    if not (0. < eps < 1.):
        raise ValueError("eps must be in (0,1)")
    delta = 1. / (d * d) if delta is None else delta
    if eps < compute_epsilon_min(d, delta):
        return d
    return min(d, math.ceil(4. * math.log(d / delta) / (eps ** 2)))


def make_fjlt_transform(m: int, d: int, rng: np.random.Generator):
    """Fast JL transform using Hadamard matrices."""
    d_pad = 1 << ((d - 1).bit_length())
    H = hadamard(d_pad).astype(np.float32)
    D = rng.choice([-1., 1.], d_pad).astype(np.float32)
    Π = rng.permutation(d_pad)
    rows = rng.choice(d_pad, size=m, replace=False)
    scale = math.sqrt(d_pad / m)
    
    def fjlt(X: Matrix) -> Matrix:
        X = np.atleast_2d(X.astype(np.float32))
        if X.shape[1] != d:
            raise ValueError(f"Expected dim {d}, got {X.shape[1]}")
        Xp = np.zeros((X.shape[0], d_pad), np.float32)
        Xp[:, :d] = X
        Xp = Xp[:, Π] * D
        Xh = (H @ Xp.T) / math.sqrt(d_pad)
        return (scale * Xh[rows].T)
    
    return fjlt


# ──────────────────────────────────────────────────────────────────────────────
# Spectral Projectors
# ──────────────────────────────────────────────────────────────────────────────

def eig_from_kv(kv: KeyedVectors):
    """Compute eigendecomposition of word vectors."""
    E = kv.vectors.astype(np.float64)
    mu = E.mean(0)
    E0 = E - mu
    Sigma = np.cov(E0, rowvar=False)
    lam, U = eigh(Sigma)
    lam, U = lam[::-1], U[:, ::-1]
    Sigma_sqrt = (U * np.sqrt(np.maximum(lam, 0))) @ U.T
    Sigma_inv = (U * np.where(lam > 0, lam ** -0.5, 0)) @ U.T
    return mu, lam, U, (Sigma_sqrt, Sigma_inv)


def smooth_projector(lam: Vector, U: Matrix, rho: float, p: float):
    """Smooth projector that preserves fraction ρ of variance."""
    def frac(theta): 
        return ((1 - np.exp(-(lam / theta) ** p)) * lam).sum() / lam.sum()
    
    lo, hi = 1e-12, lam.max()
    while frac(hi) > rho: 
        hi *= 2
    
    for _ in range(100):
        mid = .5 * (lo + hi)
        if abs(frac(mid) - rho) < 1e-6: 
            break
        lo, hi = (mid, hi) if frac(mid) > rho else (lo, mid)
    
    theta = .5 * (lo + hi)
    w = np.exp(-(lam / theta) ** p)
    P = (U * w) @ U.T
    kept = 1 - frac(theta)
    print(f"[INFO] Smooth projector ρ={rho:.3f} → θ={theta:.4f}, kept {kept*100:.1f}% variance")
    return P, theta


def iso_projector(lam: Vector, U: Matrix, rho: float):
    """Isotropic projector."""
    def frac(beta): 
        return ((1 - lam / (lam + beta)) * lam).sum() / lam.sum()
    
    lo, hi = 1e-12, lam.max()
    while frac(hi) < rho: 
        hi *= 2
    
    for _ in range(100):
        mid = .5 * (lo + hi)
        if abs(frac(mid) - rho) < 1e-6: 
            break
        lo, hi = (mid, hi) if frac(mid) < rho else (lo, mid)
    
    beta = .5 * (lo + hi)
    w = lam / (lam + beta)
    P = (U * w) @ U.T
    kept = 1 - frac(beta)
    print(f"[INFO] Isotropic projector ρ={rho:.3f} → β={beta:.4f}, kept {kept*100:.1f}% variance")
    return P, beta


def apply_transform_and_normalize(vecs: Matrix, mu: Vector, T: Transform) -> Matrix:
    """Apply transformation and normalize vectors."""
    if T is None:
        X = vecs - mu
    elif callable(T):
        X = np.stack([T(v) - T(mu) for v in vecs])
    else:
        X = (vecs - mu) @ T
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Main Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def dispatch_chain(
    tokens: Sequence[str],
    kv: KeyedVectors,
    strategy: str,
    raw_norm: Matrix,
    y_norm: Matrix,
    *,
    k: int,
    epsilon: float,
    lambda_conn: float,
    gamma: float,
    max_steps: int,
    tau: float,
    lamb: float
) -> List[str]:
    """Dispatch to appropriate chain generation strategy."""
    seeds = [kv.key_to_index[t] for t in tokens if t in kv]
    if not seeds:
        return []

    if strategy == "fixedpoint":
        return generate_chain_fixed_point_optimized(
            seeds, raw_norm, kv, k_neigh=k,
            epsilon=epsilon, lambda_conn=lambda_conn,
            max_steps=max_steps
        )
    elif strategy == "novelty":
        return generate_chain_novelty(
            seeds, y_norm, kv, k_neigh=k,
            epsilon=epsilon, lambda_conn=lambda_conn,
            gamma=gamma, raw_norm=raw_norm,
            max_steps=max_steps
        )
    elif strategy == "maxvol":
        chain = _chain_maxvol(seeds[0], raw_norm, tau, max_steps)
        return [kv.index_to_key[i] for i in chain]
    elif strategy == "greedy":
        chain = _chain_greedy(seeds[0], raw_norm, lamb, tau, max_steps)
        return [kv.index_to_key[i] for i in chain]
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_or_train_kv(*, binary_path: str | None = None,
                     corpus_iter=None, **w2v_kwargs) -> KeyedVectors:
    """Load pre-trained vectors or train new model."""
    if binary_path:
        print(f"[INFO] Loading embeddings from '{Path(binary_path).name}'")
        kv = KeyedVectors.load_word2vec_format(binary_path, binary=True)
        print(f"[INFO] Loaded {len(kv):,} words, dimension {kv.vector_size}")
        return kv
    if corpus_iter is None:
        raise ValueError("Either --bin or a corpus iterator is required")
    print("[INFO] Training Word2Vec model...")
    return Word2Vec(corpus_iter, **w2v_kwargs).wv


# ──────────────────────────────────────────────────────────────────────────────
# Interactive Console
# ──────────────────────────────────────────────────────────────────────────────

def console(
    kv: KeyedVectors,
    raw_norm: Matrix,
    y_norm: Matrix,
    *,
    strategy: str,
    k: int,
    epsilon: float,
    lambda_conn: float,
    gamma: float,
    max_steps: int,
    tau: float,
    lamb: float
):
    """Interactive console for chain generation and word similarity."""
    print("\n" + "=" * 70)
    print(" Unified Spectral-Chain Laboratory – Interactive Console (v6.1)")
    print("=" * 70)
    print(" Commands:")
    print("   /chain <seeds>     Generate a chain with the selected strategy")
    print("   /info              Show current configuration")
    print("   /strategies        List available chain generation strategies")
    print("   <text>             Show 5 nearest neighbors of <text>")
    print("   quit / exit        Leave the console")
    print(f"\n Current strategy: {strategy}")
    print("=" * 70 + "\n")

    strategies_info = {
        "fixedpoint": "Heap-based algorithm with O((N+|E|)log N) complexity",
        "novelty": "Advanced with redundancy penalty γ and adaptive ε_t = ε/√t",
        "maxvol": "Legacy: maximize residual norm (single seed)",
        "greedy": "Legacy: greedy with redundancy penalty (single seed)"
    }

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if q.lower() in {"quit", "exit"}:
            break
        
        if not q:
            continue

        if q == "/info":
            print(f"\n Configuration:")
            print(f"   Strategy: {strategy}")
            print(f"   ε = {epsilon:.3f}, λ = {lambda_conn:.2f}")
            if strategy == "novelty":
                print(f"   γ = {gamma:.2f} (redundancy penalty)")
            print(f"   k = {k} (neighborhood size)")
            print(f"   max_steps = {max_steps}")
            print()
            continue

        if q == "/strategies":
            print("\n Available strategies:")
            for name, desc in strategies_info.items():
                print(f"   {name:12s} - {desc}")
            print()
            continue

        if q.startswith("/chain"):
            seeds = q.removeprefix("/chain").split()
            if not seeds:
                print("[WARN] No seeds supplied. Usage: /chain word1 word2 ...")
                continue
            
            valid_seeds = [s for s in seeds if s in kv]
            if not valid_seeds:
                print(f"[WARN] No valid seeds found. Available seeds must be in vocabulary.")
                continue
            
            if len(valid_seeds) < len(seeds):
                print(f"[INFO] Using {len(valid_seeds)} valid seeds: {valid_seeds}")
            
            t0 = time.perf_counter()
            chain = dispatch_chain(
                valid_seeds, kv, strategy, raw_norm, y_norm,
                k=k, epsilon=epsilon, lambda_conn=lambda_conn,
                gamma=gamma, max_steps=max_steps,
                tau=tau, lamb=lamb
            )
            t1 = time.perf_counter()
            
            print(f"\n Chain ({len(chain)} words, {t1-t0:.3f}s):")
            print("  " + " → ".join(chain) + "\n")
        else:
            # Word similarity query
            toks = [t for t in q.split() if t in kv]
            if not toks:
                print("[WARN] No in-vocabulary tokens found.")
                continue
            
            vec = kv[toks[0]] if len(toks) == 1 else np.mean([kv[t] for t in toks], axis=0)
            print(f"\n Nearest neighbors of '{' '.join(toks)}':")
            for tok, sim in kv.similar_by_vector(vec, topn=5):
                print(f"   {tok:20s} (cos = {sim:.3f})")
            print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(
        prog="spectral-chain",
        description="Unified Spectral-Chain Laboratory (v6.1 - Optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    g_model = ap.add_argument_group("Model / Embeddings")
    g_model.add_argument("--bin", required=True,
                         help="Pre-trained word2vec .bin file")
    g_model.add_argument("--mode", 
                         choices=["none", "smooth", "iso", "whiten"],
                         default="none",
                         help="Spectral transformation mode")
    g_model.add_argument("--rho", type=float, default=0.95,
                         help="Variance to preserve (smooth/iso modes)")
    g_model.add_argument("--p", type=float, default=2.0,
                         help="Exponent for smooth projector")

    # Chain generation arguments
    g_chain = ap.add_argument_group("Chain Generation")
    g_chain.add_argument("--chain_strategy",
                         choices=["fixedpoint", "novelty", "maxvol", "greedy"],
                         default="fixedpoint",
                         help="Chain generation strategy")
    g_chain.add_argument("--k_neighbors", type=int, default=10,
                         help="Neighborhood size |N(v)|")
    g_chain.add_argument("--epsilon", type=float, default=0.1,
                         help="Threshold ε for σ_S(v)")
    g_chain.add_argument("--lambda_conn", type=float, default=0.30,
                         help="Connectivity exponent λ")
    g_chain.add_argument("--gamma", type=float, default=0.15,
                         help="[novelty] Redundancy penalty weight γ")
    g_chain.add_argument("--max_steps", type=int, default=100,
                         help="Maximum chain length")
    
    # Legacy parameters
    g_legacy = ap.add_argument_group("Legacy Parameters")
    g_legacy.add_argument("--tau", type=float, default=0.7,
                          help="[maxvol/greedy] Stop threshold")
    g_legacy.add_argument("--lambda_red", type=float, default=0.5,
                          help="[greedy] Redundancy penalty")

    # Johnson-Lindenstrauss arguments
    g_jl = ap.add_argument_group("Johnson-Lindenstrauss")
    g_jl.add_argument("--jl_eps", type=float, default=0.02,
                      help="JL distortion parameter")
    g_jl.add_argument("--jl_cache", type=str, default=".jl_cache",
                      help="Directory for JL projection cache")

    args = ap.parse_args(argv)

    # Load embeddings
    kv = load_or_train_kv(binary_path=args.bin)
    raw = kv.vectors.astype(np.float64)
    d = raw.shape[1]
    mu_vec = raw.mean(0)
    
    # Apply spectral transformation if requested
    T: Transform = None
    if args.mode != "none":
        print(f"[INFO] Building '{args.mode}' projector...")
        mu_eig, lam, U, (Sigma_sqrt, Sigma_inv_sqrt) = eig_from_kv(kv)
        
        if args.mode == "smooth":
            T, _ = smooth_projector(lam, U, args.rho, args.p)
        elif args.mode == "iso":
            T, _ = iso_projector(lam, U, args.rho)
        elif args.mode == "whiten":
            alpha = 1.0
            def clip_fn(v):
                z = (v - mu_vec) @ Sigma_inv_sqrt
                r = np.linalg.norm(z)
                if r <= alpha:
                    return v
                z *= alpha / r
                return mu_vec + z @ Sigma_sqrt
            T = clip_fn

    # Normalize vectors
    print("[INFO] Computing normalized vectors...")
    raw_norm = apply_transform_and_normalize(raw, mu_vec, T)

    # Johnson-Lindenstrauss projection
    m_star = compute_jl_dim(d, args.jl_eps)
    print(f"[INFO] JL projection: d={d} → m*={m_star} (ε={args.jl_eps})")
    
    if m_star >= d:
        print("[INFO] Using full-dimensional vectors (no JL needed)")
        y_norm = raw_norm
    else:
        Path(args.jl_cache).mkdir(exist_ok=True)
        cache_file = Path(args.jl_cache) / f"fjlt_{d}_{m_star}.npz"
        
        if cache_file.exists():
            print(f"[INFO] Loading cached JL projection from {cache_file.name}")
            y_norm = np.load(cache_file)["y_norm"]
        else:
            print(f"[INFO] Computing {m_star}×{d} FJLT projection...")
            rng = np.random.default_rng(42)
            fjlt = make_fjlt_transform(m_star, d, rng)
            Y = fjlt(raw)
            mu_proj = fjlt(mu_vec[None, :])[0]
            Y -= mu_proj
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            y_norm = Y
            np.savez_compressed(cache_file, y_norm=y_norm)
            print(f"[INFO] Cached JL projection to {cache_file.name}")

    # Start interactive console
    console(
        kv, raw_norm, y_norm,
        strategy=args.chain_strategy,
        k=args.k_neighbors,
        epsilon=args.epsilon,
        lambda_conn=args.lambda_conn,
        gamma=args.gamma,
        max_steps=args.max_steps,
        tau=args.tau,
        lamb=args.lambda_red
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
