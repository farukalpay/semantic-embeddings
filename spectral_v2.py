#!/usr/bin/env python
"""
spectral_null_cone.py  ·  July 2025  (semantic-chain edition, poly-projector)

New in this revision
--------------------
• mode  poly          – sparse-polynomial projector P(Σ)
      * Chebyshev least–squares fit on [0, λ₁]
      * automatic coefficient sparsification to O(d·log 1/ε) terms
      * no eigendecomposition needed (only a cheap power-iteration
        to estimate λ₁)

Example
-------
python spectral_null_cone.py --bin GoogleNews-vectors-negative300.bin.gz \
                             --mode poly --mask smooth --theta 0.7 --p 2 \
                             --poly_degree 24 --poly_eps 1e-3

The rest of the behaviour (console, /chain, etc.) is unchanged.
"""

from __future__ import annotations
import argparse, sys, re, itertools
from typing import List, Tuple, Callable, Sequence
import numpy as np
from numpy.polynomial import chebyshev as C, polynomial as Pm
from gensim.models import KeyedVectors, Word2Vec
from scipy.linalg import eigh


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Universal-sparse Chebyshev helpers
# ──────────────────────────────────────────────────────────────────────────────

def estimate_lambda_max(A: np.ndarray, it: int = 100) -> float:
    """Power iteration for λ₁ (spectral norm); O(it·d²)."""
    v = np.random.default_rng().standard_normal(A.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(it):
        v = A @ v
        v_norm = np.linalg.norm(v)
        v /= v_norm
    return float(v_norm)


def cheb_poly_coeffs(f: Callable[[np.ndarray], np.ndarray],
                     deg: int,
                     lam_max: float,
                     oversample: int = 501) -> np.ndarray:
    """Return power-basis coefficients c_k s.t.
           ‖f(λ) − Σ_k c_k λ^k‖_{∞,[0,λ₁]}   is (near)-minimal."""
    xs = np.linspace(0.0, lam_max, oversample)
    ys = f(xs)
    # least-squares Chebyshev fit over [0, λ₁]
    cheb = C.Chebyshev.fit(xs, ys, deg, domain=[0.0, lam_max])
    poly = cheb.convert(kind=Pm.Polynomial)           # power basis
    return poly.coef                                   # length = deg+1


def sparsify_coeffs(coeffs: np.ndarray,
                    d: int,
                    eps: float) -> np.ndarray:
    """
    Keep at most  ⌈d·log₂(1/ε)⌉  largest-magnitude coefficients,
    drop the rest to honour the conjectured sparsity envelope.
    """
    k_max = int(np.ceil(d * np.log2(1.0 / eps)))
    if k_max >= len(coeffs):
        return coeffs
    idx = np.argsort(np.abs(coeffs))[::-1]            # descending
    keep = idx[:k_max]
    mask = np.zeros_like(coeffs, dtype=bool)
    mask[keep] = True
    out = coeffs.copy()
    out[~mask] = 0.0
    return out


def polynomial_matrix(A: np.ndarray,
                      coeffs: np.ndarray) -> np.ndarray:
    """Compute  P = Σ_k c_k A^k  via Horner;  O(deg·d³) once."""
    d = A.shape[0]
    P_mat = coeffs[-1] * np.eye(d)
    for c in reversed(coeffs[:-1]):
        P_mat = A @ P_mat + c * np.eye(d)
    return P_mat


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_or_train_kv(*,
                     pretrained_path: str | None = None,
                     corpus_iter=None,
                     vector_size: int = 300,
                     window: int = 5,
                     min_count: int = 5,
                     epochs: int = 5,
                     workers: int = 4) -> KeyedVectors:
    if pretrained_path:
        print(f"[INFO] Loading vectors from {pretrained_path} …")
        kv = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        print(f"[INFO] {len(kv):,} tokens · dim={kv.vector_size}")
        return kv
    if corpus_iter is None:
        raise ValueError("Need --bin or a corpus iterator.")
    print("[INFO] Training Word2Vec …")
    w2v = Word2Vec(corpus_iter,
                   vector_size=vector_size,
                   window=window,
                   min_count=min_count,
                   workers=workers,
                   epochs=epochs)
    return w2v.wv


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Eigen-precomputation (for legacy modes)
# ──────────────────────────────────────────────────────────────────────────────

def eig_from_kv(kv: KeyedVectors) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                           Tuple[np.ndarray, np.ndarray]]:
    """Return μ, λ(desc), U(desc), (Σ½ , Σ⁻½)."""
    E = kv.vectors.astype(np.float64)
    mu = E.mean(axis=0)
    E_hat = E - mu
    Σ = np.cov(E_hat, rowvar=False)
    λ, U = eigh(Σ)               # ascending
    λ, U = λ[::-1], U[:, ::-1]   # descending
    Σ_sqrt  = (U * np.sqrt(λ)) @ U.T
    Σ_isqrt = (U * (λ ** -0.5)) @ U.T
    return mu, λ, U, (Σ_sqrt, Σ_isqrt)


# ──────────────────────────────────────────────────────────────────────────────
# 3-A  smooth  exp[−(λ/θ)^p]      (patched search)
# ──────────────────────────────────────────────────────────────────────────────

def _removed_var_smooth(λ: np.ndarray, theta: float, p: float) -> float:
    w = np.exp(-(λ / theta) ** p)
    return float(((1 - w) * λ).sum() / λ.sum())


def smooth_projector(λ: np.ndarray, U: np.ndarray,
                     rho: float, p: float) -> Tuple[np.ndarray, float]:
    """Find θ* s.t. removed-variance fraction = rho."""
    lo = 1e-12
    hi = λ.max()
    while _removed_var_smooth(λ, hi, p) > rho:
        hi *= 2.0
    theta = hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        g_mid = _removed_var_smooth(λ, mid, p)
        if abs(g_mid - rho) < 1e-6:
            theta = mid
            break
        if g_mid > rho:
            lo = mid
        else:
            hi = mid
        theta = mid
    w = np.exp(-(λ / theta) ** p)
    P = (U * w) @ U.T
    kept = 1.0 - _removed_var_smooth(λ, theta, p)
    print(f"[INFO] smooth ρ={rho:.3f} → θ*={theta:.4f}, p={p}, kept={kept*100:.2f}%")
    return P, theta


# ──────────────────────────────────────────────────────────────────────────────
# 3-B  iso   λ/(λ+β)
# ──────────────────────────────────────────────────────────────────────────────

def _removed_var_iso(λ: np.ndarray, beta: float) -> float:
    w = λ / (λ + beta)
    return float(((1 - w) * λ).sum() / λ.sum())


def iso_projector(λ: np.ndarray, U: np.ndarray,
                  rho: float) -> Tuple[np.ndarray, float]:
    lo = 1e-12
    hi = λ.max() * 10
    while _removed_var_iso(λ, hi) < rho:
        hi *= 2.0
    beta = hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        g_mid = _removed_var_iso(λ, mid)
        if abs(g_mid - rho) < 1e-6:
            beta = mid
            break
        if g_mid < rho:
            lo = mid
        else:
            hi = mid
        beta = mid
    w = λ / (λ + beta)
    P = (U * w) @ U.T
    kept = 1.0 - _removed_var_iso(λ, beta)
    print(f"[INFO] iso ρ={rho:.3f} → β*={beta:.4f}, kept={kept*100:.2f}%")
    return P, beta


# ──────────────────────────────────────────────────────────────────────────────
# 3-C  Mahalanobis clip
# ──────────────────────────────────────────────────────────────────────────────

def mahala_norms(vectors: np.ndarray,
                 mu: np.ndarray,
                 Σ_isqrt: np.ndarray) -> np.ndarray:
    z = (vectors - mu) @ Σ_isqrt
    return np.linalg.norm(z, axis=1)


def clip_vector(vec: np.ndarray,
                mu: np.ndarray,
                Σ_sqrt: np.ndarray,
                Σ_isqrt: np.ndarray,
                alpha: float) -> np.ndarray:
    z = (vec - mu) @ Σ_isqrt
    r = np.linalg.norm(z)
    if r <= alpha:
        return vec
    z_scaled = (alpha / r) * z
    return mu + z_scaled @ Σ_sqrt


# ──────────────────────────────────────────────────────────────────────────────
# 3-D  cumulant projector  (fast diagonal)
# ──────────────────────────────────────────────────────────────────────────────

def cumulant_projector(E_hat: np.ndarray,
                       U: np.ndarray,
                       λ: np.ndarray,
                       rho2: float) -> np.ndarray:
    Z = (E_hat @ U) / np.sqrt(λ)
    kappa = (Z ** 4).mean(axis=0) - 3
    power = kappa ** 2
    order = power.argsort()[::-1]
    cum = np.cumsum(power[order]) / power.sum()
    k_cut = int(np.searchsorted(cum, rho2)) + 1
    V = U[:, order[:k_cut]]
    Q = np.eye(U.shape[0]) - V @ V.T
    print(f"[INFO] cumulant: ρ2={rho2:.2f} → k†={k_cut} high-kurt axes removed")
    return Q


# ──────────────────────────────────────────────────────────────────────────────
# 3-E  NEW  sparse-polynomial projector
# ──────────────────────────────────────────────────────────────────────────────

def poly_projector(Σ: np.ndarray,
                   mask: str,
                   lam_max: float,
                   *,
                   theta: float | None = None,
                   beta: float | None = None,
                   p: float = 2.0,
                   deg: int = 24,
                   eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Return P ≈ w(Σ) and its sparse coefficient vector."""
    if mask == "smooth":
        if theta is None:
            raise ValueError("poly-mode smooth requires --theta.")
        f = lambda lam: np.exp(-(lam / theta) ** p)
    elif mask == "iso":
        if beta is None:
            raise ValueError("poly-mode iso requires --beta.")
        f = lambda lam: lam / (lam + beta)
    else:
        raise ValueError("mask must be 'smooth' or 'iso'.")

    coeffs = cheb_poly_coeffs(f, deg, lam_max)
    coeffs = sparsify_coeffs(coeffs, Σ.shape[0], eps)
    nonzero = np.count_nonzero(coeffs)
    print(f"[INFO] poly-{mask}: degree={deg}, kept {nonzero}/{deg+1} monomials "
          f"(ε={eps:g})")
    P_mat = polynomial_matrix(Σ, coeffs)
    return P_mat, coeffs


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Token helpers
# ──────────────────────────────────────────────────────────────────────────────

def simple_tok(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


def avg_sentence_vec(tokens: List[str],
                     kv: KeyedVectors) -> Tuple[np.ndarray, List[str], List[str]]:
    iv, oov, vecs = [], [], []
    for t in tokens:
        if t in kv:
            iv.append(t); vecs.append(kv[t])
        else:
            oov.append(t)
    if not vecs:
        raise ValueError("no in-vocabulary words")
    return np.mean(vecs, axis=0), iv, oov


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Semantic-chain algorithm  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_chain_for_seed(seed_idx: int,
                             norm_vectors: np.ndarray,
                             λ_: float,
                             τ_: float,
                             max_steps: int = 50) -> List[int]:
    S: List[int] = [seed_idx]
    x_t = seed_idx
    for _ in range(max_steps):
        vec_xt = norm_vectors[x_t]
        prox = norm_vectors @ vec_xt
        vec_S = norm_vectors[S]
        red = (norm_vectors @ vec_S.T).max(axis=1)
        score = prox - λ_ * red
        score[S] = -np.inf
        next_idx = int(np.argmax(score))
        if prox[next_idx] <= τ_ or score[next_idx] == -np.inf:
            break
        S.append(next_idx)
        x_t = next_idx
    return S


def generate_sentence_chain(sentence_tokens: Sequence[str],
                            kv: KeyedVectors,
                            norm_vectors: np.ndarray,
                            λ_: float = 0.5,
                            τ_: float = 0.7,
                            max_steps: int = 50) -> List[str]:
    chain_tokens: List[str] = []
    for tok in sentence_tokens:
        if tok not in kv:
            continue
        seed_idx = kv.key_to_index[tok]
        chain_idx = _generate_chain_for_seed(seed_idx, norm_vectors, λ_, τ_, max_steps)
        chain_tokens.extend(kv.index_to_key[i] for i in chain_idx)
    return chain_tokens


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Console
# ──────────────────────────────────────────────────────────────────────────────

def console(kv: KeyedVectors,
            mu: np.ndarray,
            transform: Callable[[np.ndarray], np.ndarray] | np.ndarray,
            h_vals: np.ndarray,
            eps: float,
            mode: str,
            Σ_isqrt: np.ndarray | None,
            norm_vectors: np.ndarray) -> None:
    vocab = list(kv.key_to_index)
    null_idx = np.where((h_vals > 0) & (h_vals <= eps))[0]
    null_tokens = sorted(((vocab[i], float(h_vals[i])) for i in null_idx),
                         key=lambda x: x[1])

    print("\nEnter word / sentence   (quit to exit,  /null to list ε-small words)")
    print("Extra commands:  /chain [λ] [τ] sentence …\n")

    float_re = re.compile(r"^[0-9]*\.?[0-9]+$")

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            q = "quit"
        if q.lower() in {"quit", "exit"}:
            print("Bye!"); break
        if not q:
            continue

        # /null  ----------------------------------------------------------------
        if q.startswith("/null"):
            if null_tokens:
                print(f"\nTokens with h_i ≤ {eps}  (count {len(null_tokens)})")
                for tok, hv in null_tokens:
                    print(f"  {tok:20s}  h_i = {hv:.5f}")
            else:
                print("[INFO] No token under ε.")
            print(); continue

        # /chain ----------------------------------------------------------------
        if q.startswith("/chain"):
            parts = q.split()
            λ_, τ_, sentence_start = 0.5, 0.7, 1
            if len(parts) >= 3 and float_re.match(parts[1]) and float_re.match(parts[2]):
                λ_, τ_, sentence_start = float(parts[1]), float(parts[2]), 3
            elif len(parts) >= 2 and float_re.match(parts[1]):
                λ_, sentence_start = float(parts[1]), 2
            sent_str = " ".join(parts[sentence_start:])
            if not sent_str:
                print("[WARN] Sentence missing after /chain"); continue
            tok_seq = simple_tok(sent_str)
            if not tok_seq:
                print("[WARN] No valid tokens found"); continue
            chain = generate_sentence_chain(tok_seq, kv, norm_vectors, λ_, τ_)
            if chain:
                print("\n" + "  ".join(chain) + "\n")
            else:
                print("[WARN] No in-vocabulary tokens in sentence\n")
            continue

        # normal query ----------------------------------------------------------
        toks = simple_tok(q)
        try:
            if len(toks) == 1:
                w = toks[0]
                if w not in kv:
                    print(f"[WARN] '{w}' OOV"); continue
                vec, iv, oov = kv[w], [w], []
            else:
                vec, iv, oov = avg_sentence_vec(toks, kv)
        except ValueError as e:
            print(f"[WARN] {e}"); continue

        if mode == "whiten":
            vec_t = transform(vec)               # still unused
            h_q = mahala_norms(vec[None, :], mu, Σ_isqrt)[0]
        else:
            P_mat = transform                    # matrix
            h_q = np.linalg.norm(P_mat @ (vec - mu))

        print("\n--- DEBUG ----------------------------------------------------")
        print(f"IV : {iv}")
        if oov:
            print(f"OOV: {oov}")
        print(f"h_query = {h_q:.5f}")
        print("----------------------------------------------------------------")

        nbrs = kv.similar_by_vector(vec, topn=10)
        print("\nTop-10 neighbours:")
        for i, (tok, sim) in enumerate(nbrs, 1):
            print(f"{i:2d}. {tok:20s}  cos={sim:.4f}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Word2Vec null-cone laboratory + semantic chains (poly edition)")
    ap.add_argument("--bin", required=True, help="word2vec .bin/.vec.gz file")

    ap.add_argument("--mode", choices=["smooth", "iso", "whiten", "combo", "poly"],
                    default="smooth")

    # legacy smooth / iso parameters ------------------------------------------
    ap.add_argument("--rho", type=float, default=0.95, help="variance fraction")
    ap.add_argument("--p", type=float, default=2.0, help="exp filter exponent")

    # poly-mode extra knobs ----------------------------------------------------
    ap.add_argument("--mask", choices=["smooth", "iso"], default="smooth",
                    help="spectral mask for poly mode")
    ap.add_argument("--theta", type=float, help="θ for smooth-mask poly mode")
    ap.add_argument("--beta", type=float, help="β for iso-mask poly mode")
    ap.add_argument("--poly_degree", type=int, default=24, help="Chebyshev degree")
    ap.add_argument("--poly_eps", type=float, default=1e-3,
                    help="coefficient truncation ε")

    # whiten / combo -----------------------------------------------------------
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="Mahalanobis radius for whiten-clip")
    ap.add_argument("--rho2", type=float, default=0.80,
                    help="kurtosis variance fraction for combo")

    # misc ---------------------------------------------------------------------
    ap.add_argument("--eps", type=float, default=0.05, help="ε for /null")
    args = ap.parse_args(argv)

    kv = load_or_train_kv(pretrained_path=args.bin)
    norm_vectors = kv.get_normed_vectors()  # cached for chain algorithm

    mu = kv.vectors.mean(axis=0)
    E_hat = kv.vectors - mu

    # ─── legacy modes that need eig stuff ─────────────────────────────────────
    if args.mode in {"smooth", "iso", "combo"}:
        mu, λ, U, (Σ_sqrt, Σ_isqrt) = eig_from_kv(kv)

    mode = args.mode
    if mode == "smooth":
        P, _ = smooth_projector(λ, U, args.rho, args.p)
        transform = P
        h = np.linalg.norm(E_hat @ P, axis=1)

    elif mode == "iso":
        P, _ = iso_projector(λ, U, args.rho)
        transform = P
        h = np.linalg.norm(E_hat @ P, axis=1)

    elif mode == "whiten":
        _, _, _, (Σ_sqrt, Σ_isqrt) = eig_from_kv(kv)
        transform = lambda v: clip_vector(v, mu, Σ_sqrt, Σ_isqrt, args.alpha)
        h = mahala_norms(kv.vectors, mu, Σ_isqrt)

    elif mode == "combo":
        P_base, _ = smooth_projector(λ, U, args.rho, args.p)
        Q = cumulant_projector(E_hat, U, λ, args.rho2)
        P = Q @ P_base
        transform = P
        h = np.linalg.norm(E_hat @ P, axis=1)

    # ─── NEW poly mode (no eigendecomp) ───────────────────────────────────────
    elif mode == "poly":
        Σ = np.cov(E_hat, rowvar=False)
        lam_max = estimate_lambda_max(Σ)
        P, coeffs = poly_projector(Σ,
                                   mask=args.mask,
                                   lam_max=lam_max,
                                   theta=args.theta,
                                   beta=args.beta,
                                   p=args.p,
                                   deg=args.poly_degree,
                                   eps=args.poly_eps)
        transform = P
        h = np.linalg.norm(E_hat @ P, axis=1)
        Σ_isqrt = None                                   # only used by whiten
    else:
        raise ValueError("unknown mode")

    console(kv, mu, transform, h, args.eps, mode, locals().get('Σ_isqrt'), norm_vectors)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
