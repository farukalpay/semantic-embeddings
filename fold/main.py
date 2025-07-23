#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated fixed‑point and semantic chain framework
================================================

This module extends the original OFI (Ordinal Folding Index) framework with
an improved semantic chain generation algorithm.  The goal of the improved
algorithm is to overcome the limitations of the original depth‑first search
that relied on a hard similarity threshold.  Instead, a redundancy‑penalised
greedy strategy is employed: at each step the next word is selected by
maximising a score that rewards proximity to the current node and penalises
similarity to any previously visited node.  The search stops when the
similarity to the best candidate drops below a user‑controlled threshold or
when a maximum length is reached.

The remainder of the file preserves the original functionality for parsing
and evaluating modal μ‑calculus formulas on a fixed Kripke model and
computing their Ordinal Folding Index (OFI).  The interface remains
command‑line driven: users may either evaluate a formula or generate a
semantic chain for a given word.

Usage
-----

To evaluate a formula and compute its OFI:

    python updated_code.py "<formula>"

To generate a semantic chain for a given word using the improved algorithm:

    python updated_code.py /chain <word> [--lambda λ] [--tau τ] [--max M]

Parameters for the chain algorithm:

* ``--lambda λ`` (or ``-l λ``) controls the redundancy penalty.  Larger
  values penalise similarity to previously selected words more heavily.
  Default is 0.5.
* ``--tau τ`` (or ``-t τ``) sets the stopping threshold for cosine similarity.
  If the similarity between the current word and the best candidate falls
  below τ, the chain terminates.  Default is 0.7.
* ``--max M`` (or ``-m M``) specifies the maximum number of words in the
  chain (including the starting word).  Default is 20.

The module will attempt to load pre‑trained GloVe embeddings from the path
specified in ``load_glove_embeddings()``.  You may adjust the file path
there to point to your local copy of the GloVe vectors.  If the file is
missing, a helpful error will be printed.
"""

import sys
import math
import argparse
from typing import List, Tuple, Dict, Set, Optional

try:
    import numpy as np
except ImportError:
    np = None  # Handle missing numpy gracefully


# ---------------------------------------------------------------------------
# Kripke model definition for symbolic (modal μ‑calculus) evaluation
# ---------------------------------------------------------------------------

# We use a small fixed model with three states labelled 0, 1, 2.  The
# transitions form a simple chain: 0 → 1 → 2.  Atomic propositions p and q
# are assigned to these states via the ``atomic_valuations`` mapping.
states: Set[int] = {0, 1, 2}
transitions: Dict[int, List[int]] = {
    0: [1],   # state 0 has successor 1
    1: [2],   # state 1 has successor 2
    2: []     # state 2 has no successors (terminal state)
}

# Atomic proposition valuations.  Proposition 'p' holds only in state 2,
# while 'q' holds nowhere.  Any proposition not listed here is assumed
# false everywhere.
atomic_valuations: Dict[str, Set[int]] = {
    'p': {2},
    'q': set()
}


# ---------------------------------------------------------------------------
# Formula parsing infrastructure
# ---------------------------------------------------------------------------

# We implement a simple recursive descent parser for a modal μ‑calculus
# fragment.  The grammar supports conjunction, disjunction, negation,
# modal operators (box and diamond), and least/greatest fixpoint binders.

def tokenize(formula_str: str) -> List[Tuple[str, Optional[object]]]:
    """Tokenise a formula string into a list of (type, value) pairs.

    Supported tokens include reserved words (mu, nu, True, False, not, and,
    or), multi‑character modal symbols ([] and <>), punctuation (parentheses,
    dot), atomic propositions (lowercase identifiers), and fixpoint variables
    (uppercase identifiers).  Greek letters μ and ν are also recognised.
    """
    tokens: List[Tuple[str, Optional[object]]] = []
    i = 0
    n = len(formula_str)
    while i < n:
        c = formula_str[i]
        if c.isspace():
            i += 1
            continue
        # Greek μ and ν (lowercase)
        if c in ('μ', 'Μ'):
            tokens.append(("MU", None))
            i += 1
            continue
        if c in ('ν', 'Ν'):
            tokens.append(("NU", None))
            i += 1
            continue
        # Alphabetic word: reserved keyword, proposition, or variable
        if c.isalpha():
            j = i
            while j < n and formula_str[j].isalpha():
                j += 1
            word = formula_str[i:j]
            lw = word.lower()
            if lw == "mu":
                tokens.append(("MU", None))
            elif lw == "nu":
                tokens.append(("NU", None))
            elif lw == "true":
                tokens.append(("BOOL", True))
            elif lw == "false":
                tokens.append(("BOOL", False))
            elif lw == "not":
                tokens.append(("NOT", None))
            elif lw == "and":
                tokens.append(("AND", None))
            elif lw == "or":
                tokens.append(("OR", None))
            else:
                # Uppercase initial -> fixpoint variable, lowercase -> proposition
                if word[0].isupper():
                    tokens.append(("VAR", word))
                else:
                    tokens.append(("PROP", word))
            i = j
            continue
        # Multi‑character modal operators: [] (box) and <> (diamond)
        if formula_str.startswith("[]", i) or formula_str.startswith("□", i):
            tokens.append(("BOX", None))
            i += 2 if formula_str.startswith("[]", i) else 1
            continue
        if formula_str.startswith("<>", i) or formula_str.startswith("◇", i):
            tokens.append(("DIAMOND", None))
            i += 2 if formula_str.startswith("<>", i) else 1
            continue
        # Single‑character punctuation and logical operators
        if c == '(': tokens.append(("LPAREN", None)); i += 1; continue
        if c == ')': tokens.append(("RPAREN", None)); i += 1; continue
        if c == '.': tokens.append(("DOT", None)); i += 1; continue
        if c == '|':
            tokens.append(("OR", None)); i += 1
            if i < n and formula_str[i] == '|': i += 1
            continue
        if c == '&':
            tokens.append(("AND", None)); i += 1
            if i < n and formula_str[i] == '&': i += 1
            continue
        if c == '!':
            tokens.append(("NOT", None)); i += 1; continue
        # Ignore other characters
        i += 1
    return tokens


class FormulaParser:
    """Recursive descent parser producing an abstract syntax tree (AST)."""
    def __init__(self, tokens: List[Tuple[str, Optional[object]]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Tuple[str, Optional[object]]]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def get(self) -> Optional[Tuple[str, Optional[object]]]:
        tok = self.peek()
        if tok:
            self.pos += 1
        return tok

    def parse_formula(self) -> Tuple:
        return self.parse_or()

    def parse_or(self) -> Tuple:
        node = self.parse_and()
        while True:
            tok = self.peek()
            if tok and tok[0] == "OR":
                self.get()
                right = self.parse_and()
                node = ("OR", node, right)
            else:
                break
        return node

    def parse_and(self) -> Tuple:
        node = self.parse_factor()
        while True:
            tok = self.peek()
            if tok and tok[0] == "AND":
                self.get()
                right = self.parse_factor()
                node = ("AND", node, right)
            else:
                break
        return node

    def parse_factor(self) -> Tuple:
        tok = self.peek()
        if not tok:
            raise SyntaxError("Unexpected end of formula")
        ttype, tval = tok
        if ttype == "NOT":
            self.get()
            sub = self.parse_factor()
            return ("NOT", sub)
        if ttype == "BOOL":
            self.get()
            return ("BOOL", tval)
        if ttype == "PROP":
            self.get()
            return ("PROP", tval)
        if ttype == "VAR":
            self.get()
            return ("VAR", tval)
        if ttype == "LPAREN":
            self.get()
            node = self.parse_formula()
            if not self.get() or self.tokens[self.pos - 1][0] != "RPAREN":
                raise SyntaxError("Missing closing parenthesis")
            return node
        if ttype in ("MU", "NU"):
            self.get()
            next_tok = self.get()
            if not next_tok or next_tok[0] != "VAR":
                raise SyntaxError(f"Expected a variable after {ttype.lower()}")
            var_name = next_tok[1]
            dot_tok = self.get()
            if not dot_tok or dot_tok[0] != "DOT":
                raise SyntaxError(f"Expected '.' after {ttype.lower()} {var_name}")
            body = self.parse_formula()
            return (ttype, var_name, body)
        if ttype == "BOX":
            self.get()
            sub = self.parse_factor()
            return ("BOX", sub)
        if ttype == "DIAMOND":
            self.get()
            sub = self.parse_factor()
            return ("DIAMOND", sub)
        raise SyntaxError(f"Unexpected token {tok}")


def parse_formula(formula_str: str) -> Tuple:
    """Parse a formula string into an abstract syntax tree (AST)."""
    tokens = tokenize(formula_str)
    parser = FormulaParser(tokens)
    ast = parser.parse_formula()
    if parser.pos != len(tokens):
        raise SyntaxError(f"Unexpected input after formula: {tokens[parser.pos:]}")
    return ast


# ---------------------------------------------------------------------------
# Formula evaluation and Ordinal Folding Index (OFI)
# ---------------------------------------------------------------------------

def eval_formula(node: Tuple, env: Dict[str, Set[int]]) -> Set[int]:
    """Evaluate the formula AST node under the given environment.

    The environment maps fixpoint variables to sets of states.  Atomic
    propositions and variables are resolved to sets of states; modal operators
    compute pre‑images via the model transitions; fixpoint binders are
    unfolded by iterative approximation.
    """
    ttype = node[0]
    if ttype == "PROP":
        prop = node[1]
        return atomic_valuations.get(prop, set())
    if ttype == "BOOL":
        val = node[1]
        return set(states) if val else set()
    if ttype == "VAR":
        varname = node[1]
        return env.get(varname, set())
    if ttype == "OR":
        _, left, right = node
        return eval_formula(left, env) | eval_formula(right, env)
    if ttype == "AND":
        _, left, right = node
        return eval_formula(left, env) & eval_formula(right, env)
    if ttype == "NOT":
        _, sub = node
        return set(states) - eval_formula(sub, env)
    if ttype == "BOX":
        _, sub = node
        sub_set = eval_formula(sub, env)
        result: Set[int] = set()
        for s in states:
            succ = transitions.get(s, [])
            if not succ:
                result.add(s)
            else:
                all_ok = True
                for t in succ:
                    if t not in sub_set:
                        all_ok = False
                        break
                if all_ok:
                    result.add(s)
        return result
    if ttype == "DIAMOND":
        _, sub = node
        sub_set = eval_formula(sub, env)
        result: Set[int] = set()
        for s in states:
            succ = transitions.get(s, [])
            for t in succ:
                if t in sub_set:
                    result.add(s)
                    break
        return result
    if ttype in ("MU", "NU"):
        inner_type, var, body = node
        is_mu = (inner_type == "MU")
        inner_current: Set[int] = set() if is_mu else set(states)
        inner_env = env.copy()
        while True:
            inner_env[var] = inner_current
            new_set = eval_formula(body, inner_env)
            if new_set == inner_current:
                return new_set
            inner_current = new_set
    raise ValueError(f"Unknown AST node type {ttype}")


def compute_ofi(formula_ast: Tuple) -> int:
    """Compute the Ordinal Folding Index (OFI) for a formula.

    The OFI corresponds to the iteration count required for convergence of
    the least or greatest fixpoint in the top‑level formula.  The function
    prints a certificate of convergence showing each approximation and
    returns the stage number at which the fixpoint is reached.
    """
    if formula_ast[0] not in ("MU", "NU"):
        result_set = eval_formula(formula_ast, {})
        print("Formula is non‑self‑referential; no unfolding needed. OFI = 0")
        print(f"Result: {sorted(result_set)}")
        return 0
    fixpoint_type, var, body = formula_ast
    is_mu = (fixpoint_type == "MU")
    current: Set[int] = set() if is_mu else set(states)
    stage = 0
    env: Dict[str, Set[int]] = {}
    print(f"Stage {stage}: {sorted(current)}")
    while True:
        env[var] = current
        new_set = eval_formula(body, env)
        if new_set == current:
            print(f"Converged at stage {stage}. OFI({fixpoint_type}{var}.<body>) = {stage}")
            return stage
        if is_mu:
            delta = new_set - current
        else:
            delta = current - new_set
        stage += 1
        sorted_new = sorted(new_set)
        if delta:
            change_type = "added" if is_mu else "removed"
            sorted_delta = sorted(delta)
            print(f"Stage {stage}: {sorted_new}   ({change_type}: {sorted_delta})")
        else:
            print(f"Stage {stage}: {sorted_new}")
        current = new_set
        if stage > 10000:
            print("WARNING: Iteration limit reached (possible non‑convergence).")
            return stage


# ---------------------------------------------------------------------------
# GloVe embedding loading and neighbour search
# ---------------------------------------------------------------------------

embeddings_loaded: bool = False
word_to_index: Dict[str, int] = {}
index_to_word: List[str] = []
embedding_matrix: Optional[np.ndarray] = None


def load_glove_embeddings(file_path: str = "./glove.6B.300d.txt") -> bool:
    """Load GloVe 6B.300d embeddings from a text file.

    The embeddings are normalised to unit length for cosine similarity
    computations.  The global variables ``word_to_index``, ``index_to_word``,
    ``embedding_matrix`` and ``embeddings_loaded`` are populated on success.
    Returns True if loading succeeded, False otherwise.
    """
    global embeddings_loaded, word_to_index, index_to_word, embedding_matrix
    if embeddings_loaded:
        return True
    if np is None:
        print("NumPy is required for embedding loading.")
        return False
    try:
        with open(file_path, 'r', encoding='utf8') as f:
            first_line = f.readline()
            if not first_line:
                print("Error: GloVe file is empty or unreadable.")
                return False
            parts = first_line.strip().split()
            dim = len(parts) - 1
        vocab: List[str] = []
        vectors: List[List[float]] = []
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < dim + 1:
                    continue
                word = parts[0]
                try:
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                if len(vals) != dim:
                    continue
                vocab.append(word)
                vectors.append(vals)
        if not vocab:
            print("Error: no embeddings found in the file.")
            return False
        if np:
            mat = np.array(vectors, dtype='float32')
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embedding_matrix = mat / norms
        else:
            embedding_matrix = None
        word_to_index = {w: i for i, w in enumerate(vocab)}
        index_to_word = vocab
        embeddings_loaded = True
        print(f"Loaded GloVe embeddings: {len(vocab)} words, dimension {dim}.")
        return True
    except FileNotFoundError:
        print(f"GloVe embeddings file not found at {file_path}. Please set the correct path.")
        return False


def get_neighbors(word: str, similarity_threshold: float, fanout: int) -> List[Tuple[str, float]]:
    """Return up to ``fanout`` nearest neighbours of ``word`` with similarity ≥ threshold.

    This helper replicates the functionality of the original DFS chain.  It
    computes cosine similarity to all vectors (which is efficient thanks to
    NumPy) and selects those exceeding ``similarity_threshold``.
    """
    if not embeddings_loaded or embedding_matrix is None or np is None:
        return []
    if word not in word_to_index:
        return []
    idx = word_to_index[word]
    vec = embedding_matrix[idx]
    sims = embedding_matrix.dot(vec)
    sims[idx] = -1.0
    if fanout <= 0:
        return []
    # Select candidate indices by partial sort
    if fanout < len(sims):
        top_idx = np.argpartition(sims, -fanout)[-fanout:]
    else:
        top_idx = np.arange(len(sims))
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    neighbors: List[Tuple[str, float]] = []
    for j in top_idx:
        if sims[j] < similarity_threshold:
            break
        neighbors.append((index_to_word[j], float(sims[j])))
        if len(neighbors) >= fanout:
            break
    return neighbors


# ---------------------------------------------------------------------------
# Taxonomic fixed‑point clustering for amphibian retrieval
# ---------------------------------------------------------------------------

def _compute_projection_basis(
    anchor_words: List[str],
    var_threshold: float = 0.95,
    r_min: int = 10,
) -> Optional[np.ndarray]:
    """
    Construct a projection basis from a list of ``anchor_words`` using
    principal component analysis (PCA).

    This function computes the singular value decomposition of the centred
    anchor embeddings and retains enough principal components to explain
    a specified fraction of the total variance.  The resulting matrix
    has shape (d, r), where r ≤ len(anchor_words) and captures the
    dominant directions of variation.

    Parameters
    ----------
    anchor_words : list of str
        Candidate words defining the amphibian lexicon.  Only anchors
        present in the loaded vocabulary will be used.
    var_threshold : float, optional
        Desired cumulative variance explained by the principal
        components.  Components are selected until this threshold is
        reached.  Defaults to 0.95 (95 %).
    min_dim : int, optional
        Minimum number of components to return.  Even if the variance
        threshold is satisfied with fewer components, at least this many
        will be used.  Defaults to 1.

    Returns
    -------
    numpy.ndarray or None
        A matrix ``B`` of shape (d, r) whose columns form an
        orthonormal basis for the retained principal subspace.  If no
        anchors are available or an error occurs, returns ``None``.
    """
    if not embeddings_loaded or embedding_matrix is None or np is None:
        return None
    # Collect embeddings for anchors that exist in vocabulary
    vecs: List[np.ndarray] = []
    for w in anchor_words:
        idx = word_to_index.get(w)
        if idx is not None:
            vecs.append(embedding_matrix[idx])
    if not vecs:
        return None
    A = np.stack(vecs, axis=0)  # shape (m, d)
    # Subtract the mean to centre the data
    mean_vec = A.mean(axis=0, keepdims=True)
    A_centered = A - mean_vec
    try:
        U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
    except Exception:
        return None
    if S.size == 0:
        return None
    # Drop directions corresponding to effectively zero singular values
    # The tolerance is scaled by the machine epsilon and a large factor (1e8)
    tol = S.max() * np.finfo(S.dtype).eps * 1e8
    mask = S > tol
    Vt = Vt[mask]
    S = S[mask]
    if S.size == 0:
        return None
    # Normalise by total variance of retained singular values
    total_var = float(np.sum(S ** 2))
    if total_var <= 0.0:
        return None
    cum = np.cumsum(S * S) / total_var
    # Determine rank needed to reach the desired variance fraction
    r = int(np.searchsorted(cum, var_threshold) + 1)
    r = max(r, r_min)
    r = min(r, Vt.shape[0])
    # Use the first r right singular vectors.  They are already orthonormal.
    B = Vt[:r].T
    return B.astype(np.float32)

# ---------------------------------------------------------------------------
# Anchor augmentation helper
# ---------------------------------------------------------------------------

def _augment_anchors(anchor_words: List[str], k_extra: int = 5) -> List[str]:
    """
    Expand a list of anchor words by including additional nearest
    neighbours for each anchor.  This heuristic is used when the user
    supplies only a handful of anchors and the resulting PCA projector
    would have too low rank.  Each supplied word contributes up to
    ``k_extra`` nearest neighbours (by cosine similarity) from the
    embedding space.  OOV words are ignored.  Duplicate entries are
    removed while preserving the original order.

    Parameters
    ----------
    anchor_words : list of str
        Initial anchor lexicon supplied by the user.
    k_extra : int, optional
        Number of nearest neighbours to append for each anchor.  Defaults
        to 5.

    Returns
    -------
    list of str
        The augmented anchor list containing the originals plus extra
        neighbours.
    """
    if not embeddings_loaded or embedding_matrix is None or np is None:
        return anchor_words
    extra: List[str] = []
    for w in anchor_words:
        idx = word_to_index.get(w)
        if idx is None:
            continue
        # Compute similarities to all vectors
        sims = embedding_matrix.dot(embedding_matrix[idx])
        # Find the top k_extra + 1 indices (including the anchor itself)
        # We use argpartition for efficiency and then sort those
        if sims.size <= k_extra + 1:
            nn_idx = np.arange(sims.size)
        else:
            nn_idx = np.argpartition(sims, -(k_extra + 1))[-(k_extra + 1):]
        # Sort the selected indices by similarity descending
        nn_idx = nn_idx[np.argsort(sims[nn_idx])[::-1]]
        for j in nn_idx:
            if j == idx:
                continue
            extra.append(index_to_word[j])
    # Deduplicate while preserving order
    merged = []
    seen: Set[str] = set()
    for w in anchor_words + extra:
        if w not in seen:
            seen.add(w)
            merged.append(w)
    return merged

# ---------------------------------------------------------------------------
# Unsupervised subspace learning and fixed‑point neighbourhood construction
# ---------------------------------------------------------------------------

def _learn_projection(C_idxs: List[int], rho: float = 0.95, r_min: int = 5) -> Optional[np.ndarray]:
    """
    Learn a principal subspace from the embeddings indexed by ``C_idxs``.

    This routine computes the singular value decomposition (SVD) of the
    centred embeddings of the current candidate set and retains the
    minimal number of principal components whose cumulative variance
    explains at least ``rho`` of the total.  A minimum number of
    components ``r_min`` is enforced to avoid degeneracy.  The returned
    matrix has shape (d, r) with orthonormal columns.

    Parameters
    ----------
    C_idxs : list of int
        Indices of words in the current candidate set.
    rho : float, optional
        Fraction of variance to retain when selecting principal
        components.  Defaults to 0.95.
    r_min : int, optional
        Minimum number of components to retain regardless of variance.
        Defaults to 5.

    Returns
    -------
    numpy.ndarray or None
        A matrix B of shape (d, r) whose columns form an orthonormal
        basis for the retained subspace, or None if ``C_idxs`` is
        empty or if embeddings are unavailable.
    """
    if not C_idxs or embedding_matrix is None or np is None:
        return None
    # Extract embeddings and centre them
    X = embedding_matrix[C_idxs].astype('float64')
    # Centre by subtracting mean
    X -= X.mean(axis=0, keepdims=True)
    # Perform SVD; we only need the right singular vectors
    try:
        U, S, VT = np.linalg.svd(X, full_matrices=False)
    except Exception:
        return None
    if S.size == 0:
        return None
    # Drop singular directions with effectively zero singular values to avoid
    # ill‑conditioned projectors.  Tolerance is scaled by machine epsilon.
    tol = S.max() * np.finfo(S.dtype).eps * 1e8
    mask = S > tol
    VT = VT[mask]
    S = S[mask]
    if S.size == 0:
        return None
    # Compute cumulative variance ratio
    var = np.cumsum(S * S) / np.sum(S * S)
    # Determine rank r to retain rho variance, enforce minimum r_min
    r = int(np.searchsorted(var, rho) + 1)
    r = max(r, r_min)
    r = min(r, VT.shape[0])
    # Since VT is orthonormal (from SVD), the first r rows already form an
    # orthonormal set of principal directions.  We avoid any further
    # normalization here to preserve orthogonality.  Return the transpose
    # (d×r) as the projection basis.
    # Transpose to obtain a d×r basis
    B = VT[:r].T.astype('float64')
    # Normalise each column of B to unit length to avoid extremely
    # small or large norms; this mitigates overflow when multiplying
    # candidate embeddings by B.  Columns correspond to principal
    # directions, and scaling them does not change the subspace they
    # span.
    col_norms = np.linalg.norm(B, axis=0, keepdims=True)
    # Avoid division by zero
    col_norms[col_norms == 0] = 1.0
    B /= col_norms
    # Replace any NaN or infinite values that may arise from numerical issues.
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    return B


def fixed_point_neighbourhood(
    seed_idx: int,
    tau0: float,
    eta0: float = 0.05,
    rho: float = 0.95,
    max_iter: int = 6,
    fp_size_cap: Optional[int] = None,
) -> Tuple[List[int], Optional[np.ndarray]]:
    """
    Construct a fixed‑point candidate set and learned projector via
    iterative neighbourhood expansion.

    Starting from the seed index ``seed_idx`` and an initial
    similarity threshold ``tau0``, this function builds a candidate set
    ``C`` of word indices satisfying the neighbourhood and energy
    conditions.  At each iteration the principal subspace of the
    current set ``C`` is estimated via PCA, an energy filter is
    applied, and the set is re‑expanded by taking τ₀‑neighbours of the
    surviving vectors.  The iteration stops when ``C`` stabilises or
    when ``max_iter`` iterations have elapsed.  The final candidate
    indices and the learned projection basis are returned.

    Parameters
    ----------
    seed_idx : int
        Index of the seed word in ``embedding_matrix``.
    tau0 : float
        Similarity threshold for selecting neighbours.
    eta0 : float, optional
        Initial energy ratio threshold.  Defaults to 0.05.
    rho : float, optional
        Fraction of variance to retain when estimating the principal
        subspace.  Defaults to 0.95.
    max_iter : int, optional
        Maximum number of expansion iterations.  Defaults to 6.
    fp_size_cap : int or None, optional
        Optional cap on the maximum size of ``C``.  If provided and
        the candidate set exceeds this size, only the ``fp_size_cap``
        most similar neighbours to the seed are kept to control
        complexity.  Defaults to None (no cap).

    Returns
    -------
    (list of int, numpy.ndarray or None)
        A tuple ``(C_idxs, B)`` where ``C_idxs`` is the list of
        candidate indices at the fixed point and ``B`` is the learned
        projection basis of shape (d, r).  ``B`` may be None if the
        projection could not be learned (e.g. due to empty candidate
        set).
    """
    if embedding_matrix is None or np is None:
        return [], None
    # Initial neighbourhood: all indices with cosine similarity ≥ tau0
    sims_seed = embedding_matrix.dot(embedding_matrix[seed_idx])
    # Select candidates above threshold, optionally cap the size
    candidate_idxs = set(int(i) for i in np.where(sims_seed >= tau0)[0])
    candidate_idxs.add(int(seed_idx))
    # Optionally cap the size of the initial candidate set
    if fp_size_cap is not None and len(candidate_idxs) > fp_size_cap:
        # Keep the top fp_size_cap indices by similarity to the seed
        top_idx = np.argpartition(sims_seed, -fp_size_cap)[-fp_size_cap:]
        candidate_idxs = set(int(i) for i in top_idx)
        candidate_idxs.add(int(seed_idx))
    C = candidate_idxs
    eta = eta0
    B = None
    for _ in range(max_iter):
        # Learn projection basis from current candidate set
        B = _learn_projection(list(C), rho=rho)
        if B is None:
            break
        # Compute energies for all candidates without forming the full projector.
        # For each candidate vector v, the energy in the learned subspace is
        # \|P v\|^2 = \|B^\top v\|^2 since B has orthonormal columns.  To avoid
        # constructing the full d×d projector P = B B^T, we compute the
        # coordinates of each v in the subspace via V @ B (shape |C|×r) and
        # take the squared norm of those coordinates.
        # Filter candidate indices to retain only those with finite embeddings
        C_list = [idx for idx in C if np.all(np.isfinite(embedding_matrix[idx]))]
        # If no finite vectors remain, stop
        if not C_list:
            C = set()
            break
        # Update the candidate set to the filtered list to avoid propagating
        # invalid embeddings in subsequent iterations
        C = set(C_list)
        V = embedding_matrix[C_list]
        # Compute energies using safe per-vector operations to avoid
        # potential overflow/underflow in large matrix multiplications.  For
        # each candidate vector v, compute its coordinates in the learned
        # subspace via B^T v, then take the squared norm of those
        # coordinates.  This loop may be slower for large |C| but is more
        # numerically stable than forming V @ B at once.
        energies = []
        # Compute energies using the dot product to avoid potential
        # matmul issues.  coords_i = v @ B computes the coordinates in
        # the subspace.  This is equivalent to B.T @ v but uses
        # numpy.dot directly, which can be more stable on some platforms.
        for v in V:
            # Compute coordinates and energy inside an errstate context to
            # suppress warnings from any spurious numerical issues (e.g. divide by zero)
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                coords_i = np.dot(v, B)  # shape (r,)
                energies.append(float(np.dot(coords_i, coords_i)))
        # Filter candidates by energy threshold
        keep = [idx for idx, er in zip(C_list, energies) if er >= eta]
        if not keep:
            # If none survive, stop
            C = set()
            break
        # Re‑expand: union of τ₀‑neighbours of surviving vectors and the survivors themselves
        C_new: Set[int] = set()
        for idx in keep:
            sims = embedding_matrix.dot(embedding_matrix[idx])
            # Add neighbours above threshold
            C_new.update(int(i) for i in np.where(sims >= tau0)[0])
            # Always include the survivor itself
            C_new.add(int(idx))
        # Optionally cap size of C_new as above
        if fp_size_cap is not None and len(C_new) > fp_size_cap:
            # Keep top neighbours by similarity to seed
            sims_seed = embedding_matrix.dot(embedding_matrix[seed_idx])
            top_idx = np.argpartition(sims_seed, -fp_size_cap)[-fp_size_cap:]
            C_new = set(int(i) for i in top_idx)
            C_new.add(int(seed_idx))
        # Check for fixed point
        if C_new == C:
            C = C_new
            break
        C = C_new
        # Update eta schedule: increase by factor 1.2 but cap at 0.3
        eta = min(0.3, 1.2 * eta)
    return list(C), B



# ---------------------------------------------------------------------------
# PCA‑based soft spectral filter clustering
# ---------------------------------------------------------------------------

def cluster_amphibian_soft(
    seed_word: str,
    anchor_words: Optional[List[str]] = None,
    tau_threshold: Optional[float] = None,
    lambda_val: float = 0.5,
    beta: float = 0.9,
    var_threshold: float = 0.95,
    eta: float = 0.05,
    epsilon: float = 1e-6,
    max_iter: int = 10,
    top_k: int = 50,
    fp_iters: int = 6,
    fp_cap: Optional[int] = None,
) -> Optional[List[Tuple[str, float]]]:
    """
    Perform unsupervised clustering around ``seed_word`` via a learned
    projection and fixed‑point neighbourhood expansion.

    This implementation ignores any external anchor lists and learns a
    domain‑specific subspace directly from the seed's neighbourhood.
    The process proceeds as follows:

    1. Select an initial candidate set ``C`` of words whose cosine
       similarity to the seed is at least ``tau_threshold`` (if
       provided) or an adaptively chosen threshold if ``tau_threshold``
       is None.
    2. Repeatedly learn a principal subspace from ``C`` (retaining
       ``var_threshold`` of the variance) and filter ``C`` by the
       energy ratio threshold ``eta``.  Then re‑expand ``C`` by adding
       τ‑neighbours of the surviving vectors.  Increase ``eta`` by
       20 % each iteration up to 0.3.  Stop when the set stabilises or
       after ``fp_iters`` iterations.
    3. With the learned subspace projector ``P`` and the final
       candidate set ``C``, compute the barycentre of the projected
       vectors.  Score each word by the dot product between its
       projection and the barycentre projection.  Return the top
       ``top_k`` words sorted by score.

    Parameters
    ----------
    seed_word : str
        The seed around which to cluster.
    anchor_words : unused
        Retained for backward compatibility but ignored.
    tau_threshold : float or None, optional
        Initial similarity threshold τ₀.  If None, it is computed
        adaptively from the seed's similarity distribution.  Defaults
        to None.
    lambda_val : float, optional
        Unused in this unsupervised clustering (kept for signature
        compatibility).
    beta : float, optional
        Unused (retained for signature compatibility).
    var_threshold : float, optional
        Fraction of variance ρ to retain when learning the subspace.
        Defaults to 0.95.
    eta : float, optional
        Initial energy threshold η₀.  Defaults to 0.05.
    epsilon : float, optional
        Unused (retained for signature compatibility).
    max_iter : int, optional
        Unused (retained for signature compatibility).
    top_k : int, optional
        Number of cluster members to return.  Defaults to 50.
    fp_iters : int, optional
        Maximum number of neighbourhood expansion iterations.  Defaults
        to 6.
    fp_cap : int or None, optional
        Optional cap on the size of the candidate set during fixed
        point expansion.  Defaults to None.

    Returns
    -------
    list of (str, float) or None
        The top ``top_k`` cluster members and their taxonomic scores,
        or None if the embeddings are unavailable or the seed is OOV.
    """
    # Preconditions
    if not embeddings_loaded or embedding_matrix is None or np is None:
        return None
    if seed_word not in word_to_index:
        return None
    seed_idx = word_to_index[seed_word]
    # Adaptive threshold if not provided
    if tau_threshold is None:
        sims_all = embedding_matrix.dot(embedding_matrix[seed_idx])
        sims_sorted = np.sort(sims_all)[::-1]
        if sims_sorted.size > 9:
            tau_threshold = 0.5 * (sims_sorted[8] + sims_sorted[9])
        else:
            q = max(2, sims_sorted.size // 4)
            tau_threshold = float(np.mean(sims_sorted[1:q])) if q > 1 else 0.25
    tau0 = float(tau_threshold)
    # Learn fixed‑point neighbourhood and projector
    C_idxs, B = fixed_point_neighbourhood(
        seed_idx,
        tau0=tau0,
        eta0=eta,
        rho=var_threshold,
        max_iter=fp_iters,
        fp_size_cap=fp_cap,
    )
    if not C_idxs or B is None:
        return None
    # Filter candidate indices to those with finite embeddings
    C_idxs = [idx for idx in C_idxs if np.all(np.isfinite(embedding_matrix[idx]))]
    if not C_idxs:
        return None
    # Instead of forming the full d×d projector P = B @ B.T, compute
    # projections using the basis and its transpose.  For each vector v,
    # the projection is p_v = B @ (B^T v).  Compute the barycentre of
    # projections by first obtaining coordinates for all candidates in
    # the subspace and then mapping the mean coordinates back.
    vecs = embedding_matrix[C_idxs]
    # Compute coordinates of each vector in the learned subspace using a
    # safe per‑vector loop to avoid overflow in a large matmul.  Store
    # coordinates as a list of r‑dimensional vectors.
    coords_list = []
    # Compute coordinates using dot product: coords_i = v @ B
    for v in vecs:
        # Compute coordinates in errstate context to avoid warnings
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            coords_list.append(np.dot(v, B))
    coords_all = np.stack(coords_list, axis=0)
    # Average coordinates (mean point in subspace)
    avg_coords = coords_all.mean(axis=0)
    # Barycentre of projections
    p_bar = B @ avg_coords
    # Score each candidate by the inner product between its projection and
    # the barycentre: score(v) = ⟨P v, p_bar⟩ where P v = B (B^T v).
    scores = []
    for idx in C_idxs:
        v = embedding_matrix[idx]
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            coords_v = np.dot(v, B)
            p_v = B @ coords_v
            score = float(p_v.dot(p_bar))
        scores.append((index_to_word[idx], score))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k] if top_k > 0 else scores

# ---------------------------------------------------------------------------
# Improved semantic chain algorithm
# ---------------------------------------------------------------------------

def compute_relevance_fixedpoint(
    word: str,
    tau: Optional[float] = None,
    lambda_val: float = 0.60,
    max_iters: int = 20,
    eps: float = 1e-6,
    top_k: int = 10,
    eta: float = 0.05,
    filter_anchors: Optional[List[str]] = None,
    var_threshold: float = 0.95,
    fp_iters: int = 6,
    fp_cap: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Compute a relevance ranking for ``word`` via an unsupervised fixed‑point
    neighbourhood and random walk.

    This implementation discards external anchor lists entirely and
    instead learns a principal subspace from the seed's τ‑neighbourhood
    using an iterative fixed‑point expansion.  At each step, the
    candidate set is filtered by an energy threshold derived from the
    learned projector and re‑expanded by τ‑neighbours.  When the set
    stabilises, a random walk confined to this set is executed and the
    resulting relevance scores are multiplied by the energy ratios to
    yield the final ranking.

    Parameters
    ----------
    word : str
        The seed word.
    tau : float or None, optional
        Similarity threshold τ₀ for the neighbourhood.  If None,
        τ₀ is chosen adaptively as the average of the 8th and 9th
        largest similarities to the seed.  Defaults to None.
    lambda_val : float, optional
        Mixing parameter for the random walk.  Defaults to 0.60.
    max_iters : int, optional
        Maximum number of iterations for the random walk (not the
        neighbourhood expansion).  Defaults to 20.
    eps : float, optional
        Convergence tolerance for the random walk.  Defaults to 1e‑6.
    top_k : int, optional
        Number of results to return.  Defaults to 10.
    eta : float, optional
        Initial energy threshold η₀ for the neighbourhood expansion.
        Defaults to 0.05.
    filter_anchors : list of str or None, optional
        Ignored in this unsupervised implementation; retained for
        backwards compatibility.
    var_threshold : float, optional
        Variance retention ρ used when learning the principal subspace.
        Defaults to 0.95.
    fp_iters : int, optional
        Maximum number of neighbourhood expansion iterations.  Defaults
        to 6.
    fp_cap : int or None, optional
        Optional cap on the size of the candidate set during fixed
        point expansion.  If None, no cap is applied.  Defaults to
        None.

    Returns
    -------
    list of (str, float)
        The top ``top_k`` words sorted by relevance·energy score.  If
        the seed is missing or the embedding has not been loaded,
        returns an empty list.
    """
    # Preconditions
    if not embeddings_loaded or embedding_matrix is None or np is None:
        return []
    if word not in word_to_index:
        return []
    seed_idx = word_to_index[word]
    # Choose tau adaptively if needed
    if tau is None:
        sims_all = embedding_matrix.dot(embedding_matrix[seed_idx])
        sims_sorted = np.sort(sims_all)[::-1]
        if sims_sorted.size > 9:
            tau = 0.5 * (sims_sorted[8] + sims_sorted[9])
        else:
            q = max(2, sims_sorted.size // 4)
            tau = float(np.mean(sims_sorted[1:q])) if q > 1 else 0.25
    tau0 = float(tau)
    # Learn the fixed‑point neighbourhood and projector
    # Cap the candidate set size by default if not explicitly provided.  A
    # moderate cap (e.g. 500) prevents extremely large SVDs and helps
    # numerical stability on massive vocabularies.
    cap_value = fp_cap if fp_cap is not None else 500
    C_idxs, B = fixed_point_neighbourhood(
        seed_idx,
        tau0=tau0,
        eta0=eta,
        rho=var_threshold,
        max_iter=fp_iters,
        fp_size_cap=cap_value,
    )
    if not C_idxs:
        return []
    # Filter out indices whose embeddings contain non‑finite values to
    # prevent numerical issues in subsequent projections
    C_idxs = [idx for idx in C_idxs if np.all(np.isfinite(embedding_matrix[idx]))]
    if not C_idxs:
        return []
    # Order nodes with seed first
    nodes = [seed_idx] + [i for i in C_idxs if i != seed_idx]
    N = len(nodes)
    # Build adjacency matrix for random walk
    P_graph = np.zeros((N, N), dtype=np.float32)
    # Precompute similarities for each node
    for i, gi in enumerate(nodes):
        vec_i = embedding_matrix[gi]
        sims_i = embedding_matrix.dot(vec_i)
        row_sum = 0.0
        for j, gj in enumerate(nodes):
            if i == j:
                continue
            sim_ij = sims_i[gj]
            if sim_ij >= tau0:
                P_graph[i, j] = sim_ij
                row_sum += sim_ij
        if row_sum > 0.0:
            P_graph[i] /= row_sum
    # Seed vector
    s_vec = np.zeros(N, dtype=np.float32)
    s_vec[0] = 1.0
    # Random walk relevance vector
    r_vec = s_vec.copy()
    for _ in range(max_iters):
        r_next = (1.0 - lambda_val) * s_vec + lambda_val * P_graph.dot(r_vec)
        if np.linalg.norm(r_next - r_vec, ord=1) < eps:
            r_vec = r_next
            break
        r_vec = r_next
    # Compute energy ratios using the learned projector B
    final: List[Tuple[str, float]] = []
    if B is not None:
        for i, idx in enumerate(nodes):
            tok = index_to_word[idx]
            rscore = float(r_vec[i])
            v = embedding_matrix[idx]
            v_norm_sq = float(v.dot(v))
            if v_norm_sq > 0.0:
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    coords = np.dot(v, B)  # shape (r,)
                    p_v = B @ coords
                    er = float(p_v.dot(p_v)) / v_norm_sq
            else:
                er = 0.0
            final.append((tok, rscore * er))
    else:
        # If no projection basis, fall back to raw relevance scores
        for i, idx in enumerate(nodes):
            tok = index_to_word[idx]
            final.append((tok, float(r_vec[i])))
    final.sort(key=lambda x: -x[1])
    return final[:top_k] if top_k > 0 else final


# ---------------------------------------------------------------------------
# Unsupervised greedy clustering (anchor‑free)
# ---------------------------------------------------------------------------
def cluster_rank_greedy(
    seed_word: str,
    tau_threshold: Optional[float] = None,
    lambda_val: float = 0.5,
    var_threshold: float = 0.95,
    eta: float = 0.05,
    top_k: int = 20,
    fp_iters: int = 6,
    fp_cap: Optional[int] = None,
) -> Optional[List[Tuple[str, float]]]:
    """
    Perform unsupervised clustering around ``seed_word`` using a learned
    principal subspace and a greedy ranking functional.

    This implementation learns a domain‑specific subspace from the
    seed's neighbourhood via ``fixed_point_neighbourhood``.  It then
    performs a greedy expansion of a set ``S`` starting from the seed,
    selecting at each step the candidate that maximises the score

        psi(v) = ⟨P v, bar_s⟩ − lambda_val * max_{u ∈ S} ⟨v, u⟩

    where ``P v = B @ (B.T @ v)`` is the projection of ``v`` onto the
    learned subspace and ``bar_s`` is the barycentre of the projected
    vectors in ``S``.  The process stops when no candidate yields a
    positive increment or when ``top_k`` elements have been selected.

    Parameters
    ----------
    seed_word : str
        The seed around which to cluster.
    tau_threshold : float or None, optional
        Initial similarity threshold τ₀.  If None, it is computed
        adaptively from the seed's similarity distribution (8th/9th
        nearest neighbours).  Defaults to None.
    lambda_val : float, optional
        Redundancy penalty λ.  Defaults to 0.5.
    var_threshold : float, optional
        Variance retention ρ used when learning the principal subspace.
        Defaults to 0.95.
    eta : float, optional
        Initial energy threshold η₀ for the fixed‑point neighbourhood.
        Defaults to 0.05.
    top_k : int, optional
        Maximum number of cluster members to return.  Defaults to 20.
    fp_iters : int, optional
        Maximum number of neighbourhood expansion iterations.  Defaults
        to 6.
    fp_cap : int or None, optional
        Optional cap on the size of the candidate set during fixed
        point expansion.  If None, a default of 500 is used.

    Returns
    -------
    list of (str, float) or None
        The top ``top_k`` cluster members and their scores, or None if
        the embeddings are unavailable or the seed is OOV.
    """
    # Preconditions
    if not embeddings_loaded or embedding_matrix is None or np is None:
        return None
    if seed_word not in word_to_index:
        return None
    seed_idx = word_to_index[seed_word]
    # Adaptive threshold if not provided
    if tau_threshold is None:
        sims_all = embedding_matrix.dot(embedding_matrix[seed_idx])
        sims_sorted = np.sort(sims_all)[::-1]
        if sims_sorted.size > 9:
            tau_threshold = 0.5 * (sims_sorted[8] + sims_sorted[9])
        else:
            q = max(2, sims_sorted.size // 4)
            tau_threshold = float(np.mean(sims_sorted[1:q])) if q > 1 else 0.25
    tau0 = float(tau_threshold)
    # Cap candidate set by default if not provided
    cap_value = fp_cap if fp_cap is not None else 500
    # Learn fixed‑point neighbourhood and projection basis
    C_idxs, B = fixed_point_neighbourhood(
        seed_idx,
        tau0=tau0,
        eta0=eta,
        rho=var_threshold,
        max_iter=fp_iters,
        fp_size_cap=cap_value,
    )
    if not C_idxs or B is None:
        return None
    # Filter candidate indices to those with finite embeddings
    C_idxs = [idx for idx in C_idxs if np.all(np.isfinite(embedding_matrix[idx]))]
    if not C_idxs:
        return None
    # Build mapping from index to projected vector for each candidate
    P_map: Dict[int, np.ndarray] = {}
    # Precompute B.T once
    BT = B.T
    for idx in C_idxs:
        v = embedding_matrix[idx]
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            coords = np.dot(v, B)  # shape (r,)
            p_v = B @ coords
        P_map[idx] = p_v
    # Initialise greedy set S with the seed
    S: List[int] = [seed_idx]
    bar_s = P_map[seed_idx].copy()
    # Greedy selection
    while len(S) < top_k:
        best_idx: Optional[int] = None
        best_phi: float = 0.0
        # Evaluate each candidate not yet in S
        for idx in C_idxs:
            if idx in S:
                continue
            p_v = P_map[idx]
            # Taxonomic proximity: inner product of projections
            taxon = float(p_v.dot(bar_s))
            # Redundancy: maximum cosine similarity to current set S
            redund = max(
                float(embedding_matrix[idx].dot(embedding_matrix[u])) for u in S
            )
            phi = taxon - lambda_val * redund
            if phi > best_phi:
                best_phi = phi
                best_idx = idx
        # Stop if no positive improvement
        if best_idx is None or best_phi <= 0.0:
            break
        # Add best candidate to S
        S.append(best_idx)
        # Update barycentre bar_s incrementally
        p_best = P_map[best_idx]
        n_old = len(S) - 1
        # bar_new = ((n_old)/n_new) * bar_old + (1/n_new) * p_best
        # where n_new = len(S)
        coef_old = n_old / len(S)
        coef_new = 1.0 / len(S)
        bar_s = coef_old * bar_s + coef_new * p_best
    # Compute final scores using projection onto bar_s
    cluster_list: List[Tuple[str, float]] = []
    for idx in S:
        p_v = P_map[idx]
        score = float(p_v.dot(bar_s))
        cluster_list.append((index_to_word[idx], score))
    cluster_list.sort(key=lambda x: -x[1])
    return cluster_list[:top_k] if top_k > 0 else cluster_list


# ---------------------------------------------------------------------------
# Accessibility solver for next-word prediction
# ---------------------------------------------------------------------------
def next_word_from_chain(
    seed_word: str,
    chain_result: List[Tuple[str, float]],
    tau: Optional[float] = None,
    gamma: float = 0.9,
    tol: float = 1e-6,
    top_k: Optional[int] = None,
    p: float = 1.3,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Given a seed word and a list of words with their relevance scores
    (as returned by the chain routine), compute a personalised‑PageRank
    accessibility value function and return the best successor along with
    the full ranking.

    The chain_result should include the seed as the first element.

    Parameters
    ----------
    seed_word : str
        The seed word (for clarity; unused internally except for
        validation).
    chain_result : list of (str, float)
        List of words and their relevance scores, starting with the seed.
        These scores are ignored for the personalised PageRank computation;
        the restart vector concentrates all mass at the seed.
    tau : float or None, optional
        Similarity threshold τ for the kernel.  If None, τ is set to
        the 95th percentile of the off‑diagonal cosine similarities.
    gamma : float, optional
        Discount factor γ in (0,1).  Defaults to 0.9.
    tol : float, optional
        Convergence tolerance for the personalised PageRank solver.
        Defaults to 1e‑6.
    top_k : int or None, optional
        If provided, only the top_k words (excluding the seed) are
        considered when selecting the next word.  If None, all
        available words are used.
    p : float, optional
        Exponent applied to each cosine similarity when constructing
        the kernel.  The weight between two words ``i`` and ``j`` is
        defined as ``w_ij = (s_ij)**p`` if ``s_ij ≥ τ`` and zero
        otherwise.  Values of ``p`` between 1 and 2 emphasise stronger
        links without shifting weaker similarities.  Default is 1.3.

    Returns
    -------
    (str, float, list)
        A tuple (best_word, best_score, ranking) where best_word is
        the recommended successor word, best_score is its accessibility
        value f(best_word), and ranking is the list of (word, f(word))
        sorted by descending f.
    """
    # Unpack words; ignore relevance scores since personalised PageRank
    # uses only the seed as the restart vector.  Filter out any words
    # whose embeddings are not finite.
    valid_words: List[str] = []
    for w, _ in chain_result:
        vec = embedding_matrix[word_to_index[w]]
        if np.all(np.isfinite(vec)):
            valid_words.append(w)
    if not valid_words:
        return seed_word, 0.0, []
    words = tuple(valid_words)
    m = len(words)
    # The seed is always at index 0
    seed_index = 0
    # Build pairwise cosine similarity matrix safely via nested loops to
    # avoid constructing a full d×d product that may overflow.  "dots"
    # stores s_ij = ⟨v_i, v_j⟩ for i,j in [0, m).
    V = np.stack([embedding_matrix[word_to_index[w]] for w in words])
    dots = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        vi = V[i]
        for j in range(i, m):
            vj = V[j]
            # Compute dot product with errstate protection
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                dot_ij = float(np.dot(vi, vj))
            dots[i, j] = dot_ij
            dots[j, i] = dot_ij
    # Determine tau if needed from the off‑diagonal similarities
    if tau is None:
        # Consider off‑diagonal entries only
        off_vals = [dots[i, j] for i in range(m) for j in range(m) if i != j]
        if off_vals:
            tau = float(np.quantile(off_vals, 0.95))
        else:
            tau = 0.0
    # Build kernel K and transition matrix P using the exponent weight.  The
    # kernel weight w_ij = (s_ij)**p if s_ij ≥ tau, otherwise 0.  This
    # preserves strong connections and discards weaker ones without
    # subtracting τ, avoiding the hard shift that breaks connectivity.
    K = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            s_ij = dots[i, j]
            # Apply threshold and exponent
            if s_ij >= tau:
                with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                    K[i, j] = float(s_ij) ** p
    # Row sums
    deg = np.sum(K, axis=1)
    P = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        if deg[i] > 0.0:
            P[i] = K[i] / deg[i]
    # Personalised PageRank: restart on seed.  e0 is the one‑hot seed
    e0 = np.zeros(m, dtype=np.float64)
    e0[seed_index] = 1.0
    restart_vec = (1.0 - gamma) * e0
    # Initialise f to e0 (mass starts entirely at seed)
    f = e0.copy()
    # Power iteration: f_{new} = (1-γ) e0 + γ P f
    while True:
        update = np.zeros(m, dtype=np.float64)
        # Compute P @ f using explicit dot products per row
        for i in range(m):
            if deg[i] > 0.0:
                update[i] = float(np.dot(P[i], f))
        f_new = restart_vec + gamma * update
        # Check convergence in L1 norm
        if np.abs(f_new - f).sum() < tol:
            f = f_new
            break
        f = f_new
    # Select best successor (exclude seed_index)
    if top_k is not None:
        idx_range = range(1, min(m, top_k + 1))
    else:
        idx_range = range(1, m)
    best_rel_val = -np.inf
    best_idx = seed_index + 1 if m > 1 else seed_index
    for i in idx_range:
        if f[i] > best_rel_val:
            best_rel_val = f[i]
            best_idx = i
    ranking = [(words[i], float(f[i])) for i in range(m)]
    ranking_sorted = sorted(ranking, key=lambda x: -x[1])
    best_word = words[best_idx]
    return best_word, float(f[best_idx]), ranking_sorted


# ---------------------------------------------------------------------------
# Command‑line interface
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for command‑line usage."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(
            "Usage:\n"
            "  updated_code.py \"<formula>\"\n"
            "     Evaluate a symbolic formula and compute its OFI.\n"
            "  updated_code.py /chain <word> [--lambda λ] [--tau τ] [--eta η] [--var ρ] [--top k] [--maxiter N] [--fp_iter M] [--fp_cap C] [--eps ε]\n"
            "     Compute a relevance ranking using an unsupervised fixed‑point neighbourhood and random walk.\n"
            "  updated_code.py /cluster <word> [--tau τ] [--var ρ] [--eta η] [--top k] [--fp_iter M] [--fp_cap C]\n"
            "     Perform unsupervised clustering via a learned projection from the seed neighbourhood.\n"
            "  updated_code.py /continue <word> [--gamma g] [--tau τ] [--top k] [--eps ε]\n"
            "     Recommend a successor word using accessibility scores on the chain graph.\n"
            "\n"
            "For example:\n"
            "  updated_code.py \"mu X. p or (<>X and not q)\"\n"
            "  updated_code.py /chain frog --tau 0.3 --top 10"
        )
        return
    # Determine primary command token
    arg0 = argv[0]
    # Detect continue mode (next-word prediction)
    if arg0.lower().startswith("/continue"):
        if len(argv) < 2:
            print("Please specify a seed word after /continue.")
            return
        # Defaults for continue mode
        gamma_val = 0.9                # discount factor
        tau_val: Optional[float] = None  # kernel threshold
        top_k = 10                     # number of top words to consider/return
        tol_val = 1e-6                # convergence tolerance for value solver
        # The second argument is the seed word
        seed_word = argv[1]
        # Parse options
        i = 2
        while i < len(argv):
            arg = argv[i]
            if arg == "--gamma":
                if i + 1 >= len(argv):
                    print("Error: --gamma requires a numeric value.")
                    return
                try:
                    gamma_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --gamma.")
                    return
                i += 2
                continue
            if arg in ("--tau", "-t"):
                if i + 1 >= len(argv):
                    print("Error: --tau requires a numeric value.")
                    return
                try:
                    tau_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --tau.")
                    return
                i += 2
                continue
            if arg in ("--top", "-k"):
                if i + 1 >= len(argv):
                    print("Error: --top requires an integer value.")
                    return
                try:
                    top_k = int(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --top.")
                    return
                i += 2
                continue
            if arg == "--eps":
                if i + 1 >= len(argv):
                    print("Error: --eps requires a numeric value.")
                    return
                try:
                    tol_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --eps.")
                    return
                i += 2
                continue
            # Additional tokens are appended to seed word
            seed_word += " " + arg
            i += 1
        if not load_glove_embeddings():
            return
        if seed_word not in word_to_index:
            print(f"Seed word '{seed_word}' not found in vocabulary.")
            return
        # Obtain a relevance chain using compute_relevance_fixedpoint with a
        # larger candidate list to ensure that important bigrams survive
        # the initial truncation.  A principled choice is c·log(m) with
        # c≈4, but here we set a floor of 60.  If ``top_k`` is larger,
        # add one extra to preserve the seed.
        try:
            import math
            # Determine an adaptive candidate size based on the total
            # vocabulary size (embedding_matrix may not be None here)
            vocab_size = embedding_matrix.shape[0] if embedding_matrix is not None else 400000
            adaptive_size = int(math.ceil(4.0 * math.log(max(vocab_size, 2))))
        except Exception:
            adaptive_size = 60
        top_k_chain = max(top_k + 1, adaptive_size, 60)
        chain_res = compute_relevance_fixedpoint(
            seed_word,
            tau=tau_val,
            lambda_val=0.6,
            max_iters=20,
            eps=1e-6,
            top_k=top_k_chain,
            eta=0.05,
            filter_anchors=None,
            var_threshold=0.95,
            fp_iters=6,
            fp_cap=None,
        )
        if not chain_res:
            print(f"No chain found for '{seed_word}'.")
            return
        # Solve the Bellman equation and select next word
        best_word, best_score, ranking = next_word_from_chain(
            seed_word,
            chain_res,
            tau=tau_val,
            gamma=gamma_val,
            tol=tol_val,
            top_k=top_k,
        )
        # Print the recommendation and full ranking
        print(f"Recommended successor for '{seed_word}': {best_word} (score={best_score:.3f})")
        print("Accessibility ranking:")
        for w, v in ranking[:max(top_k, len(ranking))]:
            print(f"    {w} (score={v:.3f})")
        return
    # Detect chain mode
    # Cluster mode: unsupervised fixed‑point clustering
    if arg0.lower().startswith("/cluster"):
        if len(argv) < 2:
            print("Please specify a seed word after /cluster.")
            return
        # Defaults for unsupervised clustering
        anchor_words: Optional[List[str]] = None  # unused
        tau_threshold: Optional[float] = None      # initial similarity threshold (None ⇒ adaptive)
        lambda_val = 0.5                           # unused (for compatibility)
        beta_val = 0.9                             # unused (for compatibility)
        var_val = 0.95                             # retained variance ρ
        eta_val = 0.05                             # initial energy threshold η₀
        eps_val = 1e-6                             # unused (for compatibility)
        fp_iters = 6                               # fixed‑point iterations
        fp_cap: Optional[int] = None               # cap on candidate set size
        top_k = 20                                 # number of results to return
        # Second argument is the seed word
        seed_word = argv[1]
        # Parse options
        i = 2
        while i < len(argv):
            arg = argv[i]
            if arg == "--anchors":
                # Anchor words are ignored in the unsupervised clustering but
                # accepted for backward compatibility; skip the value
                if i + 1 >= len(argv):
                    print("Error: --anchors requires a comma‑separated list.")
                    return
                # Consume the argument but do nothing
                i += 2
                continue
            if arg in ("--lambda", "-l"):
                if i + 1 >= len(argv):
                    print("Error: --lambda requires a numeric value.")
                    return
                try:
                    lambda_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --lambda.")
                    return
                i += 2
                continue
            # energy ratio threshold
            if arg == "--eta":
                if i + 1 >= len(argv):
                    print("Error: --eta requires a numeric value.")
                    return
                try:
                    eta_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --eta.")
                    return
                i += 2
                continue
            # variance threshold for subspace learning
            if arg == "--var":
                if i + 1 >= len(argv):
                    print("Error: --var requires a numeric value in (0,1].")
                    return
                try:
                    var_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --var.")
                    return
                if not (0.0 < var_val <= 1.0):
                    print("Error: --var must be in (0,1].")
                    return
                i += 2
                continue
            if arg == "--beta":
                if i + 1 >= len(argv):
                    print("Error: --beta requires a numeric value.")
                    return
                try:
                    beta_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --beta.")
                    return
                i += 2
                continue
            if arg in ("--tau", "-t"):
                if i + 1 >= len(argv):
                    print("Error: --tau requires a numeric value.")
                    return
                try:
                    tau_threshold = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --tau.")
                    return
                i += 2
                continue
            if arg == "--eps":
                if i + 1 >= len(argv):
                    print("Error: --eps requires a numeric value.")
                    return
                try:
                    eps_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --eps.")
                    return
                i += 2
                continue
            if arg == "--maxiter":
                if i + 1 >= len(argv):
                    print("Error: --maxiter requires an integer value.")
                    return
                try:
                    fp_iters = int(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --maxiter.")
                    return
                i += 2
                continue
            if arg in ("--top", "--max", "-k"):
                if i + 1 >= len(argv):
                    print("Error: --top/--max requires an integer value.")
                    return
                try:
                    top_k = int(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --top.")
                    return
                i += 2
                continue
            if arg == "--fp_cap":
                if i + 1 >= len(argv):
                    print("Error: --fp_cap requires an integer value.")
                    return
                try:
                    fp_cap = int(argv[i + 1])
                except ValueError:
                    print("Error: invalid value for --fp_cap.")
                    return
                i += 2
                continue
            # Additional tokens are appended to seed word
            seed_word += " " + arg
            i += 1
        if not load_glove_embeddings():
            return
        if seed_word not in word_to_index:
            print(f"Seed word '{seed_word}' not found in vocabulary.")
            return
        cluster = cluster_rank_greedy(
            seed_word,
            tau_threshold=tau_threshold,
            lambda_val=lambda_val,
            var_threshold=var_val,
            eta=eta_val,
            top_k=top_k,
            fp_iters=fp_iters,
            fp_cap=fp_cap,
        )
        if cluster is None or not cluster:
            print(f"No cluster found for '{seed_word}'.")
            return
        # Print a generic cluster heading instead of "Amphibian cluster".
        # The clustering is now anchor‑free and applies to any domain.
        print(f"Cluster for '{seed_word}':")
        for name, score in cluster:
            print(f"    {name} (score={score:.3f})")
        return
    # Original chain mode (relevance propagation)
    if arg0.lower().startswith("/chain"):
        if len(argv) < 2:
            print("Please specify a word after /chain.")
            return
        # Defaults for unsupervised fixed‑point relevance algorithm
        lambda_val = 0.6         # mixing parameter for propagation (teleport weight)
        # tau_val is None by default to enable adaptive threshold selection
        tau_val: Optional[float] = None  # similarity threshold for neighbourhood
        eta_val = 0.05          # initial energy threshold η₀
        var_val = 0.95          # variance retention ρ
        top_k = 10              # number of results to return
        max_iters = 20          # maximum number of random walk iterations
        eps_val = 1e-6          # convergence tolerance for random walk
        fp_iters = 6            # neighbourhood expansion iterations
        fp_cap: Optional[int] = None  # cap on candidate set size
        word = argv[1]
        # Parse optional parameters
        i = 2
        while i < len(argv):
            arg = argv[i]
            # Mixing parameter for relevance propagation
            if arg in ("--lambda", "-l"):
                if i + 1 >= len(argv):
                    print("Error: --lambda requires a numeric value.")
                    return
                try:
                    lambda_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid numeric value for --lambda.")
                    return
                i += 2
                continue
            # Similarity threshold for adjacency.  If the user specifies a value
            # via --tau, we override the adaptive default (None).  Otherwise
            # tau_val remains None and compute_relevance_fixedpoint() will
            # choose an appropriate threshold based on the distribution of
            # similarities to the seed.  Passing a non‑numeric value is an
            # error.  Note that an empty argument after --tau resets to
            # adaptive behaviour.
            if arg in ("--tau", "-t"):
                if i + 1 >= len(argv):
                    print("Error: --tau requires a numeric value.")
                    return
                try:
                    tau_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid numeric value for --tau.")
                    return
                i += 2
                continue
            # Energy threshold for candidate selection
            if arg == "--eta":
                if i + 1 >= len(argv):
                    print("Error: --eta requires a numeric value.")
                    return
                try:
                    eta_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid numeric value for --eta.")
                    return
                i += 2
                continue
            # Variance threshold for projector
            if arg == "--var":
                if i + 1 >= len(argv):
                    print("Error: --var requires a numeric value in (0,1].")
                    return
                try:
                    var_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid numeric value for --var.")
                    return
                if not (0.0 < var_val <= 1.0):
                    print("Error: --var must be in (0,1].")
                    return
                i += 2
                continue
            # Number of neighbours to return
            if arg in ("--max", "-m", "--top", "-k"):
                if i + 1 >= len(argv):
                    print("Error: --max/--top requires an integer value.")
                    return
                try:
                    top_k = int(argv[i + 1])
                except ValueError:
                    print("Error: invalid integer value for --max/--top.")
                    return
                i += 2
                continue
            # Maximum number of iterations
            if arg == "--maxiter":
                # random walk iterations
                if i + 1 >= len(argv):
                    print("Error: --maxiter requires an integer value.")
                    return
                try:
                    max_iters = int(argv[i + 1])
                except ValueError:
                    print("Error: invalid integer value for --maxiter.")
                    return
                i += 2
                continue
            # Fixed‑point expansion iterations
            if arg == "--fp_iter":
                if i + 1 >= len(argv):
                    print("Error: --fp_iter requires an integer value.")
                    return
                try:
                    fp_iters = int(argv[i + 1])
                except ValueError:
                    print("Error: invalid integer value for --fp_iter.")
                    return
                i += 2
                continue
            # Cap on fixed‑point neighbourhood size
            if arg == "--fp_cap":
                if i + 1 >= len(argv):
                    print("Error: --fp_cap requires an integer value.")
                    return
                try:
                    fp_cap = int(argv[i + 1])
                except ValueError:
                    print("Error: invalid integer value for --fp_cap.")
                    return
                i += 2
                continue
            # Convergence tolerance
            if arg == "--eps":
                if i + 1 >= len(argv):
                    print("Error: --eps requires a numeric value.")
                    return
                try:
                    eps_val = float(argv[i + 1])
                except ValueError:
                    print("Error: invalid numeric value for --eps.")
                    return
                i += 2
                continue
            # Additional tokens (e.g. multi‑word phrase)
            word += " " + arg
            i += 1
        # Load embeddings (default path; adjust as needed)
        if not load_glove_embeddings():
            return
        if word not in word_to_index:
            print(f"Word '{word}' not found in the embedding vocabulary.")
            return
        print(f"Semantic relevance cluster for \"{word}\" (lambda={lambda_val}, tau={tau_val}, top={top_k}):")
        results = compute_relevance_fixedpoint(
            word,
            tau=tau_val,
            lambda_val=lambda_val,
            max_iters=max_iters,
            eps=eps_val,
            top_k=top_k,
            eta=eta_val,
            filter_anchors=None,
            var_threshold=var_val,
            fp_iters=fp_iters,
            fp_cap=fp_cap,
        )
        if not results:
            # If no neighbours found above threshold, output seed only
            print(word)
            return
        # Print the seed separately and then neighbours with relevance scores.
        # The relevance scores returned by compute_relevance_fixedpoint()
        # already incorporate energy weighting and domain‑aware filtering.
        seed_printed = False
        for tok, score in results:
            if tok == word and not seed_printed:
                print(f"{tok}")
                seed_printed = True
                continue
            print(f"    {tok} (relevance={score:.2f})")
        return
    # Otherwise treat arguments as formula string
    formula_str = " ".join(argv)
    try:
        ast = parse_formula(formula_str)
    except Exception as e:
        print(f"Parsing error: {e}")
        return
    print(f"Evaluating formula: {formula_str}")
    _ = compute_ofi(ast)


if __name__ == "__main__":
    main()
