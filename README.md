# Spectral Null Cone & Semantic Chain Laboratory

A compact research toolkit for probing and manipulating **Word2Vec** spaces.  It lets you

* strip away high‑variance directions with four alternative *null‑cone operators*; and
* generate semantic *fan‑out chains* that balance topical proximity against redundancy.

The code is self‑contained (≈450 LOC), dependency‑light, and designed for interactive exploration at the command line.

---

## 1  Features

| Module                                                                      | Purpose                                                                               | Key flags                |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------ |
| **smooth**                                                                  | Exponential low‑pass filter that removes a user‑specified fraction of total variance. | `--rho`, `--p`           |
| **iso**                                                                     | Shrinkage `λ/(λ+β)` that keeps the cone isotropic while damping dominant axes.        | `--rho`                  |
| **whiten**                                                                  | Mahalanobis whitening plus radial clip, ideal for outlier suppression.                | `--alpha`                |
| **combo**                                                                   | `smooth` followed by removal of the axes with largest excess kurtosis.                | `--rho`, `--rho2`, `--p` |
| **/chain**                                                                  | In‑console command that creates semantic chains according to the formula              |                          |
| $\;\arg\max_v\;\big(cos(v,x_t)\; -\; \lambda\max_{u\in S_t}cos(v,u)\big)\;$ | parameters typed after `/chain`: `λ` (redundancy weight), `τ` (min. cosine)           |                          |

---

## 2  Installation

```bash
python3 ‑m venv venv           # optional but recommended
source venv/bin/activate
pip install -r requirements.txt  # numpy scipy gensim
```

Download a Word2Vec binary, e.g. the Google‑News 300‑dimensional model, and note its path.

---

## 3  Quick start

```bash
python spectral_null_cone.py \
       --bin /path/to/GoogleNews-vectors-negative300.bin.gz \
       --mode smooth --rho 0.95 --p 2 --eps 0.05
```

You will see something like:

```
[INFO] Loading vectors from …
[INFO] 3,000,000 tokens · dim=300
[INFO] smooth ρ=0.950 → θ*=0.0062, p=2.0, kept=5.00%

Enter word / sentence   (quit to exit,  /null to list ε-small words)
Extra commands:  /chain [λ] [τ] sentence …
>>>
```

### Generating a chain

```text
>>> /chain do simulation

do  want  simulation  simulations  computer_simulations  numerical_simulations
```

*The chain stays close to the current word yet avoids redundancy.*

A more expansive demonstration:

```text
>>> /chain make foods

make  making  made  foods  minimally_processed_foods  vegetables_legumes  unrefined_carbohydrates  beans_legumes  sweet_potatoes_carrots  carrots_sweet_potatoes  craisins  Serve_garnished  Serve_sprinkled  Sprinkle_evenly  sprinkle_evenly  Evenly_sprinkle  Gently_stir  Gently_fold  yolk_mixture  egg_yolk_mixture  gelatin_mixture  creamed_mixture  buttermilk_mixture  cornstarch_mixture  yeast_mixture  cup_confectioners_sugar  tablespoons_flour  cups_flour  cup_powdered_sugar  teaspoon_cinnamon  tablespoon_lemon_juice  tsp_salt  teaspoons_salt  ¼_teaspoon_salt  ½_teaspoon_salt  ¼_teaspoon  ½_teaspoon  ½_tsp  ¼_tsp  chili_powder_cumin  garlic_cumin  cumin_oregano  oregano_salt  cumin_salt  thyme_salt  chili_powder_salt  cumin_paprika  Add_onion_garlic  Add_onions_garlic  Add_onion_celery  Add_onions_celery  Add_onions_carrots  Chop_onion  onion_bell_pepper
```

*Starting from **make**, the chain fans into food‑related nouns, ingredient pairs, and finally instruction‑style bigrams—showcasing how the redundancy penalty keeps introducing fresh yet semantically linked tokens even across fifty steps.*

---

## 4  Command‑line flags (summary)

| Flag      | Default  | Meaning                                                |
| --------- | -------- | ------------------------------------------------------ |
| `--mode`  | `smooth` | Which null‑cone operator to apply.                     |
| `--rho`   | `0.95`   | Fraction of variance *removed* (smooth / iso / combo). |
| `--p`     | `2.0`    | Sharpness of the smooth filter.                        |
| `--alpha` | `2.0`    | Mahalanobis radius for whitening clip.                 |
| `--rho2`  | `0.80`   | Fraction of *kurtosis power* removed in combo mode.    |
| `--eps`   | `0.05`   | Threshold for listing near‑null tokens with `/null`.   |

Inside the REPL:

| Command  | Arguments            | Effect                                                                 |
| -------- | -------------------- | ---------------------------------------------------------------------- |
| `/null`  | –                    | Show tokens whose hollowed norm `h_i ≤ ε`.                             |
| `/chain` | `[λ] [τ] sentence …` | Generate a proximity‑redundancy chain. Defaults: `λ = 0.5`, `τ = 0.7`. |

---

## 5  Implementation notes

* All heavy‑weight arithmetic uses NumPy BLAS calls; even on a laptop the `/chain` search over three million vectors is near‑instantaneous.
* `kv.get_normed_vectors()` is cached once at startup, so cosine computations reduce to a single dot‑product.
* Each operator solves its scalar hyper‑parameter (θ or β) via a concise binary search that guarantees the requested variance fraction.
* The project avoids external ML frameworks to keep the dependency tree minimal and transparent.

---

## 6  Citation

If you use this toolkit in academic work, please cite:

> F. Alpay, *Spectral Null‑Cone Operators and Redundancy‑Balanced Semantic Chains*, 2025.

---

## 7  License

MIT License.  See `LICENSE` for details.

---

## 8  Contact

Faruk Alpay  ·  [alpay@lightcap.ai](mailto:alpay@lightcap.ai)
