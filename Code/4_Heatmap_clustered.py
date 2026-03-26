import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram


PRE_CSV = r"C:\Users\role6027\Desktop\pre_2024-12-01_commissioner_cosine.csv"
POST_CSV = r"C:\Users\role6027\Desktop\post_2024-12-01_commissioner_cosine.csv"

OUT_DIR = os.path.dirname(PRE_CSV)

# Vector formats
VECTOR_FORMATS = ["pdf", "svg"]  # add "eps" if needed

# Hierarchical clustering linkage method
LINKAGE_METHOD = "average"  # good default for cosine distance

# Bootstrap CI settings
BOOTSTRAP_N = 5000
CI_LEVEL = 0.95
RANDOM_SEED = 42


def read_cosine_csv(path: str) -> pd.DataFrame:
    S = pd.read_csv(path, index_col=0)
    S = S.apply(pd.to_numeric, errors="coerce")

    # Ensure string labels
    S.index = S.index.astype(str)
    S.columns = S.columns.astype(str)

    # reorder columns to match index.
    if set(S.columns) == set(S.index):
        S = S.loc[S.index, S.index]

    # Safety: diagonal should be 1
    np.fill_diagonal(S.values, 1.0)

    # Symmetrize lightly (in case of tiny floating mismatches)
    # (keeps it stable for clustering)
    S = (S + S.T) / 2.0
    np.fill_diagonal(S.values, 1.0)

    return S


def cosine_to_distance(S: pd.DataFrame) -> pd.DataFrame:
    D = 1.0 - S
    D = D.clip(lower=0.0, upper=2.0)
    np.fill_diagonal(D.values, 0.0)
    return D


def hierarchical_order(S: pd.DataFrame, method: str = "average"):
    D = cosine_to_distance(S)
    condensed = squareform(D.values, checks=False)
    Z = linkage(condensed, method=method)
    dendro = dendrogram(Z, no_plot=True, labels=list(S.index))
    order = dendro["ivl"]
    return Z, order


def plot_clustered_heatmap(S: pd.DataFrame, title: str, out_base: str, method: str = "average"):
    if S.shape[0] < 2:
        print(f"Skip heatmap: matrix too small ({S.shape[0]}x{S.shape[1]}) for {title}")
        return

    Z, order = hierarchical_order(S, method=method)
    Sr = S.loc[order, order]

    # Heatmap
    fig = plt.figure(figsize=(12, 10), dpi=240)
    ax = fig.add_subplot(111)
    im = ax.imshow(Sr.values, vmin=0.0, vmax=1.0, cmap="Greys", interpolation="nearest")
    ax.set_title(title)

    n = Sr.shape[0]
    if n <= 60:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(Sr.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(Sr.index, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")

    plt.tight_layout()
    for fmt in VECTOR_FORMATS:
        fig.savefig(f"{out_base}_clustered_heatmap.{fmt}", format=fmt)
    plt.close(fig)

    # Dendrogram
    fig = plt.figure(figsize=(12, 4), dpi=240)
    ax = fig.add_subplot(111)
    dendrogram(Z, labels=order, leaf_rotation=90, leaf_font_size=6, ax=ax)
    ax.set_title(title + " | dendrogram")
    ax.set_ylabel("Distance (1 - cosine)")
    plt.tight_layout()
    for fmt in VECTOR_FORMATS:
        fig.savefig(f"{out_base}_dendrogram.{fmt}", format=fmt)
    plt.close(fig)


def person_level_means(S: pd.DataFrame) -> np.ndarray:
    if S.shape[0] < 2:
        return np.array([], dtype=float)

    A = S.values.astype(float)
    np.fill_diagonal(A, np.nan)          # exclude self-similarity
    per_person = np.nanmean(A, axis=1)   # mean similarity to others
    per_person = per_person[~np.isnan(per_person)]
    return per_person


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, ci_level: float, seed: int):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        return np.nan, (np.nan, np.nan)

    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(x, size=x.size, replace=True)
        boot_means[i] = float(np.mean(samp))

    alpha = 1.0 - ci_level
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return float(np.mean(x)), (lo, hi)


def plot_homogeneity_personlevel(pre_person_means: np.ndarray,
                                post_person_means: np.ndarray,
                                title: str,
                                out_base: str):
    pre_mean, (pre_lo, pre_hi) = bootstrap_mean_ci(
        pre_person_means, n_boot=BOOTSTRAP_N, ci_level=CI_LEVEL, seed=RANDOM_SEED
    )
    post_mean, (post_lo, post_hi) = bootstrap_mean_ci(
        post_person_means, n_boot=BOOTSTRAP_N, ci_level=CI_LEVEL, seed=RANDOM_SEED + 1
    )

    fig = plt.figure(figsize=(6, 4), dpi=240)
    ax = fig.add_subplot(111)

    x = np.array([0, 1], dtype=float)
    y = np.array([pre_mean, post_mean], dtype=float)
    yerr = np.array([[pre_mean - pre_lo, post_mean - post_lo],
                     [pre_hi - pre_mean, post_hi - post_mean]], dtype=float)

    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(["pre", "post"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Person-level mean cosine similarity")
    ax.set_title(title)

    # Annotate with n
    ax.text(0, pre_mean, f"  {pre_mean:.3f} [{pre_lo:.3f}, {pre_hi:.3f}]  (n={len(pre_person_means)})", va="center")
    ax.text(1, post_mean, f"  {post_mean:.3f} [{post_lo:.3f}, {post_hi:.3f}]  (n={len(post_person_means)})", va="center")

    plt.tight_layout()
    for fmt in VECTOR_FORMATS:
        fig.savefig(f"{out_base}_homogeneity_mean_ci_personlevel.{fmt}", format=fmt)
    plt.close(fig)

    # Export numeric summary
    summ = pd.DataFrame([
        {"period": "pre", "mean": pre_mean, "ci_low": pre_lo, "ci_high": pre_hi,
         "n_persons": int(len(pre_person_means))},
        {"period": "post", "mean": post_mean, "ci_low": post_lo, "ci_high": post_hi,
         "n_persons": int(len(post_person_means))},
    ])
    summ.to_csv(f"{out_base}_homogeneity_mean_ci_personlevel.csv", index=False)



def main():
    S_pre = read_cosine_csv(PRE_CSV)
    S_post = read_cosine_csv(POST_CSV)

    # 1) Clustered heatmaps + dendrograms (each period separately)
    pre_base = os.path.join(OUT_DIR, "pre_2024-12-01")
    post_base = os.path.join(OUT_DIR, "post_2024-12-01")

    plot_clustered_heatmap(
        S_pre,
        title="Clustered cosine similarity | pre_2024-12-01",
        out_base=pre_base,
        method=LINKAGE_METHOD,
    )
    plot_clustered_heatmap(
        S_post,
        title="Clustered cosine similarity | post_2024-12-01",
        out_base=post_base,
        method=LINKAGE_METHOD,
    )

    # 2) Person-level homogeneity with bootstrap CIs (each period separately)
    pre_person_means = person_level_means(S_pre)
    post_person_means = person_level_means(S_post)

    out_base = os.path.join(OUT_DIR, "pre_vs_post")
    plot_homogeneity_personlevel(
        pre_person_means,
        post_person_means,
        title=f"Homogeneity (person-level mean cosine ± {int(CI_LEVEL*100)}% CI)",
        out_base=out_base
    )

    print("Done.")
    print(f"Outputs written to: {OUT_DIR}")
    print("Heatmaps/dendrograms: pre_2024-12-01_* and post_2024-12-01_*")
    print("Homogeneity plot: pre_vs_post_homogeneity_mean_ci_personlevel.*")


if __name__ == "__main__":
    main()