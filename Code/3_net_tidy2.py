

import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch


BASE_DIR = "/fs/dss/work/role6027/defence_bertopic_network_out/networks/net_sentence"

PERIODS = ["pre_2024-12-01", "post_2024-12-01"]

# Output directory for improved figures/tables
OUT_DIR = os.path.join(BASE_DIR, "better_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Reduce clutter
MIN_EDGE_WEIGHT = 5
TOP_EDGES_GLOBAL = 400          # keep only top global edges
TOP_EDGES_PER_COMMISSIONER = 20 # keep top edges per commissioner
LABEL_TOP_TOPICS = 30           # label only top topics by strength
LABEL_ALL_COMMISSIONERS = True  # if False: only label top N commissioners
LABEL_TOP_COMMISSIONERS = 40

FIGSIZE = (24, 14)
DPI = 260



def draw_curved_edges(ax, pos, edgelist, widths, alphas, color="black", base_rad=0.18):

    for i, ((u, v), lw, a) in enumerate(zip(edgelist, widths, alphas)):
        rad = base_rad + (i % 11) * 0.02  # vary curvature to separate overlaps
        (x1, y1) = pos[u]
        (x2, y2) = pos[v]
        patch = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-",           # line only
            mutation_scale=1,
            linewidth=lw,
            color=color,
            alpha=a,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1
        )
        ax.add_patch(patch)


def radial_layout(G: nx.Graph, center: str | None = None) -> dict:

    if G.number_of_nodes() == 0:
        return {}

    # choose center as highest weighted degree if not provided
    if center is None:
        strength = dict(G.degree(weight="weight"))
        center = max(strength, key=strength.get)

    pos = {center: (0.0, 0.0)}

    # neighbors sorted by edge weight to center
    nbrs = list(G.neighbors(center))
    nbrs.sort(key=lambda n: float(G.get_edge_data(center, n).get("weight", 1.0)), reverse=True)

    # ring radii
    R1 = 12.0
    R2 = 22.0
    R3 = 30.0

    # split into rings
    n1 = int(math.ceil(len(nbrs) * 0.45))
    n2 = int(math.ceil(len(nbrs) * 0.35))
    ring1 = nbrs[:n1]
    ring2 = nbrs[n1:n1 + n2]
    ring3 = nbrs[n1 + n2:]

    def place_ring(nodes, R, phase=0.0):
        k = max(1, len(nodes))
        for i, n in enumerate(nodes):
            ang = phase + 2 * math.pi * (i / k)
            pos[n] = (R * math.cos(ang), R * math.sin(ang))

    place_ring(ring1, R1, phase=0.0)
    place_ring(ring2, R2, phase=math.pi / 10)
    place_ring(ring3, R3, phase=math.pi / 5)

    # any remaining nodes not connected to center
    remaining = [n for n in G.nodes() if n not in pos]
    if remaining:
        place_ring(remaining, R3 + 10.0, phase=math.pi / 7)

    return pos


def load_bipartite_tables(base_dir: str, period: str):
    nodes_path = os.path.join(base_dir, f"{period}_bipartite_nodes.csv")
    edges_path = os.path.join(base_dir, f"{period}_bipartite_edges.csv")
    metrics_path = os.path.join(base_dir, f"{period}_commissioner_metrics.csv")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    metrics = pd.read_csv(metrics_path)
    return nodes, edges, metrics

def build_graph_from_tables(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(
            r["node"],
            node_type=r.get("node_type", ""),
            label=r.get("label", r["node"])
        )
    for _, r in edges.iterrows():
        G.add_edge(
            r["u"], r["v"],
            weight=float(r.get("weight_display", r.get("weight_count", 1.0))),
            weight_count=float(r.get("weight_count", 1.0)),
            weight_prob=float(r.get("weight_prob", r.get("weight_count", 1.0))),
        )
    return G

def filter_edges_for_plot(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    # keep edges above threshold
    edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("weight_count", 0) >= MIN_EDGE_WEIGHT]
    # keep top global edges
    edges.sort(key=lambda x: x[2].get("weight_count", 0), reverse=True)
    edges = edges[:TOP_EDGES_GLOBAL]

    for u, v, d in edges:
        H.add_edge(u, v, **d)

    # ensure top edges per commissioner (for bipartite)
    node_type = nx.get_node_attributes(G, "node_type")
    for n in G.nodes():
        if node_type.get(n) != "commissioner":
            continue
        nbrs = []
        for nbr in G.neighbors(n):
            d = G.get_edge_data(n, nbr)
            w = float(d.get("weight_count", d.get("weight", 1.0)))
            nbrs.append((nbr, w, d))
        nbrs.sort(key=lambda x: x[1], reverse=True)
        for nbr, w, d in nbrs[:TOP_EDGES_PER_COMMISSIONER]:
            if w >= 1:
                H.add_edge(n, nbr, **d)

    # remove isolates
    isolates = [n for n in H.nodes() if H.degree(n) == 0]
    H.remove_nodes_from(isolates)
    return H


def bipartite_rank_layout(G: nx.Graph) -> dict:

    node_type = nx.get_node_attributes(G, "node_type")
    com = [n for n in G.nodes() if node_type.get(n) == "commissioner"]
    top = [n for n in G.nodes() if node_type.get(n) == "topic"]

    strength = dict(G.degree(weight="weight_count"))

    com_sorted = sorted(com, key=lambda n: strength.get(n, 0), reverse=True)
    top_sorted = sorted(top, key=lambda n: strength.get(n, 0), reverse=True)

    def y_positions(nodes):
        s = np.array([max(1.0, float(strength.get(n, 1.0))) for n in nodes], dtype=float)
        steps = np.sqrt(s)
        y = np.cumsum(steps)
        y = y - y.mean()
        return y

    y_com = y_positions(com_sorted)
    y_top = y_positions(top_sorted)

    pos = {}
    for n, y in zip(com_sorted, y_com):
        pos[n] = (0.0, float(y))
    for n, y in zip(top_sorted, y_top):
        pos[n] = (1.0, float(y))

    return pos


def label_style():
    return dict(
        fontsize=16,
        fontcolor="black",
        path_effects=[pe.withStroke(linewidth=5, foreground="white")]
    )


def draw_bipartite_with_labels(G: nx.Graph, title: str, out_png: str):
    pos = bipartite_rank_layout(G)

    node_type = nx.get_node_attributes(G, "node_type")
    labels = nx.get_node_attributes(G, "label")
    strength = dict(G.degree(weight="weight_count"))

    com = [n for n in G.nodes() if node_type.get(n) == "commissioner"]
    top = [n for n in G.nodes() if node_type.get(n) == "topic"]

    def nsize(n):
        return 160 + 50.0 * math.sqrt(max(0.0, float(strength.get(n, 0.0))))

    com_sizes = [nsize(n) for n in com]
    top_sizes = [nsize(n) for n in top]

    w = np.array([float(d.get("weight_count", 1.0)) for _, _, d in G.edges(data=True)], dtype=float)
    wmax = max(1.0, float(w.max())) if w.size else 1.0
    widths = [1.0 + 6.0 * math.sqrt(x / wmax) for x in w]
    alphas = [0.18 + 0.70 * (x / wmax) for x in w]

    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.title(title)
    ax = plt.gca()

    for (u, v, d), width, a in zip(G.edges(data=True), widths, alphas):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=a, edge_color="black")

    nx.draw_networkx_nodes(
        G, pos, nodelist=com, node_shape="o",
        node_color="0.20", edgecolors="black",
        linewidths=0.9, node_size=com_sizes, alpha=0.9
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=top, node_shape="s",
        node_color="0.78", edgecolors="black",
        linewidths=0.9, node_size=top_sizes, alpha=0.9
    )

    com_sorted = sorted(com, key=lambda n: strength.get(n, 0), reverse=True)
    top_sorted = sorted(top, key=lambda n: strength.get(n, 0), reverse=True)

    com_to_label = com_sorted if LABEL_ALL_COMMISSIONERS else com_sorted[:LABEL_TOP_COMMISSIONERS]
    top_to_label = top_sorted[:LABEL_TOP_TOPICS]

    plot_labels = {}
    for n in com_to_label:
        plot_labels[n] = labels.get(n, n).strip()
    for n in top_to_label:
        lab = labels.get(n, n)
        if isinstance(lab, str) and lab.startswith("TOP_"):
            lab = lab.replace("TOP_", "T")
        plot_labels[n] = lab

    st = label_style()
    for n, lab in plot_labels.items():
        x, y = pos[n]
        node_size = nsize(n)
        fontsize = min(22, 8 + 0.12 * math.sqrt(node_size))

        txt = ax.text(
            x, y, str(lab),
            fontsize=fontsize,
            color=st["fontcolor"],
            ha="center",
            va="center",
            family="sans-serif",
            zorder=10
        )
        txt.set_path_effects(st["path_effects"])

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


def draw_one_mode_with_labels(G: nx.Graph, title: str, out_png: str, label_top_n: int = 35):
    pos = radial_layout(G)

    strength = dict(G.degree(weight="weight"))

    def nsize(n):
        return 160 + 50.0 * math.sqrt(max(0.0, float(strength.get(n, 0.0))))

    sizes = {n: nsize(n) for n in G.nodes()}

    edgelist = list(G.edges())
    w = np.array([float(G.edges[e].get("weight", 1.0)) for e in edgelist], dtype=float)
    wmax = max(1.0, float(w.max())) if w.size else 1.0
    widths = [0.8 + 5.5 * math.sqrt(x / wmax) for x in w]
    alphas = [0.22 + 0.65 * ((x / wmax) ** 0.9) for x in w]

    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.title(title)
    ax = plt.gca()

    draw_curved_edges(ax, pos, edgelist, widths, alphas, color="black", base_rad=0.20)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=[sizes[n] for n in G.nodes()],
        node_color="0.35",
        edgecolors="black",
        linewidths=0.9,
        alpha=0.9
    )

    nodes_sorted = sorted(G.nodes(), key=lambda n: strength.get(n, 0), reverse=True)
    lab_nodes = nodes_sorted[:label_top_n]

    st = label_style()
    for n in lab_nodes:
        x, y = pos[n]
        node_size = sizes.get(n, 100)
        fontsize = min(22, 8 + 0.12 * math.sqrt(node_size))

        txt = ax.text(
            x, y, str(n),
            fontsize=fontsize,
            color=st["fontcolor"],
            ha="center",
            va="center",
            family="sans-serif",
            zorder=10
        )
        txt.set_path_effects(st["path_effects"])

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


def export_label_lookup(nodes: pd.DataFrame, out_csv: str):
    nodes[["node", "node_type", "label"]].to_csv(out_csv, index=False)

def export_ranked_commissioners(metrics: pd.DataFrame, out_csv: str):
    m = metrics.copy()
    for col in ["weighted_degree_count", "betweenness_count", "pagerank_count",
                "generalist_topic_count", "generalist_entropy"]:
        if col in m.columns:
            m[f"rank_{col}"] = m[col].rank(method="min", ascending=False)
    if "specialization_hhi" in m.columns:
        m["rank_specialization_hhi"] = m["specialization_hhi"].rank(method="min", ascending=False)

    m.to_csv(out_csv, index=False)


def bipartite_to_projections(G: nx.Graph):
    node_type = nx.get_node_attributes(G, "node_type")
    com_nodes = [n for n in G.nodes() if node_type.get(n) == "commissioner"]
    top_nodes = [n for n in G.nodes() if node_type.get(n) == "topic"]

    com_labels = [G.nodes[n].get("label", n) for n in com_nodes]
    top_labels = [G.nodes[n].get("label", n) for n in top_nodes]

    X = pd.DataFrame(0.0, index=com_labels, columns=top_labels)
    for com in com_nodes:
        com_lab = G.nodes[com].get("label", com)
        for top in G.neighbors(com):
            top_lab = G.nodes[top].get("label", top)
            d = G.get_edge_data(com, top)
            X.loc[com_lab, top_lab] = float(d.get("weight_count", d.get("weight", 0.0)))

    M_com = X.values @ X.values.T
    M_top = X.values.T @ X.values

    G_com = nx.Graph()
    for i, a in enumerate(X.index):
        G_com.add_node(a)
    for i in range(len(X.index)):
        for j in range(i + 1, len(X.index)):
            w_ij = float(M_com[i, j])
            if w_ij > 0:
                G_com.add_edge(X.index[i], X.index[j], weight=w_ij)

    G_top = nx.Graph()
    for i, a in enumerate(X.columns):
        G_top.add_node(a)
    for i in range(len(X.columns)):
        for j in range(i + 1, len(X.columns)):
            w_ij = float(M_top[i, j])
            if w_ij > 0:
                G_top.add_edge(X.columns[i], X.columns[j], weight=w_ij)

    return X, G_com, G_top


def main():
    for period in PERIODS:
        nodes, edges, metrics = load_bipartite_tables(BASE_DIR, period)
        G0 = build_graph_from_tables(nodes, edges)
        G = filter_edges_for_plot(G0)

        export_label_lookup(nodes, os.path.join(OUT_DIR, f"{period}_label_lookup.csv"))
        export_ranked_commissioners(metrics, os.path.join(OUT_DIR, f"{period}_ranked_commissioners.csv"))

        draw_bipartite_with_labels(
            G,
            title=f"Bipartite (clean + labels) | {period}",
            out_png=os.path.join(OUT_DIR, f"{period}_bipartite_clean_labels.pdf")
        )

        X, G_com, G_top = bipartite_to_projections(G)

        def filter_weak_edges(H, q=0.80):
            if H.number_of_edges() == 0:
                return H
            ws = np.array([d["weight"] for _, _, d in H.edges(data=True)], dtype=float)
            thr = float(np.quantile(ws, q))
            K = H.copy()
            drop = [(u, v) for u, v, d in K.edges(data=True) if d["weight"] < max(thr, 1.0)]
            K.remove_edges_from(drop)
            iso = [n for n in K.nodes() if K.degree(n) == 0]
            K.remove_nodes_from(iso)
            return K

        G_com_viz = filter_weak_edges(G_com, q=0.80)
        G_top_viz = filter_weak_edges(G_top, q=0.80)

        draw_one_mode_with_labels(
            G_com_viz,
            title=f"Commissioner–Commissioner (radial + curved) | {period}",
            out_png=os.path.join(OUT_DIR, f"{period}_com_com_clean_labels.pdf"),
            label_top_n=35
        )
        draw_one_mode_with_labels(
            G_top_viz,
            title=f"Topic–Topic (radial + curved) | {period}",
            out_png=os.path.join(OUT_DIR, f"{period}_top_top_clean_labels.pdf"),
            label_top_n=35
        )

        X.to_csv(os.path.join(OUT_DIR, f"{period}_commissioner_topic_matrix_for_plots.csv"))

    print("Done. New plots + tables in:", OUT_DIR)

if __name__ == "__main__":
    main()