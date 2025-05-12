# gap_ring_lattice_ex3.py

import networkx as nx
import matplotlib.pyplot as plt

# === Parametri della rete ===
n_nodes = 20       # Numero di nodi
k_degree = 2      # Deve essere pari
gap = 6            # Numero di nodi da "saltare" a sinistra e destra

# === Costruzione manuale della gap-k ring lattice ===
G = nx.Graph()
G.add_nodes_from(range(n_nodes))

# Collega ogni nodo ai k/2 vicini dopo il gap su entrambi i lati
for node in range(n_nodes):
    for i in range(gap + 1, gap + 1 + (k_degree // 2)):
        neighbor_right = (node + i) % n_nodes
        neighbor_left = (node - i) % n_nodes
        G.add_edge(node, neighbor_right)
        G.add_edge(node, neighbor_left)

# === Layout "a parabola" per rendere visibili le connessioni ===
# Spazia i nodi in orizzontale e assegna una curvatura verticale
pos = {i: (i, (i - n_nodes // 2) ** 2 * 0.01) for i in range(n_nodes)}

# === Disegna la rete ===
plt.figure(figsize=(12, 6))
nx.draw(
    G, pos,
    with_labels=True,
    node_color='gold',
    edge_color='gray',
    node_size=600,
    font_weight='bold'
)
plt.title(f"Gap-{gap} Ring Lattice di grado {k_degree} (n={n_nodes})")
plt.axis('off')
plt.tight_layout()
plt.show()

# === Calcolo e visualizzazione del clustering coefficient ===
clustering_dict = nx.clustering(G)
plt.figure(figsize=(10, 5))
plt.plot(list(clustering_dict.keys()), list(clustering_dict.values()), marker='o')
plt.title("Clustering Coefficient per Nodo nella Gap-Ring Lattice")
plt.xlabel("ID Nodo")
plt.ylabel("Clustering Coefficient")
plt.grid(True)
plt.tight_layout()
plt.show()
