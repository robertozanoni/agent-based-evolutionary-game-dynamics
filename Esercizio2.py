# esercizio2_ring_lattice.py

import networkx as nx
import matplotlib.pyplot as plt

# === Parametri configurabili ===
n_nodes = 20       # Numero totale di nodi
k_degree = 10       # Grado pari: ogni nodo collegato a k/2 a sinistra e k/2 a destra

# Verifica che il grado sia pari e valido
if k_degree % 2 != 0 or k_degree >= n_nodes:
    raise ValueError("Il grado k deve essere un numero pari e minore di n.")

# === Costruzione manuale del ring lattice ===
G = nx.Graph()

# Aggiungi tutti i nodi
G.add_nodes_from(range(n_nodes))

# Collega ciascun nodo a k/2 vicini a sinistra e k/2 a destra
for i in range(n_nodes):
    for j in range(1, k_degree // 2 + 1):
        neighbor = (i + j) % n_nodes
        G.add_edge(i, neighbor)

# === Visualizzazione della rete ===
plt.figure(figsize=(8, 8))
pos = nx.circular_layout(G)  # Layout circolare
nx.draw(G, pos,
        node_color='skyblue',
        edge_color='gray',
        with_labels=True,
        font_size=10,
        node_size=500)
plt.title(f"Ring Lattice Manuale (n={n_nodes}, k={k_degree}, p=0)")
plt.tight_layout()
plt.show()

# === Calcolo e visualizzazione del clustering coefficient per nodo ===
clustering_dict = nx.clustering(G)

# Grafico: clustering coefficient per nodo
plt.figure(figsize=(10, 5))
plt.plot(list(clustering_dict.keys()), list(clustering_dict.values()),
         marker='o', linestyle='-')
plt.title("Clustering Coefficient per Nodo")
plt.xlabel("ID Nodo")
plt.ylabel("Clustering Coefficient")
plt.grid(True)
plt.tight_layout()
plt.show()
