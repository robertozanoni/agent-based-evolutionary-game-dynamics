# ring_lattice_analysis.py

import networkx as nx                      # Per costruzione e analisi di reti
import matplotlib.pyplot as plt           # Per visualizzazione grafica

# === Parametri della rete ===
n_nodes = 50           # Numero di nodi nella rete, con k=4 deve essere maggiore di 6 etc.
k_degree = 4            # Grado pari (ogni nodo ha k vicini)
rewire_prob = 0         # Nessun rewiring → ring lattice puro

if n_nodes < ((3*k_degree)/2):
    print("Attenzione: il numero di nodi deve essere maggiore di (3*k)/2.")
# === Creazione del grafo ring lattice (Watts-Strogatz con p=0) ===
G = nx.watts_strogatz_graph(n=n_nodes, k=k_degree, p=rewire_prob)

# === Visualizzazione della rete ===
plt.figure(figsize=(8, 8))                          # Imposta la dimensione della figura
pos = nx.circular_layout(G)                         # Disposizione circolare dei nodi
nx.draw(G, pos,
        node_size=300,
        node_color='skyblue',
        edge_color='gray',
        with_labels=True,
        font_size=8,
        font_weight='bold')
plt.title(f"Rete Ring Lattice (n={n_nodes}, k={k_degree}, p={rewire_prob})", fontsize=14)
plt.tight_layout()
plt.show()

# === Calcolo del clustering coefficient locale per ogni nodo ===
clustering_dict = nx.clustering(G)                  # Dizionario {nodo: clustering}

# === Visualizzazione clustering coefficient per nodo ===
plt.figure(figsize=(10, 5))
plt.plot(list(clustering_dict.keys()), list(clustering_dict.values()), marker='o', linestyle='-')
plt.title("Clustering Coefficient per Nodo nella Rete", fontsize=14)
plt.xlabel("ID Nodo")
plt.ylabel("Clustering Coefficient")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Confronto tra valore teorico e valore medio osservato ===
# Formula teorica: C = 3(k - 2) / 4(k - 1)
C_teorico = 3 * (k_degree - 2) / (4 * (k_degree - 1))
C_medio = sum(clustering_dict.values()) / len(clustering_dict)

# Stampa dei risultati
print("=== Confronto Clustering Coefficient ===")
print(f"Valore teorico (formula):     C = {C_teorico:.4f}")
print(f"Valore medio osservato:       C = {C_medio:.4f}")
print(f"Errore assoluto:              |ΔC| = {abs(C_teorico - C_medio):.4e}")
