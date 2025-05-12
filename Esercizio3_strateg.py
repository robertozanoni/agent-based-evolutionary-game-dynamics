import mesa
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd

# === PARAMETRI CONFIGURABILI ===
n_nodes = 49  # Deve esserci l'1
k_degree = 4
gap = 4
n_steps = 100
strategy_0_percent = 50
strategy_1_percent = 100 - strategy_0_percent
prob_revision = 0.1
noise = 0.03
payoff_matrix = [
    [1, 0],
    [0, 2]
]

# === CLASSE AGENTE ===
class ImitationAgent(mesa.Agent):
    def __init__(self, model, strategy):
        super().__init__(model)
        self.strategy = strategy
        self.wealth = 1
        self.pos = None

    def update_payoff(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if neighbors:
            for other in neighbors:
                payoff_self = self.model.payoff_matrix[self.strategy][other.strategy]
                payoff_other = self.model.payoff_matrix[other.strategy][self.strategy]
                self.wealth = payoff_self
                other.wealth = payoff_other
        else:
            self.wealth = 0

    def update_strategy(self):
        if self.random.random() < self.model.prob_revision:
            neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
            if neighbors:
                other = self.random.choice(neighbors)
                if other.wealth > self.wealth:
                    if self.random.random() < self.model.noise:
                        self.strategy = 1 - other.strategy
                    else:
                        self.strategy = other.strategy

# === MODELLO ===
class GapRingModel(mesa.Model):
    def __init__(self):
        super().__init__()
        self.num_agents = n_nodes
        self.prob_revision = prob_revision
        self.noise = noise
        self.payoff_matrix = payoff_matrix
        self.grid = mesa.space.NetworkGrid(self.build_gap_ring_lattice())

        # Strategia iniziale
        initial_strategies = [0] * int(n_nodes * strategy_0_percent / 100) + \
                             [1] * (n_nodes - int(n_nodes * strategy_0_percent / 100))
        random.shuffle(initial_strategies)

        self.agent_list = [ImitationAgent(self, s) for s in initial_strategies]
        for agent, node in zip(self.agent_list, self.grid.G.nodes()):
            self.grid.place_agent(agent, node)
            agent.pos = node

        self.strategy_counts = []

    def build_gap_ring_lattice(self):
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for node in range(n_nodes):
            for i in range(1, k_degree // 2 + 1):
                right = (node + i * gap) % n_nodes
                left = (node - i * gap) % n_nodes
                G.add_edge(node, right)
                G.add_edge(node, left)
        return G

    def step(self):
        count_0 = sum(1 for agent in self.agent_list if agent.strategy == 0)
        count_1 = sum(1 for agent in self.agent_list if agent.strategy == 1)
        self.strategy_counts.append({'Step': len(self.strategy_counts), 'Strategia 0': count_0, 'Strategia 1': count_1})

        random.shuffle(self.agent_list)
        for agent in self.agent_list:
            agent.update_payoff()
        random.shuffle(self.agent_list)
        for agent in self.agent_list:
            agent.update_strategy()

# === ESECUZIONE MODELLO ===
model = GapRingModel()

# === GRAFICO INIZIALE (Layout Circolare con Strategie) ===
plt.figure(figsize=(10, 6))
circular_pos = nx.circular_layout(model.grid.G)
initial_colors = ['blue' if agent.strategy == 0 else 'red' for agent in model.agent_list]

nx.draw(model.grid.G,
        circular_pos,
        node_color=initial_colors,
        edge_color='gray',
        labels={a.pos: str(a.pos) for a in model.agent_list},
        with_labels=True,
        node_size=500,
        font_color='white',
        font_size=8)
plt.title(f"Configurazione Iniziale Strategica (n={n_nodes}, k={k_degree}, gap={gap})", fontsize=14)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Strategia 0', markerfacecolor='blue', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Strategia 1', markerfacecolor='red', markersize=10)
], loc='upper right')
plt.axis('off')
plt.tight_layout()
plt.show()

# === SIMULAZIONE ===
for step in range(n_steps):
    model.step()

# === TABELLA STRATEGIE ===
df = pd.DataFrame(model.strategy_counts)
df.set_index("Step", inplace=True)
print("\nTabella strategie per step:")
print(df)

# === GRAFICO EVOLUZIONE STRATEGIE (Stacked Area) ===
df_percentuale = df.div(df.sum(axis=1), axis=0)

plt.figure(figsize=(10, 6))
plt.stackplot(df_percentuale.index,
              df_percentuale['Strategia 0'],
              df_percentuale['Strategia 1'],
              labels=['0', '1'],
              colors=['orangered', 'limegreen'])
plt.title("Distribuzione Strategie (Stacked Area)", fontsize=14)
plt.xlabel("Passi di Simulazione")
plt.ylabel("Frequenza")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === GRAFICO FINALE (Layout Circolare con Strategie) ===
plt.figure(figsize=(10, 6))
final_colors = ['blue' if agent.strategy == 0 else 'red' for agent in model.agent_list]

nx.draw(model.grid.G,
        circular_pos,
        node_color=final_colors,
        edge_color='gray',
        labels={a.pos: str(a.pos) for a in model.agent_list},
        with_labels=True,
        node_size=500,
        font_color='white',
        font_size=8)
plt.title(f"Configurazione Finale Strategica (n={n_nodes}, k={k_degree}, gap={gap})", fontsize=14)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Strategia 0', markerfacecolor='blue', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Strategia 1', markerfacecolor='red', markersize=10)
], loc='upper right')
plt.axis('off')
plt.tight_layout()
plt.show()
