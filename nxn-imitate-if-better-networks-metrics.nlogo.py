# ===========================
# Simulazione Agent-Based con mesa 3.x (modello aggiornato con shuffle_do e payoff su tutti i vicini)
# ===========================
import mesa
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random

# === PARAMETRI CONFIGURABILI ===
n_agents = 100
n_steps = 50
strategy_0_percent = 70
strategy_1_percent = 100 - strategy_0_percent
initial_probs = [strategy_0_percent / 100, strategy_1_percent / 100]
prob_revision = 0.1
noise = 0.03
network_type = "watts_strogatz"
k_avg_degree = 4
rewire_prob = 0.1

# === MATRICE DEI PAYOFF ===
payoff_matrix = [
    [1, 0],  # payoff[0][0], payoff[0][1]
    [0, 2]   # payoff[1][0], payoff[1][1]
]

# === CLASSE RETE PERSONALIZZATA ===
class Network(mesa.space.NetworkGrid):
    def __init__(self, num_nodes, model_type="watts_strogatz"):
        graph = self.build_graph(num_nodes, model_type)
        super().__init__(graph)

    def build_graph(self, n, model_type):
        if model_type == "erdos_renyi":
            return nx.erdos_renyi_graph(n=n, p=0.1)
        elif model_type == "watts_strogatz":
            return nx.watts_strogatz_graph(n=n, k=k_avg_degree, p=rewire_prob)
        elif model_type == "preferential_attachment":
            return nx.barabasi_albert_graph(n=n, m=2)
        elif model_type == "ring":
            return nx.cycle_graph(n=n)
        else:
            raise ValueError(f"Tipo di rete non supportato: {model_type}")

# === CLASSE AGENTE ===
class ImitationAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.strategy = self.random.choices([0, 1], weights=initial_probs, k=1)[0]
        self.wealth = 1
        self.payoffs = payoff_matrix

    def update_payoff(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if neighbors:
            for other in neighbors:
                payoff_self = self.payoffs[self.strategy][other.strategy]
                payoff_other = self.payoffs[other.strategy][self.strategy]
                self.wealth = payoff_self
                other.wealth = payoff_other

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
class ImitationModel(mesa.Model):
    def __init__(self):
        super().__init__()
        self.num_agents = n_agents
        self.prob_revision = prob_revision
        self.noise = noise
        self.payoff_matrix = payoff_matrix

        self.grid = Network(n_agents, model_type=network_type)

        # Create agents
        agents = ImitationAgent.create_agents(model=self, n=self.num_agents)

        # Creazione e posizionamento agenti
        for agent, node in zip(agents, self.grid.G.nodes()):
            self.grid.place_agent(agent, node)

        # DataCollector per raccogliere i dati
        self.datacollector = mesa.DataCollector(
            agent_reporters={"Strategy": "strategy", "Wealth": "wealth"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("update_payoff")
        self.agents.shuffle_do("update_strategy")

# === ESECUZIONE SIMULAZIONE ===
model = ImitationModel()
for _ in range(n_steps):
    model.step()

# === ANALISI E VISUALIZZAZIONE ===
#df = model.datacollector.get_agent_vars_dataframe()
#strategy_counts = df['Strategy'].unstack().apply(lambda x: x.value_counts(normalize=True), axis=1).fillna(0)
# wealth_history = df['Wealth'].unstack().mean(axis=1)

# plt.figure(figsize=(10, 6))
# strategy_counts.plot(marker='o')
# plt.title("Distribuzione Strategie nel Tempo")
# plt.xlabel("Passi di Simulazione")
# plt.ylabel("Frequenza Relativa")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(wealth_history, marker='o', color='green')
# plt.title("Evoluzione Ricchezza Media")
# plt.xlabel("Passi di Simulazione")
# plt.ylabel("Ricchezza Media")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.stackplot(strategy_counts.index,
              #strategy_counts.get(0, 0), strategy_counts.get(1, 0),
              #labels=['0', '1'], colors=['orangered', 'limegreen'])
# plt.title("Distribuzione Strategie (Stacked Area)", fontsize=12, fontweight='bold')
#plt.xlabel("Passi di Simulazione")
#plt.ylabel("Frequenza")
#plt.legend(loc='upper right')
#plt.grid(False)
#plt.tight_layout()
#plt.show()
