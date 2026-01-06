import numpy as np
import matplotlib.pyplot as plt

from Utils import GSignalGen as GSG
from Utils import NoDriScUtils as NDS

# Small demo graph (undirected, unweighted line graph)
N, T = 40, 2000
A = GSG.generate_random_graph_matrix(N = N, is_directed = True)

sim = GSG.GraphSignalSimulator(A=A, fs=500, T=T, R=10)

# Example 1: single spiky driver (smoothed) on node 0
X1, U1 = sim.simulate(driver_nodes=0, driver_specs="ar-6", snr_db=0, copy_behavior=False)
# specs = ["sin-6-1.0", "ar-6", "square-20-0.7"]

# Quick plot: show 4 nodes of X2
plt.figure(figsize=(10,4))
for idx, n in enumerate([0, 1, 2, 3, 5]):
    plt.plot(X1[n, :], label=f"Node {n}")
plt.legend()
plt.title("Example: Mixed drivers (sin, square, AR) diffused on graph")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

oni = NDS.jt_oni_topological(X1,sim.S,R=15)
oni_cf = NDS.jt_oni_counterfactual(X1,sim.S,R=15)

print("Topological ONI:", oni)
print("Counterfactual ONI:", oni_cf)

plt.figure()
plt.plot(oni / np.sum(np.abs(oni)),'o-',label='Topological')
plt.plot(oni_cf / np.sum(np.abs(oni_cf)),'s-',label='Counterfactual')
plt.legend(); plt.xlabel("Node"); plt.ylabel("ONI")
plt.show()