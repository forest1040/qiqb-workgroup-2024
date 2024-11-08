from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import X, Y

n = 4
state = QuantumState(n)
qulacs_circuit = QuantumCircuit(n)
qulacs_circuit.add_gate(X(0))
qulacs_circuit.add_gate(X(1))
qulacs_circuit.update_quantum_state(state)
print(state)
