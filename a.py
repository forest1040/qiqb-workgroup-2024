from quri_parts_qsci import qsci
import numpy as np
from pyscf import scf, gto, fci
from quri_parts.pyscf.mol import get_spin_mo_integrals_from_mole
from quri_parts.openfermion.mol import get_qubit_mapped_hamiltonian
from quri_parts.openfermion.ansatz import KUpCCGSD
from quri_parts.core.operator import Operator
from quri_parts.core.state import apply_circuit, ParametricCircuitQuantumState
from quri_parts.circuit.utils.circuit_drawer import draw_circuit
from quri_parts.qulacs.sampler import create_qulacs_vector_concurrent_sampler

from quri_parts.qulacs.estimator import (
    create_qulacs_vector_parametric_estimator,
    create_qulacs_vector_concurrent_parametric_estimator
)
from quri_parts.core.estimator.gradient import create_parameter_shift_gradient_estimator
from quri_parts.algo.optimizer import OptimizerStatus, LBFGS, CostFunction, GradientFunction, Optimizer, OptimizerState
from quri_parts.core.state import GeneralCircuitQuantumState

def get_hamiltonian_and_ansatz(mole: gto.Mole) -> tuple[Operator, ParametricCircuitQuantumState]:
    mf = scf.RHF(mole).run(verbose=0)

    hamiltonian, mapping = get_qubit_mapped_hamiltonian(
        *get_spin_mo_integrals_from_mole(mole, mf.mo_coeff)
    )

    ansatz_circuit = KUpCCGSD(n_spin_orbitals=2*mole.nao, k=1, fermion_qubit_mapping=mapping)
    hf_state = mapping.state_mapper([n for n in range(mapping.n_fermions)])
    ansatz_state = apply_circuit(ansatz_circuit, hf_state)
    return hamiltonian, ansatz_state

def get_cost_function(
    hamiltonian: Operator,
    ansatz_state: ParametricCircuitQuantumState,
) -> CostFunction:
    parametric_estimator = create_qulacs_vector_parametric_estimator()
    return lambda param: parametric_estimator(hamiltonian, ansatz_state, param).value.real

def get_grad_function(
    hamiltonian: Operator,
    ansatz_state: ParametricCircuitQuantumState,
) -> GradientFunction:
    cp_estimator = create_qulacs_vector_concurrent_parametric_estimator()
    gradient_estimator = create_parameter_shift_gradient_estimator(cp_estimator)
    return lambda param: np.array(gradient_estimator(hamiltonian, ansatz_state, param).values).real

def vqe(
    hamiltonian: Operator,
    ansatz_state: ParametricCircuitQuantumState,
    init_param: list[float],
    optimizer: Optimizer,
) -> OptimizerState:
    cost = get_cost_function(hamiltonian, ansatz_state)
    grad = get_grad_function(hamiltonian, ansatz_state)
    op_state = optimizer.get_init_state(init_param)
    itr = 0
    while True:
        itr += 1
        op_state = optimizer.step(op_state, cost, grad)
        print(f"{itr}:{op_state.cost}")
        if op_state.status == OptimizerStatus.CONVERGED or op_state.status == OptimizerStatus.FAILED:
            break
    return op_state

h2_mole = gto.M(atom=f"H 0 0 0; H 0 0 1", basis = 'sto-3g')
#h2_mole = gto.M(atom=f"H 0 0 0; H 0 0 1", basis = '6-31g')
#h2_mole = gto.M(atom=f"H 0 0 0; H 0 0 1", basis = 'ccpvdz')
h2_hf = scf.RHF(h2_mole).run(verbose=0)
print("Hartree Fock:", h2_hf.e_tot)
#Hartree Fock: -1.0661086493179366

h2_fci = fci.FCI(h2_hf)
print("Full CI:", h2_fci.kernel()[0])
#Full CI: -1.1011503302326187

hamiltonian, ansatz_state = get_hamiltonian_and_ansatz(h2_mole)
init_param = np.random.random(ansatz_state.parametric_circuit.parameter_count)
optimizer = LBFGS(gtol=1e-1)  # 軽く最適化する
op_state = vqe(hamiltonian, ansatz_state, init_param, optimizer)
# -0.8971780476047228
# -1.0036762536144055
# -1.067611372298007
# -1.0838100789225809
# -1.098495249402842
print("VQE optimized:", op_state.cost)
#VQE optimized: -1.098495249402842
#print("op_state:", op_state.vector)
# print("op_state:", dir(op_state))
# s = op_state.getstate()
# print("s:", s)

optimized_state = ansatz_state.bind_parameters(op_state.params)
#print("optimized_state:", optimized_state.vector)

sampler = create_qulacs_vector_concurrent_sampler()
result = qsci(
    hamiltonian=hamiltonian,
    approx_states=[optimized_state],
    sampler=sampler,
    total_shots=1000,
)
print("QSCI:", result[0][0])
#QSCI: -1.1011503302326187

# # zero state qsci
# n = ansatz_state.parametric_circuit.qubit_count
# zero_state = GeneralCircuitQuantumState(n)
# result = qsci(
#     hamiltonian=hamiltonian,
#     approx_states=[zero_state],
#     sampler=sampler,
#     total_shots=1000,
# )
# print("QSCI:", result[0][0])
# # QSCI: (0.52917721092+0j)
