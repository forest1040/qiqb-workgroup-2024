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

# H2N2分子の設定
mol = gto.M(
    atom = '''
        N   0.0000   0.0000   0.7000
        N   0.0000   0.0000  -0.7000
        H   0.0000   0.9000   1.2000
        H   0.0000  -0.9000   1.2000
    ''',
    basis = 'sto-3g',
    symmetry = True
)

hf = scf.RHF(mol).run(verbose=0)
print("Hartree Fock:", hf.e_tot)
#Hartree Fock: -1.0661086493179366

fci = fci.FCI(hf)
print("Full CI:", fci.kernel()[0])
#Full CI: -1.1011503302326187

hamiltonian, ansatz_state = get_hamiltonian_and_ansatz(mol)
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

optimized_state = ansatz_state.bind_parameters(op_state.params)
sampler = create_qulacs_vector_concurrent_sampler()
result = qsci(
    hamiltonian=hamiltonian,
    approx_states=[optimized_state],
    sampler=sampler,
    total_shots=1000,
)
print("QSCI:", result[0][0])
#QSCI: -1.1011503302326187
