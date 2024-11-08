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

qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
x q[0];
x q[1];
"""

from pyscf import scf, gto, fci

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

n = 4
# quri-partsに取り込む
from qasm_to_quri import convert_qasm_to_circuit
quri_circuit = convert_qasm_to_circuit(qasm.splitlines())

from quri_parts.core.state import GeneralCircuitQuantumState
quri_state = GeneralCircuitQuantumState(n, quri_circuit)

# QSCI(qulacs)
from quri_parts.qulacs.sampler import create_qulacs_vector_concurrent_sampler
sampler = create_qulacs_vector_concurrent_sampler()
sim_result = qsci(
    hamiltonian=hamiltonian,
    approx_states=[quri_state],
    sampler=sampler,
    total_shots=1000,
)
print("QSCI(sim):", sim_result[0][0])

# QSCI(real)
from quri_parts.riqu.backend import RiquSamplingBackend
from qsci_riqu import qsci_riqu

backend = RiquSamplingBackend()
real_result = qsci_riqu(
    hamiltonian=hamiltonian,
    approx_states=[quri_state],
    backend=backend,
    total_shots=1000,
)
print("QSCI(real):", real_result[0][0])

