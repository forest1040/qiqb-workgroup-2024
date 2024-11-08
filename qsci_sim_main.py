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

import re

def expand_qasm_macros(src_qasm):
    pi_mul_pattern = re.compile(r'pi\*([+-]?\d+(?:\.\d*)?|.\d+)')
    pi_div_pattern = re.compile(r'pi/([+-]?\d+(?:\.\d*)?|.\d+)')
    mul_pattern_minus = re.compile(r'-\d\*([+-]?\d+(?:\.\d*)?|.\d+)')
    div_pattern_minus = re.compile(r'-\d/([+-]?\d+(?:\.\d*)?|.\d+)')
    mul_pattern = re.compile(r'\d\*([+-]?\d+(?:\.\d*)?|.\d+)')
    div_pattern = re.compile(r'\d/([+-]?\d+(?:\.\d*)?|.\d+)')
    lines = src_qasm.splitlines()
    macros = {}
    res_qasm = ''
    for line in lines:
        if line == "":
            continue
        line = pi_mul_pattern.sub(lambda match : str(np.pi*float(match.group(1))), line)
        line = pi_div_pattern.sub(lambda match : str(np.pi/float(match.group(1))), line)
        line = mul_pattern_minus.sub(lambda match : str(eval(match.group(0))), line)
        line = div_pattern_minus.sub(lambda match : str(eval(match.group(0))), line)
        line = mul_pattern.sub(lambda match : str(eval(match.group(0))), line)
        line = div_pattern.sub(lambda match : str(eval(match.group(0))), line)
        #print("line:",line)
        line = line.replace('pi', str(np.pi))
        words = line.split()
        name = words[0]
        if name == 'gate':
            key = words[1]
            args = words[2].split(',')
            gates = list(map(lambda g:g + ';' , ' '.join(words[4:-1]).split(';')[:-1]))
            macros[key] = (args, gates)
        elif name in macros:
            img_args = words[1][:-1].split(',')
            data = macros[name]
            for gate in data[1]:
                res_line = gate.strip()
                for i in range(len(img_args)):
                    res_line = res_line.replace(data[0][i], img_args[i])
                res_qasm += res_line + '\n'
        else:
            res_qasm += line + '\n'
    return res_qasm


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


# qulacsに変換
from quri_parts.qulacs.circuit import convert_parametric_circuit
qulacs_circuit, param_mapper = convert_parametric_circuit(ansatz_state.parametric_circuit)
n = ansatz_state.parametric_circuit.qubit_count
#print(op_state.params)
for i, v in enumerate(param_mapper(op_state.params)):
    qulacs_circuit.set_parameter(i, v)

from qulacs import QuantumState, QuantumCircuit
state = QuantumState(n)
qulacs_circuit.update_quantum_state(state)
#print(state)

# AQCE
from AQCE_from_python import AQCE_python
qulacs_gates = AQCE_python(
    state,
    M_0=2,
    M_delta=1,
    M_max=4,
    N=2,
)
aqce_circuit = QuantumCircuit(state.get_qubit_count())
for gate in qulacs_gates:
    aqce_circuit.add_gate(gate)
    #print(gate)

#aq_state = QuantumState(n)
#circuit.update_quantum_state(aq_state)
#print(aq_state)


# qiskitに変換
from qiskit_convert import qulacs_gates_to_qiskit
qiskit_circuit = qulacs_gates_to_qiskit(aqce_circuit)

# QASMに変換
#from qiskit.qasm3 import dumps
from qiskit.qasm2 import dumps

qiskit_qasm = dumps(qiskit_circuit)
print("qiskit_qasm:", qiskit_qasm)

# QASMを整形
qasm = expand_qasm_macros(qiskit_qasm)
print("qasm:", qasm)

# qasm: OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[4];
# u(1.5707963267948966,-1.5707963267948966,0.39269908170102363) q[2];
# u(1.4696906269787318,1.5707963267948966,1.1780972450961724) q[1];
# cx q[2],q[1];
# u(3.1414742314361392,0,1.5707963267948966) q[2];
# u(1.570796326090775,1.5707843740287695,-1.5709141441868932) q[1];
# cx q[2],q[1];
# u(1.5707963267948966,-0.3926990817010232,0) q[2];
# u(0.10110569981616471,-1.1780972450961724,-1.5707963267948966) q[1];
# u(1.5707963267948966,-1.5707963267948966,-3.141592653589793) q[1];
# u(1.6719020266110614,-1.5707963267948966,-3.141592653589793) q[0];
# cx q[1],q[0];
# u(1.5707963267948966,0,1.5707963267948966) q[1];
# u(1.4702087094713374,1.4691752037207921,-3.1313531404558756) q[0];
# cx q[1],q[0];
# u(1.5707963267948966,-1.5707963267948966,-3.141592653589793) q[1];
# u(0.10110569981616471,-1.5707963267948966,-1.5707963267948966) q[0];
# u(1.5707963267948966,1.5707963267948966,-2.748893571891069) q[2];
# u(1.4696906269787318,1.5707963267948966,-1.1780972450961724) q[0];
# cx q[2],q[0];
# u(3.1098509121317184,0,1.5707963267948966) q[2];
# u(1.5707457435182879,1.567593053222831,-1.602375914610685) q[0];
# cx q[2],q[0];
# u(1.5707963267948966,2.748893571891069,-3.141592653589793) q[2];
# u(0.10110569981616473,1.1780972450961724,-1.5707963267948966) q[0];
# u(1.5707963267948966,1.5707963267948966,-1.1780972450961724) q[3];
# u(1.4696906269787315,1.5707963267948966,0.39269908169872414) q[1];
# cx q[3],q[1];
# u(3.1200879500476555,0,1.5707963267948966) q[3];
# u(1.5707731084047076,1.5686259467136612,-1.592191192702517) q[1];
# cx q[3],q[1];
# u(1.5707963267948966,1.1780972450961724,-3.141592653589793) q[3];
# u(0.10110569981616475,-0.39269908169872414,-1.5707963267948966) q[1];

# from qasm_converter_fixed import convert_QASM_to_qulacs_circuit
# convert_QASM_to_qulacs_circuit()

# quri-partsに取り込む
from qasm_to_quri import convert_qasm_to_circuit
quri_circuit = convert_qasm_to_circuit(qasm.splitlines())

from quri_parts.core.state import GeneralCircuitQuantumState
quri_state = GeneralCircuitQuantumState(n, quri_circuit)

# QSCI(qulacs)
sampler = create_qulacs_vector_concurrent_sampler()
result = qsci(
    hamiltonian=hamiltonian,
    approx_states=[quri_state],
    sampler=sampler,
    total_shots=1000,
)
print("QSCI:", result[0][0])
