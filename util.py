import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute
from qiskit.quantum_info import random_statevector, state_fidelity, Statevector
from qiskit.extensions import Initialize, UnitaryGate

# Here are 3 options of scrambling operators:
# This is the exact matrix from eq.4 in Methods from the main article (scrambling), multiplied by i (j in python).
mat1 = 0.5j * np.array([[-1, 0, 0, -1, 0, -1, -1, 0], [0, 1, -1, 0, -1, 0, 0, 1], [0, -1, 1, 0, -1, 0, 0, 1],
                        [1, 0, 0, 1, 0, -1, -1, 0], [0, -1, -1, 0, 1, 0, 0, 1], [1, 0, 0, -1, 0, 1, -1, 0],
                        [1, 0, 0, -1, 0, -1, 1, 0], [0, -1, -1, 0, -1, 0, 0, -1]])

# As mat1, but with a (-) factor.
mat2 = -0.5j * np.array([[-1, 0, 0, -1, 0, -1, -1, 0], [0, 1, -1, 0, -1, 0, 0, 1], [0, -1, 1, 0, -1, 0, 0, 1],
                         [1, 0, 0, 1, 0, -1, -1, 0], [0, -1, -1, 0, 1, 0, 0, 1], [1, 0, 0, -1, 0, 1, -1, 0],
                         [1, 0, 0, -1, 0, -1, 1, 0], [0, -1, -1, 0, -1, 0, 0, -1]])

# As mat1, with a (-j) factor).
mat3 = 0.5 * np.array([[-1, 0, 0, -1, 0, -1, -1, 0], [0, 1, -1, 0, -1, 0, 0, 1], [0, -1, 1, 0, -1, 0, 0, 1],
                       [1, 0, 0, 1, 0, -1, -1, 0], [0, -1, -1, 0, 1, 0, 0, 1], [1, 0, 0, -1, 0, 1, -1, 0],
                       [1, 0, 0, -1, 0, -1, 1, 0], [0, -1, -1, 0, -1, 0, 0, -1]])

scramble_gate2_1 = UnitaryGate(mat1)
scramble_gate2_1.label = 'scrambler_2_1'
scramble_gate2_1_c = UnitaryGate(mat1.conjugate())
scramble_gate2_1_c.label = 'scrambler_2_1_c'

scramble_gate2_2 = UnitaryGate(mat1)
scramble_gate2_2.label = 'scrambler_2_2'
scramble_gate2_2_c = UnitaryGate(mat2.conjugate())
scramble_gate2_2_c.label = 'scrambler_2_2_c'

scramble_gate2_3 = UnitaryGate(mat3)
scramble_gate2_3.label = 'scrambler_2_3'
scramble_gate2_3_c = UnitaryGate(mat3.conjugate())
scramble_gate2_3_c.label = 'scrambler_2_3_c'


# Make a circuit that makes a bell pair out of the two given qubits, give it a name.
def create_bell_pair(qc, a, b):
    """Creates a bell pair in QuantumCircuit qc between qubits a and b"""
    qc.h(a)  # Put qubit a into state |+>
    qc.cx(a, b)  # CNOT with a as control and b as target


qreg = QuantumRegister(2, 'q')
circ = QuantumCircuit(qreg)
create_bell_pair(circ, qreg[0], qreg[1])

bell_gate = circ.to_gate(label='bell')

inv_Bell_circ = circ.inverse()
inv_Bell_gate = inv_Bell_circ.to_gate(label='inv_bell')


def circ25(tfd, beta, init_q0=None):
    """ Given a tfd state, a temperature (in the form of beta, which is 1/T) and an initial state for q0 (random
    if given no such argument), returns a list consisting of the given beta, the amount of 'bad counts' (the amount
    of counts of the state |1000000>) and the successful measurement probability"""
    # Reminder: qubit 3 is 5 and 5 is 3. With this transformation, we get the right couples as in the article.

    # circuit will be our main circuit.
    qreg_q = QuantumRegister(7, 'q')
    creg_c = ClassicalRegister(7, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    if not init_q0 is None:
        if init_q0 == 'pz0':
            psi = [1, 0]
        elif init_q0 == 'pz1':
            psi = [0, 1]
        elif init_q0 == 'px0':
            psi = [1/np.sqrt(2), 1/np.sqrt(2)]
        elif init_q0 == 'px1':
            psi = [1/np.sqrt(2), -1/np.sqrt(2)]
        elif init_q0 == 'py0':
            psi = [-1j/np.sqrt(2), -1/np.sqrt(2)]
        elif init_q0 == 'py1':
            psi = [-1j/np.sqrt(2), 1/np.sqrt(2)]
    else:
        # Generating a random state_vector:
        psi = random_statevector(dims=2)

    # Make a gate from this operator:
    init_gate = Initialize(psi)
    init_gate.label = 'init_gate'

    # Save the inverse of the gate (in order to return later to the state psi - with the 'target' qubit of the teleportation)
    inverse_init_gate = init_gate.gates_to_uncompute()
    inverse_init_gate.label = 'inverse_init_gate'

    # Here we initialize qubits 1-6 states as the tfd, which we constructed in Mathematica:
    psi2 = tfd
    init_gate_1_6 = Initialize(psi2)
    init_gate_1_6.label = 'init_gate 1-6'

    # Add the initializing gate to the first qubit (the one to be teleported).
    circuit.append(init_gate, [qreg_q[0]])

    # Add the initializing gate to qubits 1-6.
    circuit.append(init_gate_1_6, [qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6]])

    circuit.barrier()

    # Apply the scrambling unitary to qubits 0-2 and U^* to 3-5.
    circuit.append(scramble_gate2_1, [0, 1, 2])
    circuit.append(scramble_gate2_1_c, [5, 4, 3])

    circuit.barrier()

    circuit.append(inv_Bell_gate, [qreg_q[0], qreg_q[3]])
    circuit.append(inv_Bell_gate, [qreg_q[1], qreg_q[4]])
    circuit.append(inv_Bell_gate, [qreg_q[2], qreg_q[5]])

    circuit.barrier()

    # Measure qubits 2, 5. Save result in corresponding classical bits.
    circuit.measure([qreg_q[2], qreg_q[5]], [creg_c[2], creg_c[5]])

    # Append to the circuit the inverse of 'init_gate' on qubit 6 (the 'target'):
    circuit.append(inverse_init_gate, [qreg_q[6]])

    circuit.barrier()

    # Add the measurement of qubit 6, to classical bit 6:
    circuit.measure(qreg_q[6], creg_c[6])

    # backend = BasicAer.get_backend('qasm_simulator')
    backend = Aer.get_backend('aer_simulator')
    shots = 20000
    job = execute(circuit, backend, shots=shots)
    counts = job.result().get_counts()
    bad_counts = counts.get('1000000', 0)
    prob_for_0000000 = counts['0000000'] / shots
    return [beta, bad_counts, prob_for_0000000]

def circ25_noMeasurements_forFidelity(tfd, beta, init_q0=None):
    """
    Exactly as circ25, only here we eliminate the measurements of all qubits. This is done so the system
    quantum state won't collapse, which is necessary for the measurement of the state fidelity
    """

    # circuit will be our main circuit.
    qreg_q = QuantumRegister(7, 'q')
    creg_c = ClassicalRegister(7, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    if not init_q0 is None:
        if init_q0 == 'pz0':
            psi = [1, 0]
        elif init_q0 == 'pz1':
            psi = [0, 1]
        elif init_q0 == 'px0':
            psi = [1/np.sqrt(2), 1/np.sqrt(2)]
        elif init_q0 == 'px1':
            psi = [1/np.sqrt(2), -1/np.sqrt(2)]
        elif init_q0 == 'py0':
            psi = [-1j/np.sqrt(2), -1/np.sqrt(2)]
        elif init_q0 == 'py1':
            psi = [-1j/np.sqrt(2), 1/np.sqrt(2)]
    else:
        # Generating a random state_vector:
        psi = random_statevector(dims=2)

    # Make a gate from this operator:
    init_gate = Initialize(psi)
    init_gate.label = 'init_gate'

    # Save the inverse of the gate (in order to return later to the state psi - with the 'target' qubit of the teleportation)
    inverse_init_gate = init_gate.gates_to_uncompute()
    inverse_init_gate.label = 'inverse_init_gate'

    # Reminder: qubit 3 is 5 and 5 is 3. With this transformation, we get the right couples as in the article.

    # Here we initialize qubits 1-6 states as the tfd, which we constructed in Mathematica:
    psi2 = tfd
    init_gate_1_6 = Initialize(psi2)
    init_gate_1_6.label = 'init_gate 1-6'

    # Add the initializing gate to the first qubit (the one to be teleported).
    circuit.append(init_gate, [qreg_q[0]])

    # Add the initializing gate to qubits 1-6.
    circuit.append(init_gate_1_6, [qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6]])

    circuit.barrier()

    # Apply the scrambling unitary to qubits 0-2 and U^* to 3-5.
    circuit.append(scramble_gate2_1, [0, 1, 2])
    circuit.append(scramble_gate2_1_c, [5, 4, 3])

    circuit.barrier()

    circuit.append(inv_Bell_gate, [qreg_q[0], qreg_q[3]])
    circuit.append(inv_Bell_gate, [qreg_q[1], qreg_q[4]])
    circuit.append(inv_Bell_gate, [qreg_q[2], qreg_q[5]])

    circuit.barrier()

    # Append to the circuit the inverse of 'init_gate' on qubit 6 (the 'target'):
    circuit.append(inverse_init_gate, [qreg_q[6]])

    circuit.barrier()

    # Save state_vector of the system (when running).

    backend = Aer.get_backend('aer_simulator')
    circuit.save_statevector()
    shots = 20000
    job = execute(circuit, backend, shots=shots)

    # Get state_vector of the system:
    st = job.result().get_statevector(circuit)

    # Return state_fidelity with [1] + [0]*127.
    return [state_fidelity(st, Statevector([1] + [0] * 127)), beta]


def circ2514(tfd, beta, init_q0):
    """ Exactly as circ25, only here we measure the state of qubits 1 and 4 additionaly."""
    # Reminder: qubit 5 is qubit 3 and 3 is 5 from the moment qubits 1-6 are initialized in the TFD state.

    # circuit will be our main circuit.
    qreg_q = QuantumRegister(7, 'q')
    creg_c = ClassicalRegister(7, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    if not init_q0 is None:
        if init_q0 == 'pz0':
            psi = [1, 0]
        elif init_q0 == 'pz1':
            psi = [0, 1]
        elif init_q0 == 'px0':
            psi = [1/np.sqrt(2), 1/np.sqrt(2)]
        elif init_q0 == 'px1':
            psi = [1/np.sqrt(2), -1/np.sqrt(2)]
        elif init_q0 == 'py0':
            psi = [-1j/np.sqrt(2), -1/np.sqrt(2)]
        elif init_q0 == 'py1':
            psi = [-1j/np.sqrt(2), 1/np.sqrt(2)]
    else:
        # Generating a random state_vector:
        psi = random_statevector(dims=2)

    # Make a gate from this operator:
    init_gate = Initialize(psi)
    init_gate.label = 'init_gate'

    # Save the inverse of the gate (in order to return later to the state psi - with the 'target' qubit of the teleportation)
    inverse_init_gate = init_gate.gates_to_uncompute()
    inverse_init_gate.label = 'inverse_init_gate'

    # Reminder: qubit 3 is 5 and 5 is 3. With this transformation, we get the right couples as in the article.

    # Here we initialize qubits 1-6 states as the tfd, which we constructed in Mathematica:
    psi2 = tfd
    init_gate_1_6 = Initialize(psi2)
    init_gate_1_6.label = 'init_gate 1-6'

    # Add the initializing gate to the first qubit (the one to be teleported).
    circuit.append(init_gate, [qreg_q[0]])

    # Add the initializing gate to qubits 1-6.
    circuit.append(init_gate_1_6, [qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4], qreg_q[5], qreg_q[6]])

    circuit.barrier()

    # Apply the scrambling unitary to qubits 0-2 and U^* to 3-5.
    circuit.append(scramble_gate2_1, [0, 1, 2])
    circuit.append(scramble_gate2_1_c, [5, 4, 3])

    circuit.barrier()

    circuit.append(inv_Bell_gate, [qreg_q[0], qreg_q[3]])
    circuit.append(inv_Bell_gate, [qreg_q[1], qreg_q[4]])
    circuit.append(inv_Bell_gate, [qreg_q[2], qreg_q[5]])

    circuit.barrier()

    # Measure qubits 2, 5, 1, 4. Save result in corresponding classical bits.
    circuit.measure([qreg_q[2], qreg_q[5], qreg_q[1], qreg_q[4]], [creg_c[2], creg_c[5], creg_c[1], creg_c[4]])

    # Append to the circuit the inverse of 'init_gate' on qubit 6 (the 'target'):
    circuit.append(inverse_init_gate, [qreg_q[6]])

    circuit.barrier()

    # Add the measurement of qubit 6, to classical bit 6:
    circuit.measure(qreg_q[6], creg_c[6])

    backend = Aer.get_backend('aer_simulator')
    shots = 20000
    job = execute(circuit, backend, shots=shots)
    counts = job.result().get_counts()

    bad_counts = counts.get('1000000', 0)
    prob_for_0000000 = counts['0000000'] / shots
    return [beta, bad_counts, prob_for_0000000]