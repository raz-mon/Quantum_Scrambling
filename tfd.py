import numpy as np
import math
import itertools
from matplotlib import pyplot as plt
from random import random, sample
from scipy.optimize import minimize
# comment

from whole_process import tfd_generator

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex
from qiskit.extensions import Initialize, UnitaryGate
from qiskit.quantum_info import random_statevector, state_fidelity, Statevector

# generate random angles, needed to initialize the problem
def random_angles(n):
    angles = []
    for i in range(n):
        angles.append(2*math.pi*random())
    return angles

"""
# we define functions to convert between the 01 notation (string) and
# -+ notation (array)
def from_01_to_pm(string):
    return np.ones(num_assets) - 2 * np.array(list(map(int,i)))
"""


def from_pm_to_01(array):
    str = ''
    for el in array:
        if el == 1:
            str = str + '0'
        elif el == -1:
            str = str + '1'
    return str


class ThetaError(Exception):
    """Error raised whenever the angles provided to the variational circuit
    don't match with the circuit shape.
    """


class VariationalCircuit(object):
    """Implement a generic variational circuit that can then be used with a
    variational optimization algorithm.
    """

    def _circuitBuilder(self):
        """ The way the circuit is build depends on the particular
        circuit structure. theta should be a list of angles.
        """
        pass

    def __init__(self, theta):
        """Initialize a new VariationalCircuit, with angles as parameters.
        """
        self.theta = theta
        self.circuit = self._circuitBuilder(theta)

    def update_theta(self, theta):
        if len(theta) == len(self.theta):
            self.circuit = self._circuitBuilder(theta)
            self.theta = theta
        else:
            print(f"theta should be a list of {len(theta)} angles.")
            raise ThetaError

class QAOA_TFD(VariationalCircuit):
    """Implement the variational circuit from https://arxiv.org/abs/2112.02068.
    The circuit is made by three types of blocks: a gate preparing the infinite
    temperature TFD, time evolution by a Hamiltonian whose ground state is the
    infinite temperature TFD, and time evolution by a Hamiltonian whose ground
    state is the zero temperature TFD. The amount of time evolution we perform
    at each step corresponds to the angles.
    """
    def _prepare_infinite_temp(self, n):
        # first we make a circuit with twice the number of qubits
        circuit = QuantumCircuit(2*n)
        # we add the R_XX gates
        for i in range(n):
            circuit.rxx(np.pi/2, i, i+n)
        # we add the R_Z gates
        for i in range(n):
            circuit.rz(np.pi/4, i)
            circuit.rz(np.pi/4, i+n)
        # we turn the circuit into a gate and return it
        return circuit.to_gate(label="UInf")

    # To build the circuit we need the following blocks
    def _U_infinite_temp(self, n, alpha1, alpha2):
        # first we make a circuit with twice the number of qubits
        circuit = QuantumCircuit(2*n)
        # we add the R_XX gates
        for i in range(n):
            circuit.rxx(alpha1, i, i+n)
        for i in range(n):
            circuit.rzz(alpha2, i, i+n)
        # we turn the circuit into a gate and return it
        return circuit.to_gate(label="UInf")

    def _U_zero_temp(self, n, gamma1, gamma2):
        # first we make a circuit with twice the number of qubits
        circuit = QuantumCircuit(2*n)
        # we add the R_XX gates
        for i in range(n-1):
            circuit.rxx(gamma2, i, i+1)
            circuit.rxx(gamma2, i+n, i+1+n)
        # we add the R_Z gates
        for i in range(n):
            circuit.rz(gamma1, i)
            circuit.rz(gamma1, i+n)
        # we turn the circuit into a gate and return it
        return circuit.to_gate(label="U0")

    def __init__(self, n, d, theta):
        """Takes a list of angles, theta, and (n,d) as args, where n is the
        number of qubits (in one copy of the system) and d is
        the depth of the circuit.
        """
        try:
            self.n = n
            self.d = d
        except (ValueError, TypeError):
            print("You should input two integers, n and d, where n is the \
                    number of qubits and d is the depth of the circuit.")
        super().__init__(theta)

    def _circuitBuilder(self, theta):
        # make a quantum circuit with 2n qubits
        circuit = QuantumCircuit(2*self.n)
        # first we prepare the infinite temperature TFD. This can be done
        circuit.append(self._prepare_infinite_temp(self.n), range(2*self.n))
        # add d layers of alternating time evolutions
        # theta = (alpha1, alpha2, gamma1, gamma2)
        for i in range(self.d):
            circuit.append(self._U_zero_temp(self.n, theta[4*i], theta[4*i+1]), range(2*self.n))
            circuit.append(self._U_infinite_temp(self.n, theta[4*i+2], theta[4*i+3]), range(2*self.n))
        return circuit

class StateCooker(object):
    """Implement a quantum algorithm that prepares the target state,
    working with qiskit.
    """

    def __init__(self, var_circuit, target_state, backend, optimizer, shots=1024):
        """As input, we need the variational circuit, which should prepare the
        state, the target state, a backend to run the simulation and
        an optimizer. Shots sets how many times we run the circuit to estimate
        the state.
        """
        self.var_circuit = var_circuit
        self.target_state = target_state
        self.simulator = Aer.get_backend(backend)
        # the optimizer should be a function that takes only two arguments:
        self.optimizer = optimizer
        self.shots = shots
        # everytime we run the circuit we add a new element to results:
        # the angles used is the key, the fidelity the value
        self.results = {}
        self.last_thetas = [5.6388085, 3.04771788, 4.7027956, 5.31299131, 0, 0, 0, 0]

    def run(self, theta=[None]):
        """Runs the circuit and return the 1-fidelity between the output state
        and the target state.
        """
        if list(theta) != [None]:
            self.var_circuit.update_theta(theta)
        circuit = self.var_circuit.circuit
        # we add a layer to the circuit needed to save the final state
        circuit.save_statevector()
        # we transpile the circuit with the chosen backend
        circuit = transpile(circuit, self.simulator)
        # we run the circuit
        out_state = self.simulator.run(circuit, shots=self.shots).result().get_statevector(circuit)
        # later we want to minimize, so we need to take 1-F
        return 1-state_fidelity(out_state, self.target_state)


    def run_1_layer(self, thetas=[None]):
        """

        :param theta:
        :return:
        """
        new_thetas = self.last_thetas[0:4]
        new_thetas = np.concatenate((new_thetas, thetas))
        return self.run(new_thetas)


    def optimize(self):
        """Use the given optimizer to find the value of theta which gives an
        out_state as close as possible to the target_state. Everytime you call
        optimize any previous result is erased. Notice that the optimal value in
        general is different from the value of the last call, which you get by
        using get_result."
        """
        self.results = {}
        result = self.optimizer(self.run, self.var_circuit.theta)
        return result

    def optimize_1_layer(self):
        self.results = {}
        result = self.optimizer(self.run_1_layer, self.var_circuit.theta[-4:])
        return result

    def get_result(self, n=1):
        """Returns the last result, or the n-th last result."""
        key = list(self.results.keys())[-n]
        return self.results[key]

    def plot(self):
        """Plot the fidelity as we optimize the angles.
        """
        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.results, label="Fidelity")
        plt.show()


def infinite_temp(framework: str):
    qreg = QuantumRegister(6)
    creg = ClassicalRegister(6)
    qc = QuantumCircuit(qreg, creg)
    # Prepare system in tfd state.
    qc.rxx(np.pi/2, qreg[0], qreg[3])
    qc.rxx(np.pi/2, qreg[1], qreg[4])
    qc.rxx(np.pi/2, qreg[2], qreg[5])
    qc.barrier()
    for n in range(6):
        qc.rz(np.pi/4, qreg[n])
    # Prepare a tfd state in the max coherence state (T->inf) and measure fidelity.
    tfd_ref = tfd_generator().generate_tfd(0)           # Beta=0 <==> T->infty.

    if framework == 'real':
        # Load account settings:
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q-research-2', group='ben-gurion-uni-1', project='main')

        # get the least-busy backend at IBM and run the quantum circuit there
        from qiskit.providers.ibmq import least_busy
        from qiskit.tools.monitor import job_monitor

        backend = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 6 and
                                                                 not b.configuration().simulator and b.status().operational == True))
        t_qc = transpile(qc, backend, optimization_level=1)
        job = backend.run(t_qc)
        job_monitor(job)  # displays job status under cell

        exp_res = job.result()
        sv = exp_res.get_statevector(qc)
        return state_fidelity(tfd_ref, sv)
    else:       # framework='sim'
        backend = Aer.get_backend('aer_simulator')
        qc.save_statevector()
        shots = 20000
        job = execute(qc, backend, shots=shots)

        # Get state_vector of the system:
        st = job.result().get_statevector(qc)

        # Return state_fidelity with [1] + [0]*127.
        return state_fidelity(st, tfd_ref)


def zero_temp(framework: str):
    qreg = QuantumRegister(6)
    creg = ClassicalRegister(6)
    qc = QuantumCircuit(qreg, creg)
    # Prepare system in tfd state.
    qc.rxx(np.pi/2, qreg[0], qreg[1])
    qc.rxx(np.pi/2, qreg[1], qreg[2])
    qc.rxx(np.pi/2, qreg[3], qreg[4])
    qc.rxx(np.pi/2, qreg[4], qreg[5])
    qc.barrier()
    for n in range(6):
        qc.rz(qreg[n])/4
    # Prepare a tfd state in the zero coherence state (T->inf) and measure fidelity.
    tfd_ref = tfd_generator().generate_tfd(pow(10, 8))  # Very large beta.

    if framework == 'real':
        # Load account settings:
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q-research-2', group='ben-gurion-uni-1', project='main')

        # get the least-busy backend at IBM and run the quantum circuit there
        from qiskit.providers.ibmq import least_busy
        from qiskit.tools.monitor import job_monitor

        backend = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 6 and
                                                                 not b.configuration().simulator and b.status().operational == True))
        t_qc = transpile(qc, backend, optimization_level=1)
        job = backend.run(t_qc)
        job_monitor(job)  # displays job status under cell

        exp_res = job.result()
        sv = exp_res.get_statevector(qc)
        return state_fidelity(tfd_ref, sv)
    else:       # framework='sim'
        backend = Aer.get_backend('aer_simulator')
        qc.save_statevector()
        shots = 20000
        job = execute(qc, backend, shots=shots)

        # Get state_vector of the system:
        st = job.result().get_statevector(qc)

        # Return state_fidelity with [1] + [0]*127.
        return state_fidelity(st, tfd_ref)


"""
def finite_temp(temp):
    # angle = f(temp)
    qreg = QuantumRegister(6)
    creg = ClassicalRegister(6)
    qc = QuantumCircuit(qreg, creg)
    # Prepare system in tfd state.
    qc.rxx(angle, qreg[0], qreg[3])
    qc.rxx(angle, qreg[1], qreg[4])
    qc.rxx(angle, qreg[2], qreg[5])
    qc.barrier()
    for n in range(6):
        # Need to replace this with e^(-i(theta/2)pz_n)
        qc.z(qreg[n])
    qc.rxx(angle, qreg[0], qreg[1])
    qc.rxx(angle, qreg[1], qreg[2])
    qc.rxx(angle, qreg[3], qreg[4])
    qc.rxx(angle, qreg[4], qreg[5])
    qc.barrier()
    qc.rzz(angle, qreg[0], qreg[3])
    qc.rzz(angle, qreg[1], qreg[4])
    qc.rzz(angle, qreg[2], qreg[5])
    qc.barrier()
    # Prepare a tfd state in the with the proper temp.
    tfd_ref = tfd_generator().generate_tfd(1/temp)  # Beta=1/T.

    # Load account settings:
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-research-2', group='ben-gurion-uni-1', project='main')

    # get the least-busy backend at IBM and run the quantum circuit there
    from qiskit.providers.ibmq import least_busy
    from qiskit.tools.monitor import job_monitor

    backend = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 6 and
                                                             not b.configuration().simulator and b.status().operational == True))
    t_qc = transpile(qc, backend, optimization_level=1)
    job = backend.run(t_qc)
    job_monitor(job)  # displays job status under cell

    exp_res = job.result()
    sv = exp_res.get_statevector(qc)
    return state_fidelity(tfd_ref, sv)
"""


class tfd_gen:

    def __init__(self, temp=None):
        self.temp = temp


    def run(self, framework, temp=None):
        """
        Run a circuit in a real IBM quantum computer or a simulator (depicted by 'framework field'), initializing the state with a tfd state of infinite temperature
        at default, unless stated otherwise in 'temp' var.
        :param temp: The temperature of the tfd state, if a finite temperature tfd state is wanted.
        :return: The statistics of the execution of the circuit.
        """
        if temp == 0:
            return zero_temp(framework)
        elif temp is None:
            return infinite_temp(framework)
        else:
            self.temp = temp
            # finite_temp(framework, temp)
            print("TBD - Finite temp.")

"""
g = tfd_gen()
fid = {}
for temp in [None, 0]:
    fid[temp] = g.run('sim', temp)
print('Fidelity with tfd generated via mathematica:\n** {temp: fid}\n** None=infty\n', fid)
"""

# let's consider a circuit with one layer
d = 2
# the numbe of angles is twice the depth of the circuit
angles = random_angles(4*d)
# we build the QAOA circuit
qaoa = QAOA_TFD(3, d, angles)
# the reference state is
tfd_ref = tfd_generator().generate_tfd(1)
# for the optimization we use minimize with the following args
#optimizer = lambda f,theta: minimize(f, theta, method='Powell',options={'return_all':True}, tol=1e-3)
optimizer = lambda f, theta: minimize(f, theta, method='COBYLA', tol=1e-2)
# finally we can define the optimization algorithm
cooker = StateCooker(qaoa, tfd_ref, 'aer_simulator', optimizer)
result = cooker.optimize_1_layer()
print(result)
print(cooker.run_1_layer([0, 0, 0, 0]))
"""
# let's consider a circuit with two layer
d = 2
angles = np.append(result.x, random_angles(4*d))
# we build the QAOA circuit
qaoa = QAOA_TFD(3, d, angles)
cooker = StateCooker(qaoa, tfd_ref, 'aer_simulator', optimizer)
result = cooker.optimize()
print(result)
"""




