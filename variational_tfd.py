import numpy as np
import math
from matplotlib import pyplot as plt
from random import random, sample
from scipy.optimize import minimize

from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import state_fidelity

# generate random angles, needed to initialize the problem
def random_angles(n):
    """
    # Old implementation:
    angles = []
    for i in range(n):
        angles.append(2*math.pi*random())
    return angles
    """
    return list(np.random.uniform(0, 2*np.pi, n))

class ThetaError(Exception):
    """Error raised whenever the angles provided to the variational circuit
    don't match with the circuit shape.
    """

class VariationalCircuit(object):
    """Implement a generic variational circuit that can then be used with a
    variational optimization algorithm.
    """

    def _circuitBuilder(self, theta):
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

        # 8 thetas (corresponds to 2 layers - d=2), first 4 are initialized to be what returns from the 1-layer
        # optimization.
        self.last_thetas = [3.00432495, 6.28315478, 4.70740837, 6.32998926, 0, 0, 0, 0]

    def run(self, theta=[None]):
        """Runs the circuit and return 1-fidelity between the output state
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
