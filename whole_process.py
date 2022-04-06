import numpy as np
import csv
from util import circ25, circ25_noMeasurements_forFidelity
from plots import generate_graphs, state_fid_plot

# The Pauli operators:
px = np.array([[0, 1], [1, 0]])
py = np.array([[0, -1j], [1j, 0]])
pz = np.array([[1, 0], [0, -1]])
# The id operator (dim=2)
id2 = np.eye(2)


class tfd_generator(object):

    def __init__(self):
        calc = self.calc_evals_evecs()
        self.evals0 = calc[0] - min(calc[0])                # Normalized to 0 as min.
        self.evecs = np.transpose(calc[1])

    def pzi(self, i):
        """
        Returns the kronecker product of which the pz operator operates on the i'th qubit
        (notice that the right-most qubit is q0!
        """
        if i == 0:
            return np.kron(id2, np.kron(id2, pz))
        elif i == 1:
            return np.kron(id2, np.kron(pz, id2))
        else:
            return np.kron(pz, np.kron(id2, id2))

    def pxi(self, i):
        """
        Returns the kronecker product of which the px operator operates on the i'th qubit
        (notice that the right-most qubit is q0!
        """
        if i == 0:
            return np.kron(id2, np.kron(id2, px))
        elif i == 1:
            return np.kron(id2, np.kron(px, id2))
        else:
            return np.kron(px, np.kron(id2, id2))


    def outpzipzip1(self, i):
        """
        Returns the kronecker product of which the pz operator operates on the i'th qubit and the i+1 qubit
        (notice that the right-most qubit is q0!
        Also, this is the periodic version (3 -> 1 also exists).
        """
        if i == 0:
            return np.kron(id2, np.kron(pz, pz))
        elif i == 1:
            return np.kron(pz, np.kron(pz, id2))
        else:
            return np.kron(pz, np.kron(id2, pz))


    def ising_ham(self, g, h):
        """ Calculates the Ising Hamiltonian of 3 qubits, given g and h"""
        s = 0
        for i in range(3):
            s -= self.outpzipzip1(i) + g * self.pxi(i) + h * self.pzi(i)
        return s

    def calc_evals_evecs(self):
        #print('ising ham: ', self.ising_ham(-1.05, 0.5), '\n')
        return np.linalg.eigh(self.ising_ham(-1.05, 0.5))

    def chop(self, expr, max=1*pow(10, -10)):
        return [i if i > max else 0 for i in expr]

    def generate_tfd(self, beta):
        self.beta = beta
        acc = [0] * pow(2, 6)
        for i in range(len(self.evecs)):
            acc += np.exp(-beta * self.evals0[i]/2) * np.ndarray.flatten(np.kron(self.evecs[i], self.evecs[i]))

        #print('acc: ', acc, '\n')
        #chopped = self.chop(acc)
        inner_prod = 0
        for i in range(pow(2, 3)):
            inner_prod += np.exp(-beta * self.evals0[i])
        tfd = acc / np.sqrt(inner_prod)

        #normalized_tfd = tfd / np.linalg.norm(tfd)
        return tfd


def run_exp(file_name, b0, bf, step, init_q0=None):
    """ Runs the experiment, for b in range of b0->bf, with step step"""
    with open(file_name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # write the header
        header = ['beta', 'bad_counts', 'probability_of_0000000']
        writer.writerow(header)
        gen = tfd_generator()
        for beta in np.arange(b0, bf, step):
            tfd = gen.generate_tfd(beta)
            data = circ25(tfd, beta, init_q0)
            writer.writerow(data)
    # f closes automatically here (due to 'with').


def run_exp_fid(file_name, b0, bf, step, init_q0 = None):
    """ Runs the experiment, for b in range of b0->bf, with step step"""
    with open(file_name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # write the header
        header = ['state_fid', 'beta']
        writer.writerow(header)
        gen = tfd_generator()
        for beta in np.arange(b0, bf, step):
            tfd = gen.generate_tfd(beta)
            data = circ25_noMeasurements_forFidelity(tfd, beta, init_q0)
            writer.writerow(data)
    # f closes automatically here (due to 'with').


# Below is SOME of the code with which I generated the data and plots of the project.

"""
# Fidelity experiment for the fixed initial state:
for eig_state in ['px0', 'px1', 'py0', 'py1', 'pz0', 'pz1']:
    file_name = eig_state + '_fid'
    run_exp_fid(file_name, 0, 1.25, 0.01, eig_state)
    state_fid_plot(file_name+'.csv')
"""

"""
# Fidelity experiment for the random initial state:
for i in range(20):
    file_name = 'random_init_fidelity'+str(i)
    run_exp_fid(file_name, 0, 1.25, 0.01)
    state_fid_plot(file_name+'.csv')
"""

"""
# Running the experiment with random initial states for q0, in order to later take the mean values.
for i in range(10):
    run_exp('random_run_' + str(i), 0, 1.5, 0.005)
    generate_graphs('random_run_' + str(i) + '.csv')
"""

"""
# The pauli operators eigenstates as initial states of q0:

b0 = 0
bf = 2.5
step = 0.01

run_exp('whole_25_pz0', b0, bf, step, 'pz0')
generate_graphs('whole_25_pz0.csv')

run_exp('whole_25_pz1', b0, bf, step, 'pz1')
generate_graphs('whole_25_pz1.csv')

run_exp('whole_25_px0', b0, bf, step, 'px0')
generate_graphs('whole_25_px0.csv')

run_exp('whole_25_px1', b0, bf, step, 'px1')
generate_graphs('whole_25_px1.csv')

run_exp('whole_25_py0', b0, bf, step, 'py0')
generate_graphs('whole_25_py0.csv')

run_exp('whole_25_py1', b0, bf, step, 'py1')
generate_graphs('whole_25_py1.csv')
"""


"""
# Getting the limit of the TFD state for large beta (T->0):
for beta in [100, 200, 300, 400, 500, 1000, 10000, 100000]:
    gen = tfd_generator()
    tfd = gen.generate_tfd(beta)
    data = circ25(tfd, beta, 'pz0')
    print(data)
"""






