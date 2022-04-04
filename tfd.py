from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex
from qiskit.extensions import Initialize
from qiskit.quantum_info import random_statevector
import numpy as np
from whole_process import tfd_generator
from qiskit.quantum_info import random_statevector, state_fidelity, Statevector
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute
from qiskit.quantum_info import random_statevector, state_fidelity, Statevector
from qiskit.extensions import Initialize, UnitaryGate

def infinite_temp(framework):
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


def zero_temp(framework):
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


g = tfd_gen()
fid = {}
for temp in [None, 0]:
    fid[temp] = g.run('sim', temp)
print('Fidelity with tfd generated via mathematica:\n** {temp: fid}\n** None=infty\n', fid)

