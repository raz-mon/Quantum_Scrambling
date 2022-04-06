import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def bad_counts_plot(file_name):
    fd = pd.read_csv(file_name)
    ys = fd['bad_counts'].values
    xs = fd['beta'].values
    plt.scatter(xs, ys)
    plt.ylabel('unsuccessful teleportation counts')
    plt.xlabel('$\\beta$ $\left[\\frac{1}{J}\\right]$')
    #plt.show()

def prob_plot(file_name):
    fd = pd.read_csv(file_name)
    ys2 = fd['probability_of_0000000'].values
    xs2 = fd['beta'].values
    plt.scatter(xs2, ys2)
    plt.ylabel('successful measurement probability')
    plt.xlabel('$\\beta$ $\left[\\frac{1}{J}\\right]$')
    #plt.show()


def state_fid_plot(file_name):
    fd = pd.read_csv(file_name)
    ys = fd['state_fid'].values
    xs = fd['beta'].values
    plot1 = plt.figure(1)
    plt.scatter(xs, ys)
    plt.title('state fidelity with $\left|0000000\\right\\rangle$ vs. $\\beta$')
    plt.ylabel('fidelity')
    plt.xlabel('$\\beta$ $\left[\\frac{1}{J}\\right]$')
    plt.savefig(file_name[:len(file_name)-4])
    # plt.show()
    plt.close(1)

def generate_graphs(file_name):
    plot1 = plt.figure(1)
    bad_counts_plot(file_name)
    # plt.title(file_name[len(file_name)-7:len(file_name)-4] + ' unsuccessful teleportation vs. $\\beta$')    # For constant initial states.
    plt.title(file_name[:len(file_name) - 4] + ' unsuccessful teleportation vs. $\\beta$')    # For means (5, 10, 20)
    plt.savefig(file_name[:len(file_name)-4] + ' unsuccessful teleportation')

    plot2 = plt.figure(2)
    prob_plot(file_name)
    # plt.title(file_name[len(file_name)-7:len(file_name)-4] + ' Successful measurement probability vs. $\\beta$')    # For constant initial states.
    plt.title(file_name[:len(file_name) - 4] + ' Successful measurement probability vs. $\\beta$')    # For means (5, 10, 20)
    plt.savefig(file_name[:len(file_name)-4] + ' Successful measurement probability')
    
    # plt.show()
    plt.close('all')



# Below is SOME of the code I used to generate the plots of the project.
"""
for str_num in ['5', '10', '20']:
    # generate_graphs('mean_'+str_num+'.csv')
    state_fid_plot('mean_'+str_num+'_csv.csv')
"""

"""
for eig_state in ['px0', 'px1', 'py0', 'py1', 'pz0', 'pz1']:
    file_name = 'whole_25_'+eig_state+'.csv'
    generate_graphs(file_name)
"""


"""
# A plot of the successful measurement probability & the bounding function:
file_name = 'mean_20.csv'
fd = pd.read_csv(file_name)
ys2 = fd['probability_of_0000000'].values[:70]
xs2 = fd['beta'].values[:70]
plt.scatter(xs2, ys2)

# Another try of a square bound:
bounding_xs = np.arange(0, 0.35, 0.001)
bounding_func_2 = lambda x: -0.52*x**2 + 0.026*x + 0.24
plt.plot(bounding_xs, [bounding_func_2(x) for x in bounding_xs], color='orange')

plt.title('lower bound for successful measurement probability')
plt.ylabel('successful measurement probability')
plt.xlabel('$\\beta$ $\left[\\frac{1}{J}\\right]$')
plt.legend(['f(x)=$-0.52 \\cdot x^{2} + 0.025 \\cdot x + 0.24$', 'successful measurement probability'])
plt.show()
"""

"""
# Plot of the unsuccessful teleportation counts vs. beta & the bounding function:
file_name = 'mean_20.csv'
fd = pd.read_csv(file_name)
ys2 = fd['bad_counts'].values[:70]
xs2 = fd['beta'].values[:70]
plt.scatter(xs2, ys2)

# Another try of a square bound:
bounding_xs = np.arange(0, 0.35, 0.001)
bounding_func_2 = lambda x: 5000*x**2 + 175*x + 9
plt.plot(bounding_xs, [bounding_func_2(x) for x in bounding_xs], color='orange')

plt.title('bounding function for unsuccessful teleportaiton counts')
plt.ylabel('unsuccessful teleportation counts')
plt.xlabel('$\\beta$ $\left[\\frac{1}{J}\\right]$')
plt.legend(['f(x)=$5000 \\cdot x^{2} + 175 \\cdot x + 9$', 'unsuccessful teleportation counts'])
plt.show()
"""

















