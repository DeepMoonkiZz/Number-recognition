import numpy as np

def init_params(type = 'init'):
    if type == 'init':
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
    else:
        W1, b1, W2, b2 = load_params()
    return W1, b1, W2, b2

def load_params():
    W1 = np.genfromtxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias/W1.csv", delimiter=",")
    b1 = np.genfromtxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias/b1.csv", delimiter=",")
    W2 = np.genfromtxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias/W2.csv", delimiter=",")
    b2 = np.genfromtxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias/b2.csv", delimiter=",")
    return W1, b1.reshape(-1, 1), W2, b2.reshape(-1, 1)

def save_params(W1, b1, W2, b2):
    np.savetxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias/W1.csv", W1, delimiter=",") 
    np.savetxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias/b1.csv", b1, delimiter=",") 
    np.savetxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias/W2.csv", W2, delimiter=",") 
    np.savetxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias/b2.csv", b2, delimiter=",")