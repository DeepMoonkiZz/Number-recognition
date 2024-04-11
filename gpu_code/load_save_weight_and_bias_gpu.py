import cupy as cp


def init_params(type='init'):
    if type == 'init':
        W1 = cp.asarray(cp.random.rand(10, 784) - 0.5)
        b1 = cp.asarray(cp.random.rand(10, 1) - 0.5)
        W2 = cp.asarray(cp.random.rand(10, 10) - 0.5)
        b2 = cp.asarray(cp.random.rand(10, 1) - 0.5)
    else:
        W1, b1, W2, b2 = load_params()
    return W1, b1, W2, b2

def load_params():
    W1 = cp.genfromtxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias_gpu/W1.csv", delimiter=",")
    b1 = cp.genfromtxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias_gpu/b1.csv", delimiter=",")
    W2 = cp.genfromtxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias_gpu/W2.csv", delimiter=",")
    b2 = cp.genfromtxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias_gpu/b2.csv", delimiter=",")
    
    W1 = cp.asarray(W1)
    b1 = cp.asarray(b1.reshape(-1, 1))
    W2 = cp.asarray(W2)
    b2 = cp.asarray(b2.reshape(-1, 1))
    
    return W1, b1, W2, b2

def save_params(W1, b1, W2, b2):
    W1_cpu = W1.get()
    b1_cpu = b1.get()
    W2_cpu = W2.get()
    b2_cpu = b2.get()

    cp.savetxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias_gpu/W1.csv", W1_cpu, delimiter=",")
    cp.savetxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias_gpu/b1.csv", b1_cpu, delimiter=",")
    cp.savetxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias_gpu/W2.csv", W2_cpu, delimiter=",")
    cp.savetxt("/home/segal/Documents/Projets_perso/Number-recognition/weight_and_bias_gpu/b2.csv", b2_cpu, delimiter=",")
