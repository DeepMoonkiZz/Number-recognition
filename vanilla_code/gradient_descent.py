from build_weight_and_bias import *
from load_save_weight_and_bias import *

def gradient_descent(X, Y, alpha, iterations, type = 'init'):
    W1, b1, W2, b2 = init_params(type)
    for i in range(iterations):        
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if int(i+1) % 10 == 0 or i == 0:
            print("Iteration: ", i+1)
            """
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
            """
    return W1, b1, W2, b2