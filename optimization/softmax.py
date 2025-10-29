import numpy as np

"""
Compute the softmax cross-entropy loss in a numerically stable way.

Softmax cross-entropy loss is a common loss function used for multi-class classification problems. It's the combination of:

* Softmax function — converts raw logits (model outputs) into probabilities.
* Cross-entropy loss — measures the difference between the predicted probabilities and the true labels

"""

# softmax function
def softmax(x):
    x = np.array(x) 
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum(axis = 0)

x = [5,6,7]
softmax(x)


"""
When computing softmax cross-entropy, numerical instability can occur due to large exponentials in the softmax function. 
To avoid overflow/underflow, we use a numerically stable version by shifting logits.
"""
def softmax_loss_function(X, y):
    
    # Number of samples
    N = X.shape[0]

    # Step 1: shifts logits for numerical stability
    z = X - np.max(X, axis=1, keepdims=True)

    # Step 2: Exponetiate
    exp_z = np.exp(z)

    # Step 3: Normalize to get softmax probabilities
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Step 4: compute the negative log likelihood
    log_probs = -np.log(probs[np.arange(N), y])

    # Step 5: Avg over all samples
    loss = np.mean(log_probs)

    return loss

X = np.array([[3.5, 5.2, 2.3], [1.2, 3.4, 1.0], [5.3, 6.0, 4.2],  [4.3, 5.0, 2.2]])
y = np.array([0, 1, 1, 2])

output = softmax_loss_function(X, y)
print(output)

