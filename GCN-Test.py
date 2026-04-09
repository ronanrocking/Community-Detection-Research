import numpy as np

# --- 1. SETUP DATA ---
X = np.array([
    [0.9, 0.1], [1.0, 0.2], [0.8, 0.4], # AI Group
    [0.4, 0.8], [0.3, 1.0], [0.2, 0.9]  # HW Group
])

A = np.array([
    [0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]
])

y = np.array([0, 0, 0, 1, 1, 1])
y_onehot = np.zeros((6, 2))
y_onehot[np.arange(6), y] = 1

# --- 2. PRE-PROCESS ADJACENCY ---
I = np.eye(6)
A_tilde = A + I
D_tilde = np.diag(np.sum(A_tilde, axis=1))
D_inv_sqrt = np.linalg.inv(np.sqrt(D_tilde))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

# --- 3. INITIALIZE WEIGHTS ---
np.random.seed(42) 
W1 = np.random.randn(2, 4) 
W2 = np.random.randn(4, 2)

# --- 4. TRAINING HYPERPARAMETERS ---
epochs = 6
learning_rate = 0.1 # Bumped this up slightly for faster convergence

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

print(f"Starting Training...")

# --- 5. THE TRAINING LOOP ---
for epoch in range(epochs):
    # FORWARD PASS
    H1 = A_norm @ (X @ W1)
    H1_relu = np.maximum(0, H1)
    logits = A_norm @ (H1_relu @ W2)
    probabilities = softmax(logits)

    # CALCULATE LOSS
    loss = -np.mean(np.sum(y_onehot * np.log(probabilities + 1e-9), axis=1))

    # BACKPROPAGATION (Assigning Blame)
    d_logits = probabilities - y_onehot
    
    # Gradients for W2
    W2_grad = H1_relu.T @ (A_norm.T @ d_logits)
    
    # Backprop through ReLU and A_norm to get to W1
    d_H1_relu = A_norm.T @ d_logits @ W2.T
    d_H1 = d_H1_relu.copy()
    d_H1[H1 <= 0] = 0 
    
    # Gradients for W1
    W1_grad = X.T @ (A_norm.T @ d_H1)

    # UPDATE WEIGHTS (Gradient Descent)
    W1 -= learning_rate * W1_grad
    W2 -= learning_rate * W2_grad

    # Print progress every 50 epochs
    print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

print("\n--- Final Results After Training ---")
print("Final Probabilities [AI Group, HW Group]:")
print(np.round(probabilities, 3))

# Checking if prediction matches ground truth
predictions = np.argmax(probabilities, axis=1)
print(f"\nPredicted Labels: {predictions}")
print(f"Actual Labels:    {y}")