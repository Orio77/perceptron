import numpy as np
from perceptron import Perceptron

class Trainer:

  def __init__(self, perceptron: Perceptron, n_epochs: int):
    self.perceptron = perceptron
    self.n_epochs = n_epochs
    ## TODO add learning rate

  def loss(self, A: np.array, y: np.ndarray): 
    """funkcja oblicza stratę między wyjściem sieci a ground truth"""

    sample_losses = -(y * np.log(A) + (1 - y) * np.log(1 - A))

    return np.mean(sample_losses)

  def backward(self, X: np.ndarray, A: np.array, y: np.ndarray):
    dw_sum = X @ (A-y).T

    return (dw_sum / X.shape[1])

  def update_weights(self):
    # TODO implement  
    return None

  def train(self, X: np.ndarray, y: np.ndarray): # wykonuje trenowanie w N epokach
    
    for epoch in range (self.n_epochs):
        A = self.perceptron.forward(X=X)

        L = self.loss(A=A, y=y)

        dw = self.backward(X=X, A=A, y=y)

        self.update_weights()

        self.predict(X=X, y=y)

    return None

  def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: # funkcja na wykonanie inferencji
    # TODO implement
    return None




if __name__ == "__main__":
  trainer = Trainer(perceptron=None, n_epochs=1)

  # Test 1: basic loss correctness
  A = np.array([0.9, 0.8, 0.1])
  y = np.array([1, 1, 0])
  loss_val = trainer.loss(A, y)
  expected = np.mean(-(y * np.log(A) + (1 - y) * np.log(1 - A)))
  print(f"Loss: {loss_val:.8f}, expected: {expected:.8f}")
  assert np.isclose(loss_val, expected), "Loss value differs from expected"
  print("Loss function numeric test passed")

  # Test 2: loss non-negativity and finiteness
  A2 = np.array([0.5, 0.5])
  y2 = np.array([0, 1])
  loss_val2 = trainer.loss(A2, y2)
  assert np.isfinite(loss_val2) and loss_val2 >= 0, "Loss should be finite and non-negative"
  print("Loss non-negativity test passed")

  # Test 3: backward single-sample case
  X_single = np.array([[1.0], [2.0]])  # shape (features=2, samples=1)
  A_single = np.array([0.7])
  y_single = np.array([1.0])
  res_single = trainer.backward(X=X_single, A=A_single, y=y_single)
  expected_single = (X_single @ (A_single - y_single).T) / X_single.shape[1]
  print("Backward single-sample result:", res_single)
  assert res_single.shape == (2,), "Backward gradient shape mismatch for single sample"
  assert np.allclose(res_single, expected_single), "Backward gradient value mismatch for single sample"
  print("Backward single-sample test passed")

  # Test 4: backward multi-sample case
  X_multi = np.array([[1.0, 1.0, 0.0],
                      [0.0, 1.0, 1.0]])  # shape (features=2, samples=3)
  A_multi = np.array([0.9, 0.2, 0.1])
  y_multi = np.array([1, 0, 0])
  res_multi = trainer.backward(X=X_multi, A=A_multi, y=y_multi)
  expected_multi = (X_multi @ (A_multi - y_multi).T) / X_multi.shape[1]
  print("Backward multi-sample result:", res_multi)
  assert res_multi.shape == (2,), "Backward gradient shape mismatch for multi-sample case"
  assert np.allclose(res_multi, expected_multi), "Backward gradient value mismatch for multi-sample case"
  print("Backward multi-sample test passed")

  print("All smoother Trainer tests passed")
  