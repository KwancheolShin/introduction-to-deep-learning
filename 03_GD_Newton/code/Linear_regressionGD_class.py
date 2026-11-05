import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionGD:

    def __init__(self, learning_rate=0.001, epochs=20000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Parameters
        self.w = 0.0
        self.b = 0.0

        self.cost_history = []


    # --------------------------------------------------
    # Cost function
    # --------------------------------------------------
    def compute_cost(self, x, y):
        y_pred = self.w * x + self.b
        cost = np.mean((y - y_pred)**2)

        return cost


    # --------------------------------------------------
    # Numerical gradient
    # --------------------------------------------------
    def numerical_gradient(self, x, y, h=1e-5):

        # ----- derivative with respect to w -----
        w_old = self.w

        self.w = w_old + h
        cost_plus = self.compute_cost(x, y)

        self.w = w_old - h
        cost_minus = self.compute_cost(x, y)

        dw = (cost_plus - cost_minus) / (2*h)

        # Restore w
        self.w = w_old


        # ----- derivative with respect to b -----
        b_old = self.b

        self.b = b_old + h
        cost_plus = self.compute_cost(x, y)

        self.b = b_old - h
        cost_minus = self.compute_cost(x, y)

        db = (cost_plus - cost_minus) / (2*h)

        # Restore b
        self.b = b_old

        return dw, db


    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def fit(self, x, y, verbose=True):

        for epoch in range(self.epochs):

            # Numerical gradient
            dw, db = self.numerical_gradient(x, y)

            # Gradient Descent
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Cost logging
            cost = self.compute_cost(x, y)
            self.cost_history.append(cost)

            if verbose and epoch % 1000 == 0:
                print(
                    f"Epoch {epoch}: "
                    f"Cost={cost:.6f}, "
                    f"w={self.w:.4f}, "
                    f"b={self.b:.4f}"
                )


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict(self, x):
        return self.w * x + self.b


    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    def plot_fit_and_cost(self, x, y):

        y_fit = self.predict(x)

        plt.figure(figsize=(10, 4))

        # Regression result
        plt.subplot(1, 2, 1)

        plt.scatter(x, y, label='Data')
        plt.plot(x, y_fit, color='red',
                 label='Regression line')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression Fit')
        plt.legend()
        plt.grid(True)


        # Cost history
        plt.subplot(1, 2, 2)

        plt.plot(self.cost_history,
                 label='Cost (MSE)')

        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# ======================================================
# Data
# ======================================================

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 2.8, 3.6, 4.5, 5.1])


# ======================================================
# Model
# ======================================================

model = LinearRegressionGD(
    learning_rate=0.01,
    epochs=20000
)


# ======================================================
# Training
# ======================================================

model.fit(x, y)


# ======================================================
# Result
# ======================================================

print("\nFinal result")
print(f"w = {model.w:.4f}")
print(f"b = {model.b:.4f}")


# ======================================================
# Plot
# ======================================================

model.plot_fit_and_cost(x, y)