import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pyswarm import pso

def IS_PSO():
    X = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    y = np.array([0, 1, 0, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def fitness_function(selected_features):
        selected_indices = np.where(selected_features)[0]
        if len(selected_indices) == 0:
            return 0.0  
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)
        return accuracy_score(y_test, y_pred)

    num_features = X_train.shape[1]
    lb = np.zeros(num_features)
    ub = np.ones(num_features)

    def is_pso(num_iterations, num_particles):
        def objective_function(selected_features):
            return -fitness_function(selected_features)

        best_features, _ = pso(objective_function, lb, ub, swarmsize=num_particles, maxiter=num_iterations)

        return best_features

    num_iterations = 50
    num_particles = 20

    selected_features = is_pso(num_iterations, num_particles)

    selected_indices = np.where(selected_features)[0]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with selected features: {accuracy:.2f}")

