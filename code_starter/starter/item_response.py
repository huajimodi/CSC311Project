from nltk import log_likelihood

from code_starter.starter.neural_network import train
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.0
    for i in range(len(data["is_correct"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]
        x = theta[u] - beta[q]
        log_lklihood += c * np.log(sigmoid(x) + 1e-9) + (1 - c) * np.log(1 - sigmoid(x) + 1e-9)
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    d_theta = np.zeros_like(theta)
    d_beta = np.zeros_like(beta)

    for i in range(len(data["is_correct"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]
        x = theta[u] - beta[q]
        p = sigmoid(x)

        d_theta[u] += c - p
        d_beta[q] += p - c

    theta += lr * d_theta
    beta -= lr * d_beta

    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    num_users = max(data["user_id"]) + 1
    num_questions = max(data["question_id"]) + 1
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    val_acc_lst = []
    val_llds = []
    train_llds = []
    for i in range(iterations):
        neg_lld_val = neg_log_likelihood(data, theta=theta, beta=beta)
        train_llds.append(neg_lld_val)

        neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_llds.append(neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        theta, beta = update_theta_beta(data, lr, theta, beta)

    log_likelihoods = {"train": train_llds, "val": val_llds}
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, log_likelihoods


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")


    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    iteration_counts = [10, 50, 100]

    best_lr = None
    best_iterations = None
    best_val_acc = None
    best_theta = None
    best_beta = None
    for lr in learning_rates:
        for iteration in iteration_counts:
            theta, beta, val_acc_lst, log_likelihoods = irt(train_data, val_data, lr, iteration)
            val_acc = evaluate(val_data, theta, beta)
            test_acc = evaluate(test_data, theta, beta)
            print(f"Learning Rate: {lr} Iteration: {iteration} Val Acc: {val_acc} Test Acc: {test_acc}")
            if best_val_acc is None or val_acc > best_val_acc:
                best_lr = lr
                best_iterations = iteration
                best_val_acc = val_acc
                best_theta = theta
                best_beta = beta
    print("\nBest Hyperparameters:")
    print(f"Learning Rate: {best_lr}")
    print(f"Iterations: {best_iterations}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
