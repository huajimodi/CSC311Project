import random

import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torch
import torch.optim as optim
import torch.nn as nn

from neural_network import AutoEncoder
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data

def bootstrap_data(data, num_bootstrap):
    num_data = data.shape[0]
    bootstrap_indices = []
    for _ in range(num_bootstrap):
        indices = np.random.choice(num_data, num_data, replace=True)
        bootstrap_indices.append(indices)
    return bootstrap_indices


def evaluate(model, train_data, valid_data):
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def aggregate_predictions(models, data):
    aggregated_output = torch.zeros(data.shape)
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(data)
            aggregated_output += output
    aggregated_output /= len(models)
    return aggregated_output


def evaluate_aggregated(aggregated_output, data):
    total = 0
    correct = 0
    for i, u in enumerate(data["user_id"]):
        guess = aggregated_output[u][data["question_id"][i]].item() >= 0.5
        if guess == data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):

        for user_id in range(num_student):
            inputs = zero_train_data[user_id].unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0)   + lamb * 0.5 * model.get_weight_norm()
            loss.backward()
            optimizer.step()



def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    k = 50
    num_questions = zero_train_matrix.shape[1]
    lr = 0.005
    num_epoch = 80
    lamb = 0.001
    num_bootstrap = 10
    bootstrap_indices = bootstrap_data(train_matrix, num_bootstrap)
    models = []

    for indices in bootstrap_indices:
        model = AutoEncoder(num_questions, k)
        bootstrap_train_data = train_matrix[indices]
        bootstrap_zero_train_data = zero_train_matrix[indices]
        train(model, lr, lamb, bootstrap_train_data, bootstrap_zero_train_data, valid_data, num_epoch)
        models.append(model)

    aggregated_output = aggregate_predictions(models, zero_train_matrix)

    valid_acc = evaluate_aggregated(aggregated_output, valid_data)
    test_acc = evaluate_aggregated(aggregated_output, test_data)

    print(f"Validation Accuracy: {valid_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
