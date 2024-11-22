import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torch

from neural_network import AutoEncoder, train
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
    """Return a list of bootstrap indices.

    :param data: 2D FloatTensor
    :param num_bootstrap: int
    :return: List
    """
    num_data = data.shape[0]
    bootstrap_indices = []
    for _ in range(num_bootstrap):
        indices = np.random.choice(num_data, num_data)
        bootstrap_indices.append(indices)
    return bootstrap_indices




def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
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
            for user_id in range(data.shape[0]):
                inputs = data[user_id].unsqueeze(0)
                output = model(inputs)
                aggregated_output[user_id] += output.squeeze(0)
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

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    k = 50
    num_questions = zero_train_matrix.shape[1]
    lr = 0.01
    num_epoch = 50
    lamb = 0.01
    num_bootstrap = 3
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
