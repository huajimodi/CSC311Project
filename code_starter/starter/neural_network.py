import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch
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


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        inner = self.g(out)
        sinner = torch.sigmoid(inner)
        outer = self.h(sinner)
        souter = torch.sigmoid(outer)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return souter


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: tuple of (training_losses, validation_losses, training_accuracies, validation_accuracies)
    """
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    # Lists to store metrics
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for user_id in range(num_student):
            inputs = zero_train_data[user_id].unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0) + lamb / 2 * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

            # Calculate training accuracy for valid entries
            predicted = (output >= 0.5).float()
            valid_entries = ~torch.tensor(nan_mask)
            correct_preds += torch.sum(predicted[valid_entries] == target[valid_entries]).item()
            total_preds += torch.sum(valid_entries).item()

        # Compute training accuracy for this epoch
        train_acc = correct_preds / total_preds

        # Compute validation accuracy and loss
        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_loss = 0.0
        for i, u in enumerate(valid_data["user_id"]):
            inputs = zero_train_data[u].unsqueeze(0)
            target = torch.FloatTensor([valid_data["is_correct"][i]])
            output = model(inputs)
            valid_loss += torch.sum((output[0][valid_data["question_id"][i]] - target) ** 2.0).item()

        # Append metrics
        training_losses.append(train_loss)
        training_accuracies.append(train_acc)
        validation_losses.append(valid_loss)
        validation_accuracies.append(valid_acc)

        # Print metrics
        print(
            f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f}\t "
            f"Training Acc: {train_acc:.4f}\tValidation Loss: {valid_loss:.6f}\tValidation Acc: {valid_acc:.4f}"
        )

    return training_losses, validation_losses, training_accuracies, validation_accuracies


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
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


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = [10, 50, 100, 200, 500]
    num_questions = zero_train_matrix.shape[1]

    # Set optimization hyperparameters.
    lr = 0.005
    num_epoch = 80
    lamb = 0.001

    # Record best hyper
    best_k = 0
    best_lamb = 0
    best_valid_acc = 0
    best_model = None
    best_training_losses, best_validation_losses, best_training_accuracies, best_validation_accuracies = (
        None, None, None, None
    )

    for i in k:
        # Initialize the model
        model = AutoEncoder(num_questions, i)
        training_losses, validation_losses, training_accuracies, validation_accuracies = (
            train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        )

        # Next, evaluate your network on validation/test data
        valid_acc = evaluate(model, zero_train_matrix, valid_data)
        print(f"Validation Accuracy for k={i}, lambda={lamb}: {valid_acc:.4f}\n")

        # Update the best hyperparameters if current model is better
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_k = i
            best_lamb = lamb
            best_model = model
            best_training_losses = training_losses
            best_validation_losses = validation_losses
            best_training_accuracies = training_accuracies
            best_validation_accuracies = validation_accuracies

    print(
        f"\nThe best model has k*={best_k} and lambda*={best_lamb} for plotting metrics..."
    )

    # Plot training loss, validation loss, training accuracy, and validation accuracy over epochs
    epochs = range(1, num_epoch + 1)

    plt.figure(figsize=(18, 10))

    # Plot Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, best_training_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, best_validation_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Training Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, best_training_accuracies, label='Training Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Validation Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, best_validation_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Evaluate the best model on the test set
    test_acc = evaluate(best_model, zero_train_matrix, test_data)
    print(
        f"\nFinal Test Accuracy for the best model (k*={best_k}, lambda*={best_lamb}): {test_acc:.4f}"
    )
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
