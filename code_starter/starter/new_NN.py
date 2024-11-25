import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import ast  # For safely evaluating string representations of lists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data, subject_meta, question_meta, C_Q_normalized)
        WHERE:
        zero_train_matrix: 2D FloatTensor where missing entries are filled with 0.
        train_data: 2D FloatTensor
        valid_data: A dictionary {user_id: list, question_id: list, is_correct: list}
        test_data: A dictionary {user_id: list, question_id: list, is_correct: list}
        subject_meta: DataFrame containing subject metadata
        question_meta: DataFrame containing question metadata
        C_Q_normalized: 2D NumPy array representing C_Q
    """
    # Load train, validation, and test data
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    # Fill in the missing entries with 0.
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0

    # Convert to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    # Load subject metadata
    subject_meta = pd.read_csv(f'{base_path}/subject_meta.csv')

    # Load question metadata
    question_meta = pd.read_csv(f'{base_path}/question_meta.csv')

    # Parse the 'subjects' column to ensure it's a list of integers
    def parse_subjects(subjects_str):
        """
        Safely parse the subjects list from a string.
        Handles formats like "[1,2]" or "1,2".
        """
        try:
            # Attempt to parse as a list
            subjects = ast.literal_eval(subjects_str)
            if isinstance(subjects, list):
                return subjects
        except:
            pass
        # If not a list, split by comma and convert to integers
        return [int(s.strip()) for s in subjects_str.split(',') if s.strip().isdigit()]

    question_meta['subjects'] = question_meta['subject_id'].apply(parse_subjects)

    # Construct the Question-Subject Assignment Matrix A
    unique_subjects = subject_meta['subject_id'].unique()
    unique_subjects_sorted = np.sort(unique_subjects)
    num_questions = question_meta.shape[0]
    num_subjects = unique_subjects_sorted.shape[0]

    # Initialize the assignment matrix with zeros
    A = np.zeros((num_questions, num_subjects), dtype=int)

    # Populate the assignment matrix
    for idx, row in question_meta.iterrows():
        question_idx = idx  # Assuming question_meta is indexed from 0 to Q-1
        subjects = row['subjects']
        for s in subjects:
            # Find the column index for subject s
            if s in unique_subjects_sorted:
                subject_col = np.where(unique_subjects_sorted == s)[0][0]
                A[question_idx, subject_col] = 1

    # Compute the Subject Correlation Matrix C_S using cosine similarity on TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(subject_meta['name'])
    C_S = cosine_similarity(tfidf_matrix, tfidf_matrix)  # Shape: [S, S]

    # Compute the Question Correlation Matrix C_Q = A * C_S * A^T
    C_Q = np.dot(np.dot(A, C_S), A.T)  # Shape: [Q, Q]

    # Normalize C_Q to ensure similarity scores are between 0 and 1
    sum_subjects = A.sum(axis=1).reshape(-1, 1)  # Shape: [Q, 1]
    # To avoid division by zero, replace zeros with ones
    sum_subjects[sum_subjects == 0] = 1
    normalization_matrix = np.sqrt(np.dot(sum_subjects, sum_subjects.T))  # Shape: [Q, Q]
    C_Q_normalized = C_Q / normalization_matrix
    # Replace any NaN values resulting from division by zero with zero
    C_Q_normalized = np.nan_to_num(C_Q_normalized)

    return zero_train_matrix, train_matrix, valid_data, test_data, subject_meta, question_meta, C_Q_normalized


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
        inner = self.g(inputs)
        sinner = torch.sigmoid(inner)
        outer = self.h(sinner)
        souter = torch.sigmoid(outer)
        return souter


def train_s(model, lr, lamb, train_data, zero_train_data, valid_data, C_Q, num_epoch):
    """Train the neural network with regularization and record metrics.

    :param model: Module
    :param lr: float, learning rate
    :param lamb: float, regularization strength
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param C_Q: 2D NumPy array, Question Correlation Matrix
    :param num_epoch: int, number of epochs
    :return: tuple of (training_losses, validation_losses, training_accuracies, validation_accuracies)
    """
    # Define optimizer and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_student = train_data.shape[0]
    num_questions = train_data.shape[1]

    # Convert C_Q to a torch tensor
    C_Q_tensor = torch.FloatTensor(C_Q)  # Shape: [Q, Q]

    # Lists to store metrics
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for user_id in range(num_student):
            inputs = zero_train_data[user_id].unsqueeze(0)  # Shape: [1, Q]
            target = train_data[user_id].unsqueeze(0).clone()  # Shape: [1, Q]

            optimizer.zero_grad()
            output = model(inputs)  # Shape: [1, Q]

            # Create mask for NaN entries
            nan_mask = torch.isnan(train_data[user_id])
            target[0][nan_mask] = output[0][nan_mask]

            # Compute reconstruction loss
            recon_loss = criterion(output, target)

            # Compute regularization term based on C_Q and decoder weights
            # h.weight shape: [Q, k]
            # We want to compute Tr(W * C_Q * W^T), encouraging similar questions to have similar weights
            decoder_weights = model.h.weight  # Shape: [Q, k]
            reg_term = torch.trace(torch.matmul(torch.matmul(decoder_weights.t(), C_Q_tensor), decoder_weights))
            reg_loss = lamb * reg_term

            # Total loss
            loss = recon_loss + reg_loss
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

            # Calculate training accuracy for valid entries
            predicted = (output >= 0.5).float().squeeze(0)  # Shape: [Q]
            valid_entries = ~nan_mask  # Shape: [Q]
            target_squeezed = target.squeeze(0)  # Shape: [Q]

            correct_preds += torch.sum(predicted[valid_entries] == target_squeezed[valid_entries]).item()
            total_preds += torch.sum(valid_entries).item()

        # Compute training accuracy for this epoch
        train_acc = correct_preds / total_preds if total_preds > 0 else 0.0

        # Compute validation accuracy and loss
        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_loss = 0.0
        for i, u in enumerate(valid_data["user_id"]):
            inputs = zero_train_data[u].unsqueeze(0)
            target_val = torch.FloatTensor([valid_data["is_correct"][i]])
            output_val = model(inputs)
            valid_loss += torch.sum((output_val[0][valid_data["question_id"][i]] - target_val) ** 2.0).item()

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
    """Evaluate the model on the validation data.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: Dict
    :return: float, accuracy
    """
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for i, u in enumerate(valid_data["user_id"]):
            inputs = train_data[u].unsqueeze(0)
            output = model(inputs)

            guess = (output[0][valid_data["question_id"][i]] >= 0.5).float().item()
            if guess == valid_data["is_correct"][i]:
                correct += 1
            total += 1

    return correct / float(total) if total > 0 else 0.0


def visualize_question_correlation(question_correlation_df, num_visualize=20):
    """
    Visualize a subset of the Question Correlation Matrix using a heatmap.

    :param question_correlation_df: DataFrame, Question Correlation Matrix
    :param num_visualize: int, number of questions to visualize
    :return: None
    """
    # Select a subset of questions
    subset_questions = question_correlation_df.index[:num_visualize]
    subset_correlation = question_correlation_df.loc[subset_questions, subset_questions]

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap
    sns.heatmap(
        subset_correlation,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    plt.title(f'Question Correlation Matrix Heatmap (Subset of {num_visualize} Questions)')
    plt.xlabel('Question ID')
    plt.ylabel('Question ID')
    plt.show()


def visualize_clustered_question_correlation(question_correlation_df, num_visualize=50):
    """
    Visualize the clustered Question Correlation Matrix using a clustermap.

    :param question_correlation_df: DataFrame, Question Correlation Matrix
    :param num_visualize: int, number of questions to visualize
    :return: None
    """
    # Select a subset of questions
    subset_questions = question_correlation_df.index[:num_visualize]
    subset_correlation = question_correlation_df.loc[subset_questions, subset_questions]

    # Generate a clustermap
    sns.clustermap(
        subset_correlation,
        method='average',
        cmap='viridis',
        figsize=(12, 10),
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    plt.title(f'Clustered Question Correlation Matrix Heatmap (Subset of {num_visualize} Questions)')
    plt.show()


def main():
    # Load all data including correlation matrices
    zero_train_matrix, train_matrix, valid_data, test_data, subject_meta, question_meta, C_Q_normalized = load_data()

    # Optional: Visualize a subset of the Question Correlation Matrix
    # Convert C_Q_normalized to DataFrame for easy indexing
    question_ids = question_meta['question_id'].tolist()
    question_correlation_df = pd.DataFrame(
        C_Q_normalized,
        index=question_ids,
        columns=question_ids
    )

    # Visualize the correlation matrix (adjust num_visualize as needed)
    visualize_question_correlation(question_correlation_df, num_visualize=20)
    # Or use clustered heatmap
    # visualize_clustered_question_correlation(question_correlation_df, num_visualize=50)

    #####################################################################
    # Try out different k and select the best k using the validation set.
    #####################################################################
    # Define the list of k values to experiment with
    k_values = [50]  # You can adjust this list based on your requirements
    num_questions = zero_train_matrix.shape[1]

    # Set optimization hyperparameters
    lr = 0.001
    num_epoch = 80
    lamb = 0.001

    # Initialize variables to keep track of the best hyperparameters
    best_k = 0
    best_lamb = 0
    best_valid_acc = 0

    # Iterate over different k values to find the best one
    for k in k_values:
        print(f"\n--- Training AutoEncoder with k={k} ---")
        # Initialize the model
        model = AutoEncoder(num_question=num_questions, k=k)

        # Train the model and record metrics
        training_losses, validation_losses, training_accuracies, validation_accuracies = train_s(
            model, lr, lamb, train_matrix, zero_train_matrix, valid_data, C_Q_normalized, num_epoch
        )

        # Evaluate the model on validation data
        valid_acc = evaluate(model, zero_train_matrix, valid_data)
        print(f"Validation Accuracy for k={k}, lambda={lamb}: {valid_acc:.4f}")

        # Update the best hyperparameters if current model is better
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_k = k
            best_lamb = lamb

    # Retrain the best model and record metrics for plotting
    print(
        f"\nRetraining the best model with k*={best_k} and lambda*={best_lamb} for plotting metrics..."
    )

    # Initialize the best model
    best_model = AutoEncoder(num_question=num_questions, k=best_k)

    # Train the best model and record metrics
    training_losses, validation_losses, training_accuracies, validation_accuracies = train_s(
        best_model, lr, best_lamb, train_matrix, zero_train_matrix, valid_data, C_Q_normalized, num_epoch
    )

    # Plot training loss, validation loss, training accuracy, and validation accuracy over epochs
    epochs = range(1, num_epoch + 1)

    plt.figure(figsize=(18, 10))

    # Plot Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Validation Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, validation_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Training Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, training_accuracies, label='Training Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Validation Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy', color='green')
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

if __name__ == "__main__":
    main()
