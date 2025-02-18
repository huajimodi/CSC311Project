import numpy as np
import pandas as pd
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
from sklearn.cluster import KMeans
import ast  # For safely evaluating string representations of lists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def question_correlation_matrix(A, C_S):
    """Compute the Question Correlation Matrix C_Q using the Assignment Matrix A and Subject Correlation Matrix C_S.

    :param A: 2D NumPy array representing the Question-Subject Assignment Matrix
    :param C_S: 2D NumPy array representing the Subject Correlation Matrix
    :return: 2D NumPy array representing the Question Correlation Matrix
    """
    # Compute the Question Correlation Matrix C_Q = A * C_S * A^T
    C_Q = np.dot(np.dot(A, C_S), A.T)  # Shape: [Q, Q]

    # Normalize C_Q to ensure similarity scores are between 0 and 1
    # Compute the diagonal elements of C_Q
    C_Q_diag = np.diag(C_Q)  # Shape: [Q]

    # Compute normalization matrix
    normalization_matrix = np.sqrt(
        np.outer(C_Q_diag, C_Q_diag))  # Shape: [Q, Q]

    # Avoid division by zero by replacing zeros with a small epsilon
    epsilon = 1e-8
    normalization_matrix[normalization_matrix == 0] = epsilon

    # Normalize C_Q
    C_Q_normalized = C_Q / normalization_matrix

    # Replace any NaN values resulting from division by zero with zero
    C_Q_normalized = np.nan_to_num(C_Q_normalized)

    # Set diagonal entries to 1
    np.fill_diagonal(C_Q_normalized, 1.0)

    # Clip values to [0, 1]
    C_Q_normalized = np.clip(C_Q_normalized, 0, 1)

    return C_Q_normalized


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
    return [int(s.strip()) for s in subjects_str.split(',') if
            s.strip().isdigit()]


def get_correlation_matrix(base_path="./data"):
    # Load subject metadata
    subject_meta = pd.read_csv(f'{base_path}/subject_meta.csv')

    # Load question metadata
    question_meta = pd.read_csv(f'{base_path}/question_meta.csv')
    question_meta['subjects'] = question_meta['subject_id'].apply(
        parse_subjects)

    # Construct the Question-Subject Assignment Matrix A
    unique_subjects = subject_meta['subject_id'].unique()
    unique_subjects_sorted = np.sort(unique_subjects)
    num_questions = question_meta.shape[0]
    num_subjects = unique_subjects_sorted.shape[0]

    # Initialize the assignment matrix with zeros
    A = np.zeros((num_questions, num_subjects), dtype=int)

    # Populate the assignment matrix
    for idx, row in question_meta.iterrows():
        question_idx = idx
        subjects = row['subjects']
        for s in subjects:
            # Find the column index for subject s
            if s in unique_subjects_sorted:
                subject_col = np.where(unique_subjects_sorted == s)[0][0]
                A[question_idx, subject_col] = 1

    # Compute the Subject Correlation Matrix C_S using cosine similarity on TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(subject_meta['name'])
    C_S = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return question_correlation_matrix(A, C_S)


def load_data(base_path="./data", k_mean=14):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data, subject_meta, question_meta, C_Q_normalized)
        WHERE:
        zero_train_matrix: 2D FloatTensor where missing entries are filled based on correlation.
        train_data: 2D FloatTensor
        valid_data: A dictionary {user_id: list, question_id: list, is_correct: list}
        test_data: A dictionary {user_id: list, question_id: list, is_correct: list}
        question_meta: DataFrame containing question metadata
        C_Q_normalized: 2D NumPy array representing the normalized Question Correlation Matrix
    """
    # Load train, validation, and test data
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    question_meta = pd.read_csv(f'{base_path}/question_meta.csv')
    C_Q_normalized = get_correlation_matrix(base_path)

    # Fill in missing entries in the train_matrix based on correlation
    zero_train_matrix = train_matrix.copy()
    num_students, num_questions = zero_train_matrix.shape
    zero_train_matrix[zero_train_matrix == 0] = -1

    # Cluster questions based on the correlation matrix
    num_clusters = k_mean
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(C_Q_normalized)

    for student_id in range(num_students):
        for question_id in range(num_questions):
            if np.isnan(zero_train_matrix[student_id, question_id]):
                # Find other questions in the same cluster
                cluster = clusters[question_id]
                cluster_questions = np.where(clusters == cluster)[0]

                # Get valid answers from the student in the cluster
                valid_answers = [zero_train_matrix[student_id, q] for q in cluster_questions if
                                 not np.isnan(zero_train_matrix[student_id, q])]

                if len(valid_answers) > 1:
                    # Use majority voting
                    zero_train_matrix[student_id, question_id] = 1 if np.mean(valid_answers) > 0.5 else -1
                else:
                    zero_train_matrix[student_id, question_id] = 0

    # Convert to FloatTensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data, question_meta, C_Q_normalized


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


def train(
        model, lr, lamb, train_data, zero_train_data, valid_data, C_Q,
        num_epoch
):
    """Train the neural network with regularization and record metrics.

    :param model: Module
    :param lr: float, learning rate
    :param lamb: float, regularization parameter
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param C_Q: 2D NumPy array, Question Correlation Matrix
    :param num_epoch: int, number of epochs
    :return: tuple of (training_losses, validation_losses, training_accuracies, validation_accuracies)
    """
    # Ensure C_Q matches the number of questions
    num_questions = train_data.shape[1]
    C_Q = C_Q[:num_questions, :num_questions]
    C_Q_tensor = torch.FloatTensor(C_Q)

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=lr)

    num_students = train_data.shape[0]

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

        for user_id in range(num_students):
            inputs = zero_train_data[user_id].unsqueeze(0)
            target = train_data[user_id].unsqueeze(0).clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Create mask for NaN entries
            nan_mask = torch.isnan(train_data[user_id])
            target[0][nan_mask] = output[0][nan_mask]

            # Compute regularization term
            decoder_weights = model.h.weight
            reg_term = torch.trace(
                torch.matmul(
                    torch.matmul(decoder_weights.t(), C_Q_tensor),
                    decoder_weights
                )
            ).clamp(min=0)

            # Compute reconstruction loss
            loss = (
                    torch.sum((output - target) ** 2.0)
                    + lamb / 2 * model.get_weight_norm()
            )
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # Calculate training accuracy for valid entries
            predicted = (output >= 0.5).float().squeeze(0)
            valid_entries = ~nan_mask
            target_squeezed = target.squeeze(0)

            correct_preds += torch.sum(
                predicted[valid_entries] == target_squeezed[valid_entries]
            ).item()
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
            valid_loss += torch.sum(
                (output_val[0][
                     valid_data["question_id"][i]] - target_val) ** 2.0
            ).item()

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

            guess = (output[0][
                         valid_data["question_id"][i]] >= 0.5).float().item()
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
    subset_correlation = question_correlation_df.loc[
        subset_questions, subset_questions]

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


def main():
    # Set optimization hyperparameters
    lr = 0.005
    num_epoch = 80
    lamb = 0.001
    k_mean = 14

    # Load all data including correlation matrices
    zero_train_matrix, train_matrix, valid_data, test_data, question_meta, C_Q_normalized = load_data(k_mean=k_mean)

    question_ids = question_meta['question_id'].tolist()
    question_correlation_df = pd.DataFrame(
        C_Q_normalized,
        index=question_ids,
        columns=question_ids
    )

    visualize_question_correlation(question_correlation_df, num_visualize=20)

    #####################################################################
    k = 50
    num_questions = zero_train_matrix.shape[1]

    print(f"\n--- Training AutoEncoder with k={k} ---")
    model = AutoEncoder(num_question=num_questions, k=k)

    # Train the model and record metrics
    training_losses, validation_losses, training_accuracies, validation_accuracies = (
        train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, C_Q_normalized, num_epoch)
    )

    # Evaluate the model on validation data
    valid_acc = evaluate(model, zero_train_matrix, valid_data)
    print(f"Validation Accuracy for k={k}, : {valid_acc:.4f}")

    #####################################################################
    epochs = range(1, num_epoch + 1)

    plt.figure(figsize=(18, 10))

    # Plot Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_accuracies, label='Training Acc', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Training Acc')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Validation Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, validation_accuracies, label='Validation Acc', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Acc')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate the best model on the test set
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(
        f"\nFinal Test Accuracy for the best model (k*={k}): {test_acc:.4f}"
    )


if __name__ == "__main__":
    main()
