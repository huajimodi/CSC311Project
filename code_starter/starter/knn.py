import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #

    k_values = [1, 6, 11, 16, 21, 26]
    valid_acc_user = []
    valid_acc_item = []

    for k in k_values:
        acc_user = knn_impute_by_user(sparse_matrix, val_data, k)
        acc_item = knn_impute_by_item(sparse_matrix, val_data, k)
        valid_acc_user.append(acc_user)
        valid_acc_item.append(acc_item)
        print(f"k={k}: User-based Acc={acc_user:.4f}, Item-based Acc={acc_item:.4f}")

    # Determine best k and strategy
    best_user_k = k_values[np.argmax(valid_acc_user)]
    best_item_k = k_values[np.argmax(valid_acc_item)]

    # Choose the best overall
    if max(valid_acc_user) >= max(valid_acc_item):
        best_k = best_user_k
        strategy = 'User-based'
    else:
        best_k = best_item_k
        strategy = 'Item-based'

    print(f"Best strategy: {strategy} with k={best_k}")

    # Evaluate on test set
    if strategy == 'User-based':
        test_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_k)
    else:
        test_accuracy = knn_impute_by_item(sparse_matrix, test_data, best_k)

    print(f"Test accuracy with {strategy} k={best_k}: {test_accuracy:.4f}")

    # # User based graph
    # plt.figure(figsize=(10, 6))
    # plt.plot(k_values, valid_acc_user, label='User-based Imputation')
    # plt.xlabel('Number of Neighbors (k)')
    # plt.ylabel('Validation Accuracy')
    # plt.title('k-NN Imputation Performance')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # plt.savefig('knn_imputation.png')

    # # Item based graph
    # plt.figure(figsize=(10, 6))
    # plt.plot(k_values, valid_acc_item, label='Item-based Imputation')
    # plt.xlabel('Number of Neighbors (k)')
    # plt.ylabel('Validation Accuracy')
    # plt.title('k-NN Imputation Performance')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # plt.savefig('knn_imputation.png')

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, valid_acc_user, label='User-based Imputation')
    plt.plot(k_values, valid_acc_item, label='Item-based Imputation')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Accuracy')
    plt.title('k-NN Imputation Performance')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('knn_imputation.png')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
