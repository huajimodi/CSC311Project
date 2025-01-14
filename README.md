# Extended Neural Network for Imputing Sparse Student-Question Data

Welcome to the **Extended Neural Network** project! This repository demonstrates how we move beyond a simple autoencoder-based neural network for modeling student responses, by introducing **cluster-based imputation** of missing data using a **question correlation matrix** derived from subject metadata and TF-IDF similarity. Our improvements mitigate the effects of extremely sparse data and increase predictive accuracy for student performance.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Techniques](#key-techniques)
3. [Methodology](#methodology)
    - [Data Preprocessing](#data-preprocessing)
    - [Autoencoder Architecture](#autoencoder-architecture)
    - [Question Correlation Matrix](#question-correlation-matrix--c_q-)
    - [K-Means Clustering for Imputation](#k-means-clustering-for-imputation)
    - [Training and Regularization](#training-and-regularization)
4. [Implementation](#implementation)
    - [Base Neural Network Code](#base-neural-network-code)
    - [Extended Neural Network Code](#extended-neural-network-code)
5. [Usage](#usage)
6. [Performance and Results](#performance-and-results)
7. [Limitations](#limitations)
8. [References and Acknowledgments](#references-and-acknowledgments)

---

## Overview

**Motivation**  
Our dataset is extremely sparse: roughly $94\%$ of the student-question entries are missing. A naive assumption that all missing answers are incorrect ($0$) injects significant noise into the training matrix. Instead, we propose a better **cluster-based imputation** strategy informed by question similarities, leading to a more robust representation of student knowledge.

**Contributions**  
1. **Label Shifting**: We shift all incorrect answers from $0$ to $-1$, preserving $0$ explicitly for *missing* entries only. This helps the model differentiate truly incorrect answers ($-1$) from simply missing data ($0$).  
2. **Question Correlation Matrix**: We generate a matrix $C_Q$ that captures how similar or correlated two questions are, based on their subject metadata (via TF-IDF).  
3. **K-Means Clustering**: For each question cluster, we estimate a student's missing responses using a simple majority-voting (mean-based) approach within that cluster.  
4. **Extended Autoencoder**: Our base autoencoder is enhanced by plugging in these more meaningful imputed values and by incorporating an additional regularization term.

---

## Key Techniques

1. **Autoencoder**  
   - A two-layer linear network ($g$ and $h$) with sigmoid activations.  
   - Learns a compressed ($k$-dimensional) representation of student responses and reconstructs the input vector.

2. **Imputation via Question Similarity**  
   - Subject correlation matrix $C_S$ built using TF-IDF on subject names.  
   - Question correlation matrix $C_Q = A C_S A^{T}$, where $ A $ is the question-subject assignment matrix.  
   - Normalizing $C_Q$ ensures question similarities lie between $0$ and $1$.

3. **K-Means**  
   - We cluster questions into $k_{\text{means}}$ groups based on $C_Q^{\text{normalized}}$.  
   - Missing responses for each student are filled using a mean-based decision from the questions in the same cluster.

4. **Regularization**  
   - We include a weight-decay penalty $\lambda$ on the autoencoder’s weights.  
   - Improves generalization and stabilizes training.

---

## Methodology

### Data Preprocessing

1. **Load Sparse Matrix**: We begin with a matrix $X \in \mathbb{R}^{N \times Q}$, where $N$ is the number of students and $Q$ is the number of questions. Originally, many entries are NaN.  
2. **Label Shifting**:  
   - **$-1$** for incorrect answers,  
   - **$0$** for missing answers,  
   - **$+1$** for correct answers.  

### Autoencoder Architecture

We use a simple autoencoder comprising:
- **Encoder**: $\mathbf{z} = \sigma \bigl(W_1 \cdot \mathbf{x} \bigr)$  
- **Decoder**: $\hat{\mathbf{x}} = \sigma \bigl(W_2 \cdot \mathbf{z} \bigr)$  

where $\sigma$ is the elementwise sigmoid function, and $\mathbf{x} \in \mathbb{R}^Q$ is the vector of a single student’s responses across all questions.

### Question Correlation Matrix $ C_Q $

Given:
- **Question-Subject Assignment Matrix** $A \in \mathbb{R}^{Q \times S}$, with $A_{q,s} = 1$ if question $q$ is linked to subject $s$, else $0$.  
- **Subject Correlation Matrix** $C_S \in \mathbb{R}^{S \times S}$ computed via TF-IDF similarity on subject names.  

We define:

![Equation](https://latex.codecogs.com/svg.latex?C_Q%20%3D%20A%20C_S%20A^T)

We then normalize $C_Q$ so that its diagonal entries are $1$, ensuring all similarity scores lie in $[0,1]$.

### K-Means Clustering for Imputation

To impute missing entries:

1. **Clustering**:  
```math
\{\mathcal{C}_1, \mathcal{C}_2, \ldots, \mathcal{C}_K\} = \text{K-Means}(C_Q^{\text{normalized}}, K)
```

   Each question is assigned to one cluster $\mathcal{C}_k$.

2. **Missing Entry Imputation**:  
   For student $s_m$ and question $q_n$ with a missing response ($X_{m,n} = \text{NaN}$):  
   
   - Find the cluster $\mathcal{C}_k$ to which $q_n$ belongs.  
   - Let $\mathcal{A}_k^{(m)}$ be the set of valid (non-missing) answers student $s_m$ has for all questions in $\mathcal{C}_k$.  
   - If $\mathcal{A}_k^{(m)}$ is non-empty, compute the mean $\mu_m^{(k)}$. If $\mu_m^{(k)} > 0$, impute $+1$, else $-1$. If $\mathcal{A}_k^{(m)}$ is empty, assign $0$.

### Training and Regularization

**Loss Function**  
We minimize the **reconstruction loss** (sum of squared errors) plus a **weight-decay** term:  
```math
\mathcal{L}(\theta)\;=\;\sum_{m=1}^{N}\bigl\|\hat{\mathbf{x}}_m-\mathbf{x}_m\bigr\|^2\;+\;\frac{\lambda}{2}\Bigl(\|W_1\|_F^2+\|W_2\|_F^2\Bigr)
```
We use **Stochastic Gradient Descent (SGD)** with a chosen learning rate (e.g., `lr = 0.005`) and run for `num_epoch = 80` epochs.


## Implementation

Below are two main scripts in this repository:

### Base Neural Network Code

- **File**: `nn.py` (or a similar name)  
- **Purpose**: Implements the autoencoder with zero-based imputation (filling missing entries with $0$) and trains across various latent dimensions $k$.

Key points:
- `AutoEncoder(nn.Module)`:  
  - **Forward** uses two linear layers (`self.g`, `self.h`), each followed by a sigmoid.  
  - **train** function includes reconstruction loss on known entries plus weight norm penalty.  

```python
class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        super(AutoEncoder, self).__init__()
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def forward(self, inputs):
        hidden = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(hidden))
        return out
```

### Extended Neural Network Code

- **File**: `extended_nn.py` (or similar)  
- **Purpose**: Improves data preprocessing by:
  1. **Computing** the question correlation matrix $C_Q$ via TF-IDF-based subject similarity $C_S$.  
  2. **Clustering** questions with K-Means ($k_{\text{means}}=14$ by default).  
  3. **Imputing** missing entries in $\mathbf{X}$ based on cluster membership.  
  4. **Training** the same autoencoder architecture but on the updated training matrix with a possible extra correlation-based regularization term.

**Steps**:
1. `get_correlation_matrix(...)`  
   - Loads subject metadata and computes $C_S$ via `cosine_similarity` on TF-IDF.  
   - Returns $C_Q^{\text{normalized}}$.  
2. `load_data(k_mean=14, ...)`  
   - Reads the question metadata, subject meta, and merges them via an assignment matrix $A$.  
   - Applies `KMeans` to cluster questions.  
   - Imputes student responses.  
3. `AutoEncoder` remains structurally the same.  
4. `train(...)` function includes the usual reconstruction loss and an optional *decoder-correlation* penalty.

```python
decoder_weights = model.h.weight
reg_term = torch.trace(
    torch.matmul(
        torch.matmul(decoder_weights.t(), C_Q_tensor),
        decoder_weights
    )
).clamp(min=0)
```

---

## Usage

1. **Install Dependencies**  
   - [PyTorch](https://pytorch.org/)  
   - [scikit-learn](https://scikit-learn.org/) for KMeans  
   - [pandas](https://pandas.pydata.org/) for CSV loading  
   - [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) for visualization

2. **Prepare Data**  
   - Place your CSV files and sparse data files in a `./data/` directory or supply a custom path in the code.

3. **Train the Extended Model**  
   - Run `python extended_nn.py`.  
   - Hyperparameters (learning rate, $\lambda$, `k_mean`) can be modified at the top of the script.

4. **Check Results**  
   - Training and validation losses/accuracies are printed each epoch.  
   - Final test accuracy is displayed upon completion.

---

## Performance and Results

- **Extended Model Accuracy**: With our cluster-based imputation + label shifting + correlation-based regularization, we achieve a **test accuracy of ~0.6997**, outperforming the base neural network variants.  
- **Visualizations**:  
  - A subset of the correlation matrix $C_Q^{\text{normalized}}$ can be plotted (e.g., 20 questions).
    ![question_correlation_matrix](https://github.com/user-attachments/assets/9524f76b-0ff9-4e4b-a360-b5c19d10305c)

  - Training vs. validation accuracy curves help diagnose overfitting or underfitting.

**Comparative Table**  
Below is a summary of test accuracies for multiple approaches:

| Model                           | Test Accuracy |
| ------------------------------- | ------------- |
| User-based kNN                  | 0.6890        |
| Item-based kNN                  | 0.6894        |
| IRT                             | 0.6994        |
| Base NN (no regularization)     | 0.6808        |
| Base NN (with regularization)   | 0.6861        |
| Extended NN (no regularization) | 0.6991        |
| **Extended NN (with reg.)**     | **0.6997**    |

---

## Limitations

1. **Subject/Question Similarity**  
   - Our method assumes similarity among all questions linked to the same subjects, neglecting their varying difficulty. A student might be able to handle easier questions but fail advanced ones in the same subject cluster.

2. **Ignoring Additional Metadata**  
   - We do not incorporate extra student attributes (e.g., age, subscription status, etc.). Such features might further enhance personalized predictions.

3. **High Computational Cost**  
   - With large-scale data, the cluster-based imputation can be expensive. We handle $\sim 900{,}000$ missing entries, which can be time-consuming to fill based on cluster membership.

---

## References and Acknowledgments

- **PyTorch** for autoencoder implementation.  
- **scikit-learn** for KMeans clustering and cosine similarity.  
- **pandas** for data wrangling and CSV handling.  
- **matplotlib** & **seaborn** for plotting metrics and heatmaps.

We hope this extended neural network approach, incorporating more intelligent imputation via question correlation, demonstrates a practical improvement over naive zero-based fill strategies in highly sparse educational data.
