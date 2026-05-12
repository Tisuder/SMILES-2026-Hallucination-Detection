# Solution: Hallucination Detection in Small Language Models (SLMs)


## 1. Feature Engineering & Aggregation Strategy

The core of the solution lies in the `aggregation.py` implementation, which transforms high-dimensional tensor data into informative feature vectors.

* **Aggregation Window:** After extensive testing, a window of the **last 20 tokens** of the response was selected. Experiments with larger windows (e.g., 40 tokens) introduced semantic noise, suggesting that the "hallucination signal" in SLMs is most concentrated in the final stages of token generation.
* **Strategic Layer Selection:** Instead of focusing solely on the final layers, the solution utilizes a distributed layer selection strategy: `[-16, -12, -8, -4, -1]`.
* *Rationale:* Capturing states from the middle of the network (where logical structure is formed) and the end of the network (where the final output is refined) allows the probe to detect "representation drift" or internal inconsistencies.


* **Feature Construction:** Mean pooling is applied across the selected tokens for each layer, and the resulting vectors are concatenated into a single **4480-dimensional** feature vector ($5 \text{ layers} \times 896 \text{ hidden dims}$).

## 2. Model Architecture (The Probe)

To handle the $p \gg n$ problem (4480 features vs ~480 training samples per fold), a highly regularized **Shallow MLP probe** was implemented in `probe.py`:

* **Architecture:** A 2-layer MLP (64 hidden units with ReLU activation). This provides enough non-linearity to distinguish complex patterns without the capacity to overfit.
* **Dimensionality Reduction:** Dimensionality Reduction: PCA with 50 components is applied before training. This removes correlated noise and forces the model to learn only the most significant directions of variance.
* **Regularization**: Weight Decay (L2): Set to 0.5 in the Adam optimizer. This aggressive penalty keeps weights small and ensures the model generalizes to unseen test data.
* **Optimization**: Adam optimizer ($Lr=0.001$) trained for **400 epochs**.

## 3. Experiments and Failed Attempts (Extended)

Beyond the architectural changes, several optimization strategies were explored but ultimately discarded:

* **High-Capacity Architectures:** We experimented with deeper neural networks (3-5 hidden layers) and wider layers (up to 512 units). While these models achieved near-perfect accuracy on the training folds, the validation AUROC dropped significantly. This confirmed that high-capacity models were merely "memorizing" the high-dimensional noise rather than learning the underlying features of hallucinations.
* **PCA Dimensionality Tuning:** We performed a grid search for the optimal number of PCA components ($n \in [20, 50, 100, 200, 500]$):
* **$n < 50$:** The model lost too much variance, failing to capture the subtle differences between truthful and hallucinated states.
* **$n > 100$:** The probe started picking up "spurious correlations" (noise) inherent in the 4480-dimensional embedding space, leading to poor generalization.
* **Result:** $n=50$ was identified as the "sweet spot" for dimensionality reduction.


* **Complex Regularization:** We tried implementing Dropout ($p=0.5$) and BatchNorm. However, on such a small dataset, these techniques made the training unstable and yielded inconsistent results across folds.

### Why the Simple Solution Won?

Following **Occam's Razor**, the final choice of a shallow MLP with aggressive L2 weight decay proved to be the most robust. It provides just enough non-linearity to separate the classes while the heavy regularization forces the model to ignore non-essential features, ensuring it performs well on the unseen test set.

## 4. Ablation Study

The following table summarizes the experiments conducted to reach the final configuration. All metrics are based on **Stratified 5-Fold CV Mean AUROC**.

| Configuration | Tokens | Layers | Features | Val AUROC | Status |
| --- | --- | --- | --- | --- | --- |
| **Optimized MLP Probe** | **20** | **[-16, -12, -8, -4, -1]** | **Concatenated** | **0.7514** | **Final** |
| Base Configuration | 20 | [0, 6, 12, 18, 24] | Concatenated | 0.7508 | Baseline |
| Extended Window | 40 | [0, 6, 12, 18, 24] | Concatenated | 0.7459 | Noisy |
| Geometric Features | 40 | [0, 6, 12, 18, 24] | +Stats/Norms | 0.7418 | Redundant |
| Tail-Only Focus | 40 | [-8, -6, -4, -2, -1] | Concatenated | 0.7386 | Deficient |

**Key Finding**: Increasing the window size or adding hand-crafted geometric features (norms, similarity) did not yield improvements. Raw activations from distributed layers provided the most robust signal.
### Summary of Best Results (Ablation Study)
Base (Uniform layers, 20 tokens): 0.7508

Final (Distributed layers -16 to -1, 20 tokens): 0.7514

## 5. Validation Methodology

* **Stratified 5-Fold Cross-Validation:** Essential for ensuring the model generalizes across the small and imbalanced dataset (Hallucination ratio ~2.3:1).
* **Primary Metric:** **AUROC** was chosen over Accuracy or F1-score as it provides a threshold-independent measure of the model's ability to distinguish between classes, which is critical for the "blind" testing phase of the competition.

## 6. Reproducibility Instructions
Requirements
The solution requires the following Python environment:

Execution Pipeline
To reproduce the results and generate predictions.csv:

Place aggregation.py, probe.py, and split_data.py in the same directory as the official solution.py.

Run the standard pipeline:
```
python solution.py
```

