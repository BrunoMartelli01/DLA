

# Lab 4: Adversarial Learning and OOD Detection Laboratory

## Overview
This laboratory is divided into three main exercises, progressing from baseline **OOD detection** on CIFAR-10 to adversarial training with FGSM and advanced OOD scoring with ODIN.

The notebook focuses on two connected goals: detecting out-of-distribution samples and improving model robustness against adversarial attacks.

---

## Exercise 1: OOD Detection and Performance Evaluation
**Objective:** Build a simple OOD detection pipeline and evaluate it with quantitative metrics on an in-distribution dataset and an out-of-distribution one.

*   **Key Components:**
    *   **Dataset Setup:** The notebook uses CIFAR-10 as the in-distribution dataset, FakeData as the OOD dataset, and creates a 90%/10% train-validation split from the original CIFAR-10 training set using `Subset` and `DataLoader`.
    *   **`CNN` Classifier:** A custom classifier with 5 convolutional layers and 3 fully connected layers is trained on CIFAR-10, and the best checkpoint is saved as `cifar10.pth` based on validation loss.
    *   **Maximum Logits:** The classifier is used as an OOD detector by scoring samples with the confidence of its output distribution.
    *   **`Autoencoder` Baseline:** A convolutional autoencoder with 3 encoder layers and 3 decoder layers is trained to reconstruct CIFAR-10 images, and its best checkpoint is saved as `cifar10_ae.pth`.
    *   **Reconstruction-Based Scoring:** The autoencoder assigns OOD scores from the reconstruction error, using the negative mean MSE so that better reconstructions correspond to higher in-distribution confidence.
    *   **Evaluation Metrics:** The lab implements AUROC, AUPR, FPR@95TPR, and Detection Error@95TPR, and plots ROC and Precision-Recall curves for visual comparison.

*   **Outcome:** The CNN reaches about 70% test accuracy on CIFAR-10 and provides a simple baseline for OOD detection.
*   **Outcome:** The autoencoder offers a reconstruction-based alternative that, in the notebook analysis, shows stronger overall separation than the CNN across the full score range.

---

## Exercise 2: Enhancing Robustness to Adversarial Attack
**Objective:** Implement FGSM, generate adversarial examples, and train a more robust classifier by augmenting training with adversarially perturbed samples.

*   **Key Components:**
    *   **FGSM Attack:** The notebook defines `fgsm_attack_batch`, which computes the gradient of the loss with respect to the input, takes the sign of that gradient, applies a perturbation of magnitude `eps`, and clamps the result to the valid image range.
    *   **Qualitative Adversarial Analysis:** The lab visualizes adversarial perturbations and shows example attacks on CIFAR-10 classes to inspect how small input changes can fool the model.
    *   **Adversarial Training:** The function `train_with_fgsm` trains the CNN using a weighted combination of clean and adversarial loss, monitors clean accuracy, adversarial accuracy, validation loss, and OOD metrics, and saves the best model as `cifar10_adv.pth`.
    *   **Training Configuration:** In the reported experiment, the adversarial model is trained for 10 epochs with `eps = 32/255`, `alpha = 0.3`, `lr = 0.001`, and weight decay.

*   **Outcome:** FGSM training strongly improves adversarial robustness compared with the clean-trained model.
*   **Outcome:** It also improves OOD detection quality, while slightly reducing clean accuracy, showing the classic trade-off between robustness and standard performance.

---

## Exercise 3: Advanced OOD Detection with ODIN
**Objective:** Implement ODIN as a stronger OOD detector by combining temperature scaling and gradient-based input preprocessing.

*   **Key Components:**
    *   **Temperature Scaling:** The logits are divided by a temperature parameter `T` before softmax so that score calibration and gradient behavior can be adjusted during detection.
    *   **Input Preprocessing:** The ODIN score perturbs the input with a small signed gradient step in the direction `x_perturbed = x - eps * sign(grad)`, which is designed to amplify the score gap between ID and OOD samples.
    *   **`compute_scores_ODIN`:** The notebook defines a dedicated scoring function that computes gradients on the input, perturbs samples, runs a second forward pass, and returns the maximum softmax probability after temperature scaling.
    *   **Curve Comparison:** ROC and Precision-Recall curves are plotted side by side to compare ODIN against baseline scoring methods.

*   **Outcome:** ODIN improves the OOD detection metrics over the plain baseline detector shown in the notebook.
*   **Outcome:** The lab presents ODIN as a practical extension of the baseline OOD pipeline, even if the gain is smaller than the one obtained through adversarial training.

---

## File Structure & Usage

### 1. `Lab4.ipynb` (The Lab Runner)
This notebook is the main entry point and contains the full implementation of all three exercises: baseline OOD detection, adversarial training with FGSM, and ODIN-based OOD scoring.

Run it with:
```bash
jupyter notebook Lab4.ipynb
