---

# Lab 3: Transfer Learning & Prompt Learning Laboratory

## Overview
This laboratory is divided into three main exercises, progressing from basic **feature extraction** with pre-trained transformers to advanced **parameter-efficient fine-tuning** using prompt learning with CLIP.

---

## Exercise 1: Sentiment Analysis (Warm-up)
**Objective:** Build a sentiment analysis pipeline using a pre-trained DistilBERT model on the Rotten Tomatoes movie review dataset.

**Key Components:**
*   **Dataset Loading**: Using the HuggingFace `Datasets` library to load and explore the [Cornell Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) dataset (5,331 positive + 5,331 negative reviews), with splits `train`, `validation`, and `test`.
*   **DistilBERT + Tokenizer**: Loading `distilbert/distilbert-base-uncased` via `AutoModel` and `AutoTokenizer`, tokenizing batches with padding, and inspecting the `[CLS]`, `[SEP]`, `[PAD]` special tokens.
*   **Feature Extraction Pipeline**: Using HuggingFace `pipeline("feature-extraction")` to extract the `[CLS]` token embedding (768-dimensional) from the last transformer layer for each review.
*   **SVM Classifier**: Training a `sklearn.svm.SVC(kernel="linear")` on the extracted `(8530, 768)` feature matrix as a stable baseline.

---

## Exercise 2: Fine-tuning DistilBERT
**Objective:** Improve sentiment analysis performance by fully fine-tuning DistilBERT end-to-end on the Rotten Tomatoes dataset.

**Key Components:**
*   **Token Preprocessing**: Using `Dataset.map` with a batched lambda to tokenize all splits efficiently, adding `input_ids` and `attention_mask` columns directly to the dataset object.
*   **`AutoModelForSequenceClassification`**: Loading DistilBERT with a randomly initialized classification head (`pre_classifier` + `classifier` linear layers) on top of the frozen base, ready for supervised fine-tuning.
*   **HuggingFace `Trainer`**: Configuring `TrainingArguments` with epoch-level evaluation, `load_best_model_at_end=True`, L2 weight decay, and a low learning rate (`5e-7`). `EarlyStoppingCallback(early_stopping_patience=10)` prevents overfitting.
*   **`DataCollatorWithPadding`**: Handling dynamic padding at batch construction time for memory efficiency.

---

## Exercise 3: Parameter-Efficient CLIP Fine-tuning (CoOp)
**Objective:** Adapt a frozen CLIP model to the `Flowers102` image classification task using prompt learning (CoOp), without modifying any of CLIP's original weights.

**Key Components:**
*   **`CoOpCLIP` (Prompt Learning)**: Implements **Context Optimization (CoOp)**. Only a small set of learnable `context_vectors` (16 tokens) is trained. All CLIP parameters are frozen (`requires_grad=False`). Prompts are constructed as `[SOS] [learned context] [classname] [EOT]`.
*   **Zero-Shot Baseline**: Evaluating `openai/clip-vit-base-patch16` directly on Flowers102 without any training, establishing the starting performance.
*   **Few-Shot Data Splits**: A custom `create_few_shot_split` function builds class-balanced subsets with exactly `n` examples per class (tested for `shots ∈ {1, 2, 4, 8, 16}`).
*   **CLIP Internals Exploration**: Manual replication of the text encoder pipeline (causal attention mask → transformer encoder → LayerNorm → EOT pooling → text projection) to verify a correct understanding of the architecture.

---

## File Structure & Usage

### 1. `Lab3.ipynb` (The Lab Runner)
This is the main entry point for all three exercises. Run it sequentially:
```bash
jupyter notebook Lab3.ipynb
