# **Lelapa AI Buzuzu-Mavi Challenge: Solution Documentation**

**Competitor:** Gritty_K

## 1. Overview and Objectives

*   **Purpose:** This solution addresses the Lelapa AI Buzuzu-Mavi Challenge, which aims to create more efficient and effective versions of the InkubaLM language model specifically for Swahili (swa) and Hausa (hau). The challenge encourages participants to improve performance on downstream tasks while potentially reducing model size.
*   **Problem Addressed:** The base InkubaLM-0.4B model, while powerful, is resource-intensive and may not be optimally performant for specific African languages and tasks out-of-the-box. This solution focuses on adapting the model for improved accuracy on Sentiment Analysis, Natural Language Inference (AfriXNLI), and Machine Translation (English-to-Swahili/Hausa).
*   **Solution Approach:** This solution employs **full supervised fine-tuning (SFT)** of the `lelapa/InkubaLM-0.4B` model. Unlike parameter-efficient fine-tuning (PEFT) methods like QLoRA, this approach updates *all* model parameters to specialize it for the target tasks and languages using the provided training data. A key preprocessing step involves balancing target sequence lengths across different tasks to stabilize training. Crucially, during inference, Machine Translation tasks are handled by `csv-trans`) and then passed to the fine-tuned model.
*   **Objectives:**
      *   Fine-tune the InkubaLM-0.4B model to excel in sentiment analysis, AfriXNLI, and machine translation for Swahili and Hausa.
      *   Contribute to the development of accessible and performant language models for African languages.
      *   Expected Outcomes: Deliver a compact, high-performing model capable of supporting real-world applications like call center operations, educational tools, medical bots, and agricultural assistance in Swahili and Hausa-speaking regions.

## 2. Architecture (Textual Description)

The solution follows a standard fine-tuning and inference pipeline:

1.  **Data Acquisition (ETL - Extract):** Datasets for Sentiment (`lelapa/SentimentTrain`, `lelapa/SentimentTest`), Machine Translation (`lelapa/MTTrain`, `lelapa/MTTest`), and AfriXNLI (`lelapa/XNLITrain`, `lelapa/XNLITest`) are loaded directly from the Hugging Face Hub using the `datasets` library.
2.  **Data Preprocessing (ETL - Transform):**
    *   Datasets for each task (Train and Test separately) are combined, ensuring a unified structure with all necessary columns (`ID`, `instruction`, `inputs`, `premise`, `targets`, etc.). Missing columns are filled with empty strings.
    *   Data is converted to Pandas DataFrames for easier manipulation.
    *   The specific `task` (sentiment, mt, afrixnli) is extracted from the `ID` string.
    *   **Target Length Balancing:** For the training dataset, target sequences for Sentiment and AfriXNLI tasks are repeated 11 times to approximate the average length of Machine Translation targets, aiming to prevent the model from overly focusing on short generations.
    *   Data is converted back to `datasets.Dataset` format.
    *   Each data point is formatted into an instruction-following prompt structure:
        ```
        ### Instruction: {instruction}
        ### Input: {premise}\n{inputs}
        ### Response: {targets}
        ```
        (For test data, the `### Response:` part is included but left empty for the model to fill).
3.  **Model Loading:** The base `lelapa/InkubaLM-0.4B` model and its corresponding tokenizer are loaded using `transformers.AutoModelForCausalLM` and `transformers.AutoTokenizer`. **Notably, 4-bit quantization (QLoRA) is explicitly disabled (`use_4bit=False`) for full parameter fine-tuning.**
4.  **Fine-Tuning (Modeling):**
    *   The `trl.SFTTrainer` is configured for supervised fine-tuning.
    *   The `DataCollatorForCompletionOnlyLM` is used to ensure the model only computes loss and learns based on the `### Response:` part of the formatted prompt.
    *   The model is trained for 41 epochs on the preprocessed and balanced training dataset, updating all of its parameters.
    *   Checkpoints are saved periodically.
5.  **Inference:**
    *   The final fine-tuned model checkpoint (`checkpoint-3520` in the provided script) is loaded.
    *   The preprocessed test dataset (without target length balancing) is used as input.
    *   For each test sample:
        *   **Sentiment & AfriXNLI:** The formatted prompt is fed to the `inference_model.generate()` function to produce a response.
        *   **Machine Translation:** The `inputs` field (English text) is passed to `csv-trans` library for translation to the target language (Swahili 'sw' or Hausa 'ha') then passed to the fine-tuned model which handles the other two tasks as well.
    *   **Post-processing:**
        *   Sentiment: The first word of the generated response is mapped to an integer label (0, 1, 2).
        *   XNLI: The first word is checked if it's a digit, converted to an integer, and taken modulo 3. Defaults to 0 if processing fails.
        *   MT: `csv-trans` is used.
6.  **Output Generation:** A `submission.csv` file is created containing the `ID` and the final processed `Response` for each test sample.

## 3. ETL Process

*   **Extract:**
    *   **Sources:** Hugging Face Hub datasets (`lelapa/SentimentTrain`, `lelapa/MTTrain`, `lelapa/XNLITrain` and corresponding `Test` splits).
    *   **Formats:** Parquet files hosted on Hugging Face.
    *   **Method:** `datasets.load_dataset()` function.
*   **Transform:**
    *   **Logic:**
        *   Combining datasets (`concatenate_datasets`).
        *   Schema unification (adding missing columns, casting `targets` to string).
        *   Task extraction from `ID`.
        *   **Target Length Balancing (Training Only):** Repeating Sentiment/XNLI targets using `balance_target_lengths(repetition_factor=11)`. *Rationale: To mitigate potential bias towards generating very short sequences due to the prevalence of single-word/digit answers in Sentiment/XNLI tasks compared to longer MT outputs.*
        *   Instruction Formatting: Applying `formatting_prompts_func` to structure the input for the SFTTrainer/inference.
    *   **Tools:** `datasets`, `pandas`.
*   **Load:**
    *   **Training:** The transformed `Dataset` object is directly passed to `trl.SFTTrainer`.
    *   **Inference:** The transformed test `Dataset` (as a Pandas DataFrame) is iterated over for prediction generation.

## 4. Data Modeling

*   **Model:** `lelapa/InkubaLM-0.4B` - A 0.4 billion parameter causal language model.
*   **Approach:** Full Supervised Fine-Tuning (SFT). All model parameters were unfrozen and updated during training.
*   **Feature Engineering:** Primarily handled by the instruction formatting (`formatting_prompts_func`), which clearly separates instructions, context (inputs/premise), and the target response the model should learn to generate. The target length balancing is also a form of data transformation influencing model learning.
*   **Training:**
    *   **Framework:** `trl` (specifically `SFTTrainer`) built on `transformers` and `pytorch`.
    *   **Optimizer:** `adamw_bnb_8bit` specified, though effectiveness might be standard AdamW as `BitsAndBytesConfig` for model layers was not enabled (`use_4bit=False`).
    *   **Epochs:** 41
    *   **Batch Size:** 2 per device * 8 gradient accumulation steps = 16 effective batch size.
    *   **Sequence Length:** Max 256 tokens.
    *   **Collator:** `DataCollatorForCompletionOnlyLM` (masks loss calculation for prompt tokens).
    *   **Environment:** Kaggle GPU instance.
*   **Model Validation:** Performed implicitly via the Zindi platform's public (30% data) and private (70% data) leaderboard splits. Two submissions selected for final private leaderboard scoring.

## 5. Inference

*   **Deployment:** Model checkpoint loaded directly from disk (`/kaggle/input/.../checkpoint-3520`) into the inference notebook using `AutoModelForCausalLM.from_pretrained`.
*   **Input:** Test data loaded and formatted using `formatting_prompts_func`.
*   **Prediction Process:**
    *   **Sentiment/XNLI:** `inference_model.generate()` with `max_new_tokens=20`, `do_sample=False`.
    *   **Machine Translation:** call to `csv-trans` library.
*   **Output Interpretation & Post-processing:**
    *   Sentiment: `encode_sentiment_label` function maps generated text ("Kyakkyawa", "Chanya", etc.) to 0, 1, 2.
    *   XNLI: Extracts the first token, converts to `int`, takes `% 3`. Handles non-digit outputs by defaulting to 0.
    *   MT: Uses the string output from `csv-trans` passed to the fine-tuned model.
*   **Model Size:** The number of parameters in the loaded inference model is calculated as **664,160,256** (~0.66 billion).

## 6. Run Time

*   **Training:** Approximately **2 hours and 32 minutes** for 41 epochs on a Kaggle GPU environment
*   **Inference:**  Approximately **5 minutes** for the 900 samples.

## 7. Performance Metrics

*   **Primary Evaluation (Zindi):** Weighted average score based on:
    *   Sentiment Analysis: F1 Score
    *   AfriXNLI: F1 Score
    *   Machine Translation: CHaR-F Score
*   **Final Score Calculation:** `zindi_score = (0.389459491 + (1-(0.66))*0.389459491 )/2`
    *   `PrivateLB_score`: Score achieved on the private leaderboard. 0.389459491
    *   `size`: Number of parameters in the submitted model = **664,160,256** **~0.66B**.
    *   `PARAM_SIZE`: Size of the original InkubaLM-0.4B model
    
*   **Reported Scores:**
    *   Public Leaderboard Score: 0.442620461
    *   Private Leaderboard Score: 0.389459491
*   **Training Metric:** Training Loss (monitored during training via `SFTTrainer` logging).

## 8. Error Handling and Logging

*   **Training:**
    *   `SFTTrainer` provides basic logging of training steps, loss, and progress.
    *   Checkpointing (`save_strategy="epoch"`, `save_total_limit=2`) allows resuming training in case of interruptions.

## 9. Maintenance and Monitoring

*   **Model Updates:** Requires re-running the fine-tuning notebook (`41 epochs_zindi-lelapa-full-finetune-Inkub`). This could be triggered by new data availability, changes in task requirements, or the need for hyperparameter tuning.
*   **Dependencies:** Solution relies on specific versions of `transformers`, `trl`, `datasets`, `torch`, `pandas`, `peft`, `bitsandbytes`, `csv-trans`, etc. A `requirements.txt` or environment definition file would be needed for consistent reproduction.
*   **Monitoring:**
    *   Performance should be tracked via evaluation on hold-out sets using the official Zindi metrics (F1, CHaR-F) or by monitoring Zindi leaderboard scores if applicable post-competition.

## 10. Additional Notes & Insights

*   **Full Fine-Tuning Choice:** This solution opted for full fine-tuning rather than PEFT/LoRA. This means while it aims for higher specialization on the tasks, it does *not* achieve model compression compared to the base InkubaLM-0.4B. This impacts the final score calculation, as the size reduction component is negligible.
*   **Target Length Balancing:** The technique of repeating shorter target sequences during training was implemented to encourage the model to learn longer generations and potentially improve stability across the diverse task types. Its effectiveness should be validated against training without balancing.

*   **Reproducibility:** Ensure the `kaggle_secrets` (for `HFtoken`) are handled correctly and that the specified checkpoint path (`/kaggle/input/41-epochs-zindi-lelapa-full-finetune-inkub/sft_model/balanced/checkpoint-3520`) is accessible during code review. 

