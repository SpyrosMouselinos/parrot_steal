from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    """
    Configuration for training model.
    """

    model_ckpt: Optional[str] = field(
        default="https://huggingface.co/BlazeLlama/piwpaw_medium", metadata={"help": "Model name or path of model to be trained."}
    )
    save_dir: Optional[str] = field(
        default="./models", metadata={"help": "Save dir where model repo is cloned and models updates are saved to."}
    )
    dataset_name_train: Optional[str] = field(
        default="lvwerra/codeparrot-clean-train", metadata={"help": "Name or path of training dataset."}
    )
    dataset_name_valid: Optional[str] = field(
        default="lvwerra/codeparrot-clean-valid", metadata={"help": "Name or path of validation dataset."}
    )
    train_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for training."})
    valid_batch_size: Optional[int] = field(default=12, metadata={"help": "Batch size for evaluation."})
    weight_decay: Optional[float] = field(default=0.1, metadata={"help": "Value of weight decay."})
    shuffle_buffer: Optional[int] = field(
        default=10000, metadata={"help": "Size of buffer used to shuffle streaming dataset."}
    )
    learning_rate: Optional[float] = field(default=3e-4, metadata={"help": "Learning rate fo training."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "Learning rate."})
    num_warmup_steps: Optional[int] = field(
        default=500, metadata={"help": "Number of warmup steps in the learning rate schedule."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "Number of gradient accumulation steps."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Use gradient checkpointing to reduce memory footprint."}
    )
    max_train_steps: Optional[int] = field(default=23_000, metadata={"help": "Maximum number of training steps."})
    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Sequence lengths used for training."})
    seed: Optional[int] = field(default=1, metadata={"help": "Training seed."})
    save_checkpoint_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Interval to save checkpoints. Measured as number of forward passes not training steps."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "States path if the training should continue from a checkpoint folder."}
    )
    tokenized: Optional[bool] = field(default=False, metadata={"help": "If True the data is pretokenized."})


@dataclass
class EvaluationArguments:
    """
    Configuration for evaluating model.
    """

    model_ckpt: Optional[str] = field(
        default="BlazeLlama/piwpaw_medium", metadata={"help": "Model name or path of model to be evaluated."}
    )
    dataset_name: Optional[str] = field(
        default="lvwerra/codeparrot-clean-valid", metadata={"help": "Name or path of validation dataset."}
    )
    batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size used for evaluation."})
    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Length of sequences to be evaluated."})
    seed: Optional[int] = field(default=1, metadata={"help": "Random seed used for evaluation."})


@dataclass
class HumanEvalArguments:
    """
    Configuration for running evaluation on HumanEval dataset.
    """

    model_ckpt: Optional[str] = field(
        default="BlazeLlama/piwpaw_medium", metadata={"help": "Model name or path of model to be evaluated."}
    )
    num_workers: Optional[int] = field(default=8, metadata={"help": "Number of workers used for code evaluation."})
    num_tasks: Optional[int] = field(
        default=None,
        metadata={"help": "The number of human-eval tasks to run. If not included all tasks are evaluated."},
    )
    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Sample from the language model's output distribution."}
    )
    temperature: Optional[float] = field(default=0.8, metadata={"help": "Sampling temperature used for generation."})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "Maximum number of newly generated tokens."})
    top_k: Optional[int] = field(default=0, metadata={"help": "Top-k parameter used for generation."})
    top_p: Optional[float] = field(default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."})
    batch_size: Optional[int] = field(default=10, metadata={"help": "Number of generations to run in parallel."})
    n_samples: Optional[int] = field(
        default=200, metadata={"help": "Number of completions to generate for each sample."}
    )
    seed: Optional[int] = field(default=1, metadata={"help": "Random seed used for evaluation."})
    output_file: Optional[str] = field(
        default="eval_results.json", metadata={"help": "Random seed used for evaluation."}
    )
    HF_ALLOW_CODE_EVAL: Optional[str] = field(
        default="0", metadata={"help": "Allow `code_eval` to execute Python code on machine"}
    )
    device_int: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "Determine which device to run the `text-generation` Pipeline on. -1 is CPU and any zero or positive"
                " number corresponds to which GPU device id to run on."
            )
        },
    )



@dataclass
class TokenizerTrainingArguments:
    """
    Configuration for tokenizer training.
    """

    base_tokenizer: Optional[str] = field(
        default="facebook/incoder-1B", metadata={"help": "Base tokenizer to build new tokenizer from."}
    )
    dataset_name: Optional[str] = field(
        default="transformersbook/codeparrot-train", metadata={"help": "Dataset to train tokenizer on."}
    )
    text_column: Optional[str] = field(default="content", metadata={"help": "Column containing text data to process."})
    vocab_size: Optional[int] = field(default=200_000, metadata={"help": "Number of examples to train tokenizer on."})
    n_examples: Optional[int] = field(
        default=32768, metadata={"help": "Number of examples to train the tokenizer on."}
    )
    tokenizer_name: Optional[str] = field(default="BlazeLlama/piwpaw_medium", metadata={"help": "Name of new tokenizer."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenizer to the hub."})


@dataclass
class PretokenizationArguments:
    """
    Configuration for data pretokenization.
    """

    tokenizer_dir: Optional[str] = field(
        default="lvwerra/codeparrot", metadata={"help": "Name or path to the tokenizer."}
    )
    dataset_name: Optional[str] = field(
        default="lvwerra/codeparrot-clean-train", metadata={"help": "Name or path to the dataset to pretokenize."}
    )
    tokenized_data_repo: Optional[str] = field(
        default="tokenized-codeparrot-train", metadata={"help": "Repo name of the pretokenized data."}
    )
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers used for code evaluation."})


@dataclass
class InitializationArguments:
    """
    Configuration for initializing new model.
    """

    config_name: Optional[str] = field(
        default="custom-medium", metadata={"help": "Configuration to use for model initialization."}
    )
    tokenizer_name: Optional[str] = field(
        default="BlazeLlama/piwpaw_medium", metadata={"help": "Tokenizer attached to model."}
    )
    model_name: Optional[str] = field(default="BlazeLlama/piwpaw_medium", metadata={"help": "Name of the created model."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenizer to the hub."})
