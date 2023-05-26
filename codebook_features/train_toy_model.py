"""Train script for Toy Codebook models."""

import copy
import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import transformers
from torch.utils.data import IterableDataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

import wandb
from codebook_features import models, run_clm
from codebook_features import trainer as cb_trainer

shortened_args = {
    "model_name_or_path": "mod",
    "learning_rate": "lr",
    "per_device_train_batch_size": "bs",
    "codebook_type": "cbt",
    "num_codes": "cbs",
    "num_codebooks": "ncb",
    "layers_to_snap": "cb_layers",
    "similarity_metric": "sim",
    "codebook_at": "cb_at",
    "loss": "loss",
    "train_model_params": "train_mod",
    "model_lr_factor": "mod_lrf",
    "k_codebook": "k",
    "dataset_name": "ds",
}


@dataclass
class ModelConfigArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""

    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_type: str = field(default="gptneox")
    hidden_size: int = field(default=128)
    intermediate_size: int = field(default=512)
    num_hidden_layers: int = field(default=4)
    num_attention_heads: int = field(default=4)
    rotary_emb_base: int = field(default=10000)
    seq_len: int = field(default=128)
    vocab_size: int = field(default=11)


class ToyGraph:
    """Toy graph that constructs an automata with N states and fixed number of edges per state."""

    def __init__(self, N: int = 100, transition_matrix=None, seed=None, edges=10):
        """Initialize the automata.

        Args:
        ----
            N: number of states in the automata.
            transition_matrix: transition matrix of probabilities of shape (N, N) describing the automata.
                If None, a random transition matrix is generated.
            seed: random seed for generating the transition matrix.
            edges: number of edges per state.
        """
        self.rng = np.random.default_rng(seed=seed)
        self.edges = edges
        if transition_matrix is None:
            self.transition_matrix = np.zeros((N, N))
            for i in range(N):
                self.transition_matrix[
                    i, self.rng.choice(N, size=edges, replace=False)
                ] = 1
            self.transition_matrix = (
                self.transition_matrix
                / self.transition_matrix.sum(axis=1, keepdims=True)
            )
            self.N = N
        else:
            self.transition_matrix = transition_matrix
            self.N = self.transition_matrix.shape[0]
        assert self.transition_matrix.shape == (N, N)

        self.state = 0
        self.digits = int(np.ceil(np.log10(N)))

    def step(self):
        """Step the automata."""
        self.state = self.rng.choice(self.N, p=self.transition_matrix[self.state])
        return self.state

    def step_with(self, state):
        """Step the automata from a given state."""
        return self.rng.choice(self.N, p=self.transition_matrix[state])

    def reset(self):
        """Reset the automata to the initial state."""
        self.state = 0
        return self.state

    def set_seed(self, seed):
        """Set the random seed."""
        self.rng = np.random.default_rng(seed=seed)

    def save(self, path):
        """Save the automata to a given path."""
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "automata.npy", self.transition_matrix)

    @classmethod
    def load(cls, path):
        """Load the automata from a given path."""
        transition_matrix = np.load(path)
        edges = (transition_matrix[0] != 0).sum()
        return cls(transition_matrix.shape[0], transition_matrix, edges=edges)

    def generate_trajectory(self, length):
        """Generate a trajectory of a given length."""
        trajectory = [self.rng.choice(self.N)]
        for _ in range(length - 1):
            trajectory.append(self.step_with(trajectory[-1]))
        return trajectory

    def generate_trajectories(self, length, start_states=None):
        """Generate trajectories of a given length from a given set of start states."""
        if start_states is None:
            curr_states = np.array(self.state * length)
        else:
            curr_states = copy.deepcopy(start_states)
        trajectories = np.zeros((len(start_states), length))
        for i in range(length):
            for j in range(len(start_states)):
                curr_states[j] = self.step_with(curr_states[j])
                trajectories[j, i] = curr_states[j]
        return trajectories

    def verify_trajectory(self, traj):
        """Verify that a given trajectory is valid."""
        for i in range(len(traj) - 1):
            if self.transition_matrix[traj[i], traj[i + 1]] == 0:
                print("Fail index:", i)
                print(traj[i], traj[i + 1])
                return False
        return True

    def nbrs(self, state):
        """Return the set of states that can be transitioned to from the given state."""
        return np.where(self.transition_matrix[state] != 0)[0]

    def nbrs_to(self, state):
        """Return the set of states that can transition to the given state."""
        return np.where(self.transition_matrix[:, state] != 0)[0]

    def transition_accuracy(self, trajs):
        """Compute the transition accuracy of a given set of trajectories.

        Also computes the accuracy of the first transition across the trajectories.
        """
        correct_transitions, correct_first_transitions, total_transitions = 0, 0, 0
        for traj in trajs:
            total_transitions += len(traj) - 1
            for i in range(len(traj) - 1):
                if self.transition_matrix[traj[i], traj[i + 1]] != 0:
                    correct_transitions += 1
                    if i == 0:
                        correct_first_transitions += 1
        if len(traj) > 1:
            return (
                correct_transitions / total_transitions,
                correct_first_transitions / len(trajs),
            )
        else:
            return 0, 0

    def seq_to_traj(self, sequences):
        """Convert a sequence of digits to a trajectory."""
        if type(sequences) == str:
            sequences = [sequences]
        trajs = []
        for seq in sequences:
            seq = seq.replace("<|endoftext|>", "")
            trajs.append([])
            for i in range(0, len(seq), self.digits):
                if len(seq[i : i + self.digits]) == self.digits:
                    trajs[-1].append(int(seq[i : i + self.digits]))
        return trajs

    def traj_to_str(self, traj):
        """Convert a trajectory to a string of digits."""
        return "".join(["0" * (self.digits - len(str(x))) + str(x) for x in traj])


class ToyDataset(IterableDataset):
    """Dataset for generating trajectories from a given automata."""

    def __init__(self, graph, tokenizer, seq_len, max_samples=-1, save_tokens=False):
        """Initialize the dataset.

        Args:
            graph: The automata to generate trajectories from.
            tokenizer: The tokenizer to use.
            seq_len: The length of the trajectories to generate.
            max_samples: The maximum number of samples to generate.
            save_tokens: Whether to save the tokens generated.
        """
        self.graph = graph
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.save_tokens = save_tokens
        self.tokens = []
        assert self.seq_len % self.graph.digits == 0

    def __iter__(self):
        """Generate a tokenized trajectory."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.graph.set_seed(worker_info.seed)
        i = 0
        while self.max_samples == -1 or i < self.max_samples:
            i += 1
            token_dict = self.tokenize(
                self.graph.generate_trajectory(self.seq_len // self.graph.digits)
            )
            if self.save_tokens:
                self.tokens.append(token_dict["input_ids"])
            yield token_dict

    def tokenize(self, traj):
        """Convert a trajectory to a tokenized input."""
        inp_str = self.graph.traj_to_str(traj)
        inp_dict = {
            k: v.reshape(-1)
            for k, v in self.tokenizer(inp_str, return_tensors="pt").items()
        }
        inp_dict["labels"] = inp_dict["input_ids"].clone()
        return inp_dict


class ToyModelTrainer(cb_trainer.CodebookTrainer):
    """Trainer for the toy model."""

    def __init__(self, toy_graph, gen_seq_len, *args, **kwargs):
        """Initialize the trainer.

        Args:
            toy_graph: The automata to generate trajectories from.
            gen_seq_len: The length of the trajectories to generate.
            args: Arguments to pass to the base trainer.
            kwargs: Keyword arguments to pass to the base trainer.
        """
        super().__init__(*args, **kwargs)
        self.toy_graph = toy_graph
        self.gen_seq_len = gen_seq_len

    def log(self, logs) -> None:
        """Additional metrics to log for the toy model.

        Logs the transition accuracy of the generated trajectories.
        """
        if all("eval_" in k for k in logs.keys()):
            metric_prefix = "eval_"
            gen_seq = self.model.generate(
                num_return_sequences=10,
                max_length=self.gen_seq_len,
                min_length=self.gen_seq_len,
                do_sample=True,
            )
            gen_seq = [self.tokenizer.decode(gen_seq[i]) for i in range(len(gen_seq))]
            traj = self.toy_graph.seq_to_traj(gen_seq)
            ov_trans_acc, first_trans_acc = self.toy_graph.transition_accuracy(traj)
            logs[f"{metric_prefix}transition_accuracy"] = ov_trans_acc
            logs[f"{metric_prefix}first_transition_accuracy"] = first_trans_acc
        super().log(logs)


def create_tokenizer(path, vocab_size):
    """Create a tokenizer for the toy model."""
    path = pathlib.Path(path)
    if not path.exists():
        path.mkdir()

    vocab_dict = {f"{i}": i for i in range(vocab_size - 1)}
    vocab_dict["<|endoftext|>"] = vocab_size - 1
    vocab = json.dumps(vocab_dict)
    print("Vocab:")
    print(vocab)
    with open(path / "vocab.json", "w") as f:
        f.write(vocab)

    with open(path / "merges.txt", "w") as f:
        f.write("")

    tokenizer = GPT2TokenizerFast(
        vocab_file=str(path / "vocab.json"),
        merges_file=str(path / "merges.txt"),
        pad_token="<|endoftext|>",
    )
    if not (path / "special_tokens_map.json").exists():
        tokenizer.save_pretrained(path)
    return tokenizer


def load_model(config_args, cfg_dict):
    """Loads the model based on the config."""
    if config_args.model_path is not None:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config_args.model_path
        )
    elif config_args.model_type == "gptneox":
        config = GPTNeoXConfig(
            vocab_size=config_args.vocab_size,
            hidden_size=config_args.hidden_size,
            num_hidden_layers=config_args.num_hidden_layers,
            num_attention_heads=config_args.num_attention_heads,
            intermediate_size=config_args.intermediate_size,
            rotary_emb_base=config_args.rotary_emb_base,
            bos_token_id=config_args.vocab_size - 1,
            eos_token_id=config_args.vocab_size - 1,
            max_position_embeddings=config_args.seq_len,
        )
        model = GPTNeoXForCausalLM(config=config)
    elif config_args.model_type == "gpt2":
        config = GPT2Config(
            vocab_size=config_args.vocab_size,
            n_embd=config_args.hidden_size,
            n_layer=config_args.num_hidden_layers,
            n_head=config_args.num_attention_heads,
            n_inner=config_args.intermediate_size,
            bos_token_id=config_args.vocab_size - 1,
            eos_token_id=config_args.vocab_size - 1,
            max_position_embeddings=config_args.seq_len,
        )
        model = GPT2LMHeadModel(config=config)
    else:
        raise ValueError(f"Unknown model type {config_args.model_type}")

    if cfg_dict["apply_codebook"]:
        cb_config = models.CodebookModelConfig(**cfg_dict["codebook_args"])
        model = models.wrap_codebook(model_or_path=model, config=cb_config)
        model.disable_logging()

    return model


@hydra.main(config_path="config", config_name="toy_main")
def main(cfg):
    """Train codebook based models parametrized using hydra.

    Args:
        cfg: hydra config.

    Returns: tuple of metrics for trained model and the baseline metrics.
    """
    training_args = run_clm.TrainingArguments(**cfg.training_args)
    training_args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    model_args = run_clm.ModelArguments(model_name_or_path="toy/model")
    config_args = ModelConfigArguments(**cfg.model_config_args)
    data_args = run_clm.DataTrainingArguments(
        dataset_name="toy_graph", max_eval_samples=2048
    )
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    flat_cfg_dict = pd.json_normalize(cfg_dict, sep="@").to_dict(orient="records")[0]
    flat_cfg_dict = {k.split("@")[-1]: v for k, v in flat_cfg_dict.items()}

    # prepare tags and wandb run name from tags
    tags = sorted(cfg.tags)
    for key in sorted(cfg.tag_keys):
        tags.append(f"{shortened_args[key]}: {flat_cfg_dict[key]}")
    if tags:
        cfg_dict["training_args"]["run_name"] = training_args.run_name = ", ".join(tags)

    automata = ToyGraph(
        N=cfg.toy_dataset_args.num_states,
        edges=cfg.toy_dataset_args.num_edges,
        seed=cfg.toy_dataset_args.seed,
    )
    tokenizer = create_tokenizer("toy/", config_args.vocab_size)
    train_dataset = ToyDataset(
        automata, tokenizer=tokenizer, seq_len=config_args.seq_len
    )
    eval_dataset = ToyDataset(
        automata,
        tokenizer=tokenizer,
        seq_len=config_args.seq_len,
        max_samples=cfg.toy_dataset_args.max_eval_samples,
    )

    model = load_model(config_args, cfg_dict)

    optimizers = (None, None)
    if isinstance(model, models.CodebookModel):
        if training_args.train_model_params:
            params = [
                {
                    "params": model.get_codebook_params(),
                    "lr": training_args.learning_rate,
                    # weight decay for codebook params is used through
                    # `codebook_weight_decay` param that is used directly
                    # to compute regularized loss.
                    "weight_decay": 0.0,
                },
                {
                    "params": model.get_model_params(),
                    "lr": training_args.model_lr_factor * training_args.learning_rate,
                    "weight_decay": training_args.weight_decay,
                },
            ]
        else:
            params = model.get_codebook_params()
        if len(params) > 0:
            optimizer = torch.optim.AdamW(
                params,
                training_args.learning_rate,
            )
            optimizers = (optimizer, None)

    callbacks = [cb_trainer.WandbCallback()]

    trainer = ToyModelTrainer(
        toy_graph=automata,
        gen_seq_len=config_args.seq_len,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=run_clm.compute_metrics,
        preprocess_logits_for_metrics=run_clm.preprocess_logits_for_metrics,
        optimizers=optimizers,
        callbacks=callbacks,
    )

    if training_args.local_rank <= 0:
        wandb.init(
            project="toy_graph",
            name=training_args.run_name,
            tags=tags,
            settings=wandb.Settings(code_dir="."),
            config=cfg_dict,
        )

    automata.save("toy")

    lm_datasets = {"train": train_dataset, "validation": eval_dataset}
    metrics = run_clm.run_trainer(
        model_args, data_args, training_args, trainer, lm_datasets, last_checkpoint=None
    )
    return metrics


if __name__ == "__main__":
    main()