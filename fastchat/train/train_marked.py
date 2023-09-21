# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle, Conversation
from fastchat.model.model_adapter import get_conv_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = False


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    conversation_template_name: str = "zero_shot_marked"
    keep_begin_marker: bool = True
    keep_end_marker: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    conv_template: Conversation,
    keep_begin_marker: bool = True,
    keep_end_marker: bool = True,
) -> Dict:
    roles = {"human": conv_template.roles[0], "gpt": conv_template.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv_template.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv_template.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv_template.roles[j % 2], f"{i}"
            conv_template.append_message(role, sentence["value"])
        conversations.append(conv_template.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = torch.empty_like(input_ids)

    assert conv_template.sep_style == SeparatorStyle.COLON_SURROUND

    begin_output = tokenizer.convert_tokens_to_ids(conv_template.sep)
    end_output = tokenizer.convert_tokens_to_ids(conv_template.sep2)

    for target, inp in zip(targets, input_ids):
        is_inside_output = False

        # Keep track of output index explicitly because begin and end token might need to be filtered out
        out_idx = 0
        for in_idx in range(len(inp)):
            token = inp[in_idx].item()

            if token == begin_output:
                is_inside_output = True
                if keep_begin_marker:
                    inp[out_idx] = token
                    target[out_idx] = IGNORE_TOKEN_ID
                    out_idx += 1
            elif token == end_output:
                is_inside_output = False
                if keep_end_marker:
                    inp[out_idx] = token
                    target[out_idx] = token
                    out_idx += 1
            elif token == tokenizer.pad_token_id:
                # Padding is only on the right side --> the remaining positions can be set in one go after the loop
                break
            else:
                inp[out_idx] = token
                target[out_idx] = token if is_inside_output else IGNORE_TOKEN_ID
                out_idx += 1
        # If begin or end tokens are filtered out, some of the input/target tokens are not yet set
        inp[out_idx:] = tokenizer.pad_token_id
        target[out_idx:] = IGNORE_TOKEN_ID

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, preprocess_kwargs):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, **preprocess_kwargs)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, preprocess_kwargs):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.preprocess_kwargs = preprocess_kwargs

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, **self.preprocess_kwargs)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        conv_template: Conversation,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    preprocess_kwargs = {
        "conv_template": conv_template,
        "keep_begin_marker": data_args.keep_begin_marker,
        "keep_end_marker": data_args.keep_end_marker,
    }
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, preprocess_kwargs=preprocess_kwargs)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, preprocess_kwargs=preprocess_kwargs)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    conv_template = get_conv_template(data_args.conversation_template_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
        additional_special_tokens=[conv_template.sep, conv_template.sep2]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<pad>"]})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    model.resize_token_embeddings(len(tokenizer))

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, conv_template=conv_template)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
