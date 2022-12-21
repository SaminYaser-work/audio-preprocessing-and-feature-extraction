# # Uncomment this part if you want to setup your wandb project
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from datasets import load_dataset, load_metric
import librosa
from transformers import (
    Trainer,
    is_apex_available,
)
from torch import nn
from packaging import version
from typing import Any, Dict, Union
from transformers import TrainingArguments
from transformers import EvalPrediction
from transformers import Wav2Vec2FeatureExtractor
import transformers
from typing import Dict, List, Optional, Union
from transformers.models.hubert.modeling_hubert import (
    HubertPreTrainedModel,
    HubertModel
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
from transformers.file_utils import ModelOutput
import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import shutil
from datasets import load_from_disk
import sys
import os
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import wandb


%env WANDB_WATCH = all
%env WANDB_LOG_MODEL = 1

dses = ['savee', 'tess', 'ravdess', 'crema',
        'bser', 'subesco', 'all_en', 'all_es', 'all']
augs = [True, False]


def start_training():


!wandb login f3417b4f564d8a58bbdcd6731ab041c2f58210c6 - -relogin

# %%
wandb.init(
    project="hubert-ser_v2",
    entity="f00d"
)

# %%


# import torchaudio


# %%
# os.environ["WANDB_DISABLED"] = "true"

# %% [markdown]
# ## Prepare Data for Training

# %%
# We need to specify the input and output column
input_column = "path"
output_column = "emotion"

# %%
ds = 'savee'
aug = True
variant = 'base'

if not aug:
    ds = ds + '_no_aug'

model_name_or_path = "facebook/hubert-base-ls960"
# model_name_or_path = 'facebook/hubert-large-ls960-ft'
# model_name_or_path = "facebook/hubert-xlarge-ll60k" # Needs too much memory
# model_name_or_path = "facebook/hubert-large-ll60k" # Need to try again

pooling_mode = "mean"

# %%
# train_dir = f'../input/all-en-with-augment-hf/{ds}/{variant}/train'
# val_dir = f'../input/all-en-with-augment-hf/{ds}/{variant}/val'
# test_dir = f'../input/all-en-with-augment-hf/{ds}/{variant}/test'

train_dir = f'../input/hf-hubert-ser/hf_datasets/{ds}/{variant}/train'
val_dir = f'../input/hf-hubert-ser/hf_datasets/{ds}/{variant}/val'
test_dir = f'../input/hf-hubert-ser/hf_datasets/{ds}/{variant}/test'

# train_dir = f'../input/bn-hf-hubert/{ds}/{variant}/train'
# val_dir = f'../input/bn-hf-hubert/{ds}/{variant}/val'
# test_dir = f'../input/bn-hf-hubert/{ds}/{variant}/test'
print(train_dir)

# %%
# # we need to distinguish the unique labels in our SER dataset
# label_list = train_dataset.unique(output_column)
# label_list.sort()  # Let's sort it for determinism
# num_labels = len(label_list)
# print(f"A classification problem with {num_labels} classes: {label_list}")

if ds == 'crema' or ds == 'crema_no_aug':
    label_list = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
elif ds == 'bser' or ds == 'bser_no_aug':
    label_list = ['angry', 'happy', 'neutral', 'sad', 'surprise']
else:
    label_list = ['angry', 'disgust', 'fear',
                  'happy', 'neutral', 'sad', 'surprise']


num_labels = len(label_list)
num_labels

# %% [markdown]
# In order to preprocess the audio into our classification model, we need to set up the relevant HuBERT scale in this case `facebook/hubert-base-ls960` by [FaceBook HuBERT-Base](https://huggingface.co/facebook/hubert-base-ls960). To handle the context representations in any audio length we use a merge strategy plan (pooling mode) to concatenate that 3D representations into 2D representations.
#
# There are three merge strategies `mean`, `sum`, and `max`. In this example, we achieved better results on the mean approach. In the following, we need to initiate the config and the feature extractor from the Dimitris model.

# %%

# %%
# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)

# %%
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    model_name_or_path,)
target_sampling_rate = feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

# %% [markdown]
# # Preprocess Data

# %%


def speech_file_to_array_fn(path):
    #     speech_array, sampling_rate = torchaudio.load(path)
    #     resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    #     speech = resampler(speech_array).squeeze().numpy()

    speech_array, sampling_rate = librosa.load(path)
    resampler = librosa.resample(speech_array, orig_sr=sampling_rate,
                                 target_sr=target_sampling_rate, res_type="kaiser_best")
    return resampler


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1
    return label


def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(
        path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list)
                   for label in examples[output_column]]

    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result

# %%
# train_dataset = train_dataset.map(
#     preprocess_function,
#     batch_size=100,
#     batched=True,
#     num_proc=4
# )
# eval_dataset = eval_dataset.map(
#     preprocess_function,
#     batch_size=100,
#     batched=True,
#     num_proc=4
# )


# %%
train_dataset = load_from_disk(train_dir)
eval_dataset = load_from_disk(val_dir)

dirpath = Path('./test')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

shutil.copytree(test_dir, './test')

test_dataset = load_from_disk('./test')


# %%
# idx = 0
# print(f"Training input_values: {train_dataset[idx]['input_values']}")
# print(f"Training attention_mask: {train_dataset[idx]['attention_mask']}")
# print(f"Training labels: {train_dataset[idx]['labels']} - {train_dataset[idx]['emotion']}")

# %% [markdown]
# Great, now we've successfully read all the audio files, resampled the audio files to 16kHz, and mapped each audio to the corresponding label.

# %% [markdown]
# ## Model
#
# Before diving into the training part, we need to build our classification model based on the merge strategy.

# %%


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# %%

# from transformers.models.wav2vec2.modeling_wav2vec2 import (
#     Wav2Vec2PreTrainedModel,
#     Wav2Vec2Model
# )


class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(
            hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# %% [markdown]
# ## Training
#
# The data is processed so that we are ready to start setting up the training pipeline. We will make use of 🤗's [Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer) for which we essentially need to do the following:
#
# - Define a data collator. In contrast to most NLP models, XLSR-Wav2Vec2 has a much larger input length than output length. *E.g.*, a sample of input length 50000 has an output length of no more than 100. Given the large input sizes, it is much more efficient to pad the training batches dynamically meaning that all training samples should only be padded to the longest sample in their batch and not the overall longest sample. Therefore, fine-tuning XLSR-Wav2Vec2 requires a special padding data collator, which we will define below
#
# - Evaluation metric. During training, the model should be evaluated on the word error rate. We should define a `compute_metrics` function accordingly
#
# - Load a pretrained checkpoint. We need to load a pretrained checkpoint and configure it correctly for training.
#
# - Define the training configuration.
#
# After having fine-tuned the model, we will correctly evaluate it on the test data and verify that it has indeed learned to correctly transcribe speech.

# %% [markdown]
# ### Set-up Trainer
#
# Let's start by defining the data collator. The code for the data collator was copied from [this example](https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2/run_asr.py#L81).
#
# Without going into too many details, in contrast to the common data collators, this data collator treats the `input_values` and `labels` differently and thus applies to separate padding functions on them (again making use of XLSR-Wav2Vec2's context manager). This is necessary because in speech input and output are of different modalities meaning that they should not be treated by the same padding function.
# Analogous to the common data collators, the padding tokens in the labels with `-100` so that those tokens are **not** taken into account when computing the loss.

# %%


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature_extractor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(
            label_features[0], int) else torch.float

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch


# %%
data_collator = DataCollatorCTCWithPadding(
    feature_extractor=feature_extractor, padding=True)

# %% [markdown]
# Next, the evaluation metric is defined. There are many pre-defined metrics for classification/regression problems, but in this case, we would continue with just **Accuracy** for classification and **MSE** for regression. You can define other metrics on your own.

# %%
is_regression = False

# %%


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

# %% [markdown]
# Now, we can load the pretrained checkpoint into our classification model with a pooling strategy.


# %%
drive_dir = "./"
!ls {drive_dir}
output_dir = os.path.join(drive_dir, "ckpts", "hubert-base-english-ser")
!ls {output_dir}

# %%

last_checkpoint = None
checkpoints = []
if os.path.exists(output_dir):
    for subdir in os.scandir(output_dir):
        if subdir.is_dir():
            checkpoints.append(subdir.path)


if len(checkpoints) > 0:
    checkpoints = list(sorted(checkpoints, key=lambda ckpt: ckpt.split(
        '/')[-1].split('-')[-1], reverse=True))
    model_name_or_path = os.path.join("./", checkpoints[0].split("/")[-1])
    last_checkpoint = model_name_or_path
    shutil.copytree(checkpoints[0], model_name_or_path)

# %%
print(f"model_name_or_path: {model_name_or_path}")
print(f"last_checkpoint: {last_checkpoint}")

# %%
model = HubertForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

# %% [markdown]
# The first component of XLSR-Wav2Vec2 consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal. This part of the model has already been sufficiently trained during pretraining and as stated in the [paper](https://arxiv.org/pdf/2006.13979.pdf) does not need to be fine-tuned anymore.
# Thus, we can set the `requires_grad` to `False` for all parameters of the *feature extraction* part.

# %%
model.freeze_feature_extractor()

# %% [markdown]
# In a final step, we define all parameters related to training.
# To give more explanation on some of the parameters:
# - `learning_rate` and `weight_decay` were heuristically tuned until fine-tuning has become stable. Note that those parameters strongly depend on the Common Voice dataset and might be suboptimal for other speech datasets.
#
# For more explanations on other parameters, one can take a look at the [docs](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer#trainingarguments).
#
# **Note**: If one wants to save the trained models in his/her google drive the commented-out `output_dir` can be used instead.

# %%

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    save_strategy="steps",
    num_train_epochs=5.0,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    save_total_limit=1,
    do_train=True,
    do_eval=True,
    do_predict=True,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
    #     push_to_hub = True
)

# %% [markdown]
# For future use we can create our training script, we do it in a simple way. You can add more on you own.

# %%


if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

#         model.train()
#         inputs = self._prepare_inputs(inputs)

#         if self.use_amp:
#             with autocast():
#                 loss = self.compute_loss(model, inputs)
#         else:
#             loss = self.compute_loss(model, inputs)

#         if self.args.gradient_accumulation_steps > 1:
#             loss = loss / self.args.gradient_accumulation_steps

#         if self.use_amp:
#             self.scaler.scale(loss).backward()
#         elif self.use_apex:
#             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
#                 scaled_loss.backward()
#         elif self.deepspeed:
#             self.deepspeed.backward(loss)
#         else:
#             loss.backward()

#         return loss.detach()

        model.train()
        inputs = self._prepare_inputs(inputs)

        with autocast():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        return loss.detach()


# %% [markdown]
# Now, all instances can be passed to Trainer and we are ready to start training!

# %%
trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor,
)

# %% [markdown]
# ### Training

# %%
if training_args.do_train:
    print(f"last_checkpoint: {last_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    feature_extractor.save_pretrained(training_args.output_dir)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# %% [markdown]
# ## Evaluation

# %%

# %%
# test_dataset = load_dataset("csv", data_files={"test": "./test.csv"}, delimiter="\t")["test"]
# test_dataset

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %%
# model_name_or_path = "m3hrdadfi/hubert-base-greek-speech-emotion-recognition"
# config = AutoConfig.from_pretrained(model_name_or_path)
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
# model = HubertForSpeechClassification.from_pretrained(model_name_or_path).to(device)

# %%


def speech_file_to_array_fn(batch):
    #     speech_array, sampling_rate = torchaudio.load(batch["path"])
    #     speech_array = speech_array.squeeze().numpy()
    speech_array, sampling_rate = librosa.load(batch["path"])
    speech_array = librosa.resample(
        speech_array, sampling_rate, feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = feature_extractor(
        batch["speech"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

# %%
# test_dataset = test_dataset.map(speech_file_to_array_fn)


# %%
result = test_dataset.map(predict, batched=True, batch_size=2)

# %%
label_names = [config.id2label[i] for i in range(config.num_labels)]
label_names

# %%
y_true = [config.label2id[name] for name in result["emotion"]]
y_pred = result["predicted"]

# print(y_true[:5])
# print(y_pred[:5])

# %%
print(classification_report(y_true, y_pred, target_names=label_names, digits=6))

# %%

print('f1_weighted:', f1_score(y_true, y_pred, average='weighted'))
print('Pre_weighted:', precision_score(y_true, y_pred, average='weighted'))
print('Re_weighted:', recall_score(y_true, y_pred, average='weighted'))
print('Acc:', accuracy_score(y_true, y_pred))

# %%

cm = confusion_matrix(y_true, y_pred)
print('cm:', cm)
cm = cm / cm.astype(float).sum(axis=1)
# print('cm_norm:', cm)

ax = plt.subplot()
# annot=True to annotate cells, ftm='g' to disable scientific notation
sns.heatmap(cm, annot=True, fmt='.2g', ax=ax)

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(label_names)
ax.yaxis.set_ticklabels(label_names)

# %%
# wandb.alert(
#     title="Done",
#     text=f"done with {ds}"
# )

# %%
# trainer.save_model("./hubert")

# %%
# shutil.make_archive('hubert_all_7_aug_large', 'zip', './hubert')

# %%
stop

# %%
wandb.finish()

# %%
!rm - rf / kaggle/working/*
