# %%
import pandas as pd
import librosa
from transformers import Wav2Vec2FeatureExtractor

dses = ['savee', 'crema', 'tess', 'ravdess', 'bser', 'subesco']


def save_features(dses):

    for ds in dses:

        df = pd.read_csv('../Datasets/custom_db/df.csv')
        df = df[df['dataset'] == ds]
        df = df.rename(columns={'file': 'path'})
        df['path'] = df['path'].apply(
            lambda path: '../Datasets/custom_db/' + path[2:])

        print(df.shape)
        df

        # %%
        print("Labels: ", df["emotion"].unique())
        print()
        df.groupby("emotion").count()[["path"]]

        # %%
        save_path = "./"

        train_df = df[df['split'] == 'train']
        train_aug_df = df[(df['split'] == 'train') |
                          (df['split'] == 'augment')]
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']

        train_df = train_df.reset_index(drop=True)
        train_aug_df = train_aug_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(f"{save_path}/train.csv", sep="\t",
                        encoding="utf-8", index=False)
        train_aug_df.to_csv(f"{save_path}/train_aug.csv",
                            sep="\t", encoding="utf-8", index=False)
        val_df.to_csv(f"{save_path}/val.csv", sep="\t",
                      encoding="utf-8", index=False)
        test_df.to_csv(f"{save_path}/test.csv", sep="\t",
                       encoding="utf-8", index=False)

        print(train_aug_df.shape)
        print(train_df.shape)
        print(val_df.shape)
        print(test_df.shape)

        # %% [markdown]
        # ## Prepare Data for Training

        # %%
        # Loading the created dataset using datasets
        from datasets import load_dataset

        data_files = {
            "train": f"{save_path}train.csv",
            "validation": f"{save_path}val.csv",
        }

        data_files_aug = {
            "train": f"{save_path}train_aug.csv",
            "validation": f"{save_path}val.csv",
        }

        dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
        dataset_aug = load_dataset(
            "csv", data_files=data_files_aug, delimiter="\t", )
        train_dataset = dataset["train"]
        train_aug_dataset = dataset_aug["train"]
        eval_dataset = dataset["validation"]

        # %%
        # We need to specify the input and output column
        input_column = "path"
        output_column = "emotion"

        # %%
        # we need to distinguish the unique labels in our SER dataset
        label_list = train_dataset.unique(output_column)
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
        print(
            f"A classification problem with {num_labels} classes: {label_list}")

        # %% [markdown]
        # # Preprocess Data

        # %%
        model_name_or_path = "facebook/hubert-base-ls960"
        # model_name_or_path = 'facebook/hubert-large-ls960-ft'
        # model_name_or_path = "facebook/hubert-xlarge-ll60k" # Needs too much memory
        # model_name_or_path = "facebook/hubert-large-ll60k" # Need to try again

        pooling_mode = "mean"

        # %%
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name_or_path,)
        target_sampling_rate = feature_extractor.sampling_rate
        print(f"The target sampling rate: {target_sampling_rate}")

        # %%
        def speech_file_to_array_fn(path):
            speech_array, sampling_rate = librosa.load(path, sr=None)
            speech_array, _ = librosa.effects.trim(speech_array, top_db=25)

            if (sampling_rate != target_sampling_rate):
                raise ValueError(
                    f"Sampling rate mismatch between file and target sampling rate. {sampling_rate} != {target_sampling_rate}")

            return speech_array

        def label_to_id(label, label_list):
            if len(label_list) > 0:
                return label_list.index(label) if label in label_list else -1
            return label

        def preprocess_function(examples):
            speech_list = [speech_file_to_array_fn(
                path) for path in examples[input_column]]
            target_list = [label_to_id(label, label_list)
                           for label in examples[output_column]]

            result = feature_extractor(
                speech_list, sampling_rate=target_sampling_rate)
            result["labels"] = list(target_list)

            return result

        # %%
        train_dataset = train_dataset.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=4
        )

        train_aug_dataset = train_aug_dataset.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=4
        )

        eval_dataset = eval_dataset.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=4
        )

        # %%
        train_dataset.save_to_disk(f'../Datasets/hf_datasets/{ds}/base/train')
        train_aug_dataset.save_to_disk(
            f'../Datasets/hf_datasets/{ds}/base/train_aug')
        eval_dataset.save_to_disk(f'../Datasets/hf_datasets/{ds}/base/val')

        # %%
        test_dataset = load_dataset(
            "csv", data_files={"test": "./test.csv"}, delimiter="\t")["test"]
        test_dataset

        # %%
        # model_name_or_path = "m3hrdadfi/hubert-base-greek-speech-emotion-recognition"
        # config = AutoConfig.from_pretrained(model_name_or_path)
        # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        # model = HubertForSpeechClassification.from_pretrained(model_name_or_path).to(device)

        # %%
        def speech_file_to_array_fn(batch):
            speech_array, sampling_rate = librosa.load(batch["path"], sr=None)
            if sampling_rate != target_sampling_rate:
                raise ValueError(
                    f"Sampling rate mismatch between file and target sampling rate. {sampling_rate} != {target_sampling_rate}")
            batch["speech"] = speech_array
            return batch

        # %%
        test_dataset = test_dataset.map(speech_file_to_array_fn)

        # %%
        test_dataset.save_to_disk(f'../Datasets/hf_datasets/{ds}/base/test')


save_features(dses)
