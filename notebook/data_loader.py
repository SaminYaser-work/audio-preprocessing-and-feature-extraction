'''
@author: Samin Yaser
@description: This file contains the data loader class for the four english datasets
'''

import os
import librosa
import resampy
import soundfile as sf
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class data_loader:

    CLASS6 = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    CLASS7 = ['angry', 'disgust', 'fear',
              'happy', 'neutral', 'sad', 'surprise']

    def __init__(self, Crema_path, Ravdess_path, Savee_path, Tess_path):
        self.Crema_Path = Crema_path
        self.Ravdess_Path = Ravdess_path
        self.Savee_Path = Savee_path
        self.Tess_Path = Tess_path

    def get_crema_df(self):
        crema = []
        for wav in os.listdir(self.Crema_Path):
            emotion = wav.partition(".wav")[0].split('_')
            if emotion[2] == 'SAD':
                crema.append(('sad', self.Crema_Path+'/'+wav))
            elif emotion[2] == 'ANG':
                crema.append(('angry', self.Crema_Path+'/'+wav))
            elif emotion[2] == 'DIS':
                crema.append(('disgust', self.Crema_Path+'/'+wav))
            elif emotion[2] == 'FEA':
                crema.append(('fear', self.Crema_Path+'/'+wav))
            elif emotion[2] == 'HAP':
                crema.append(('happy', self.Crema_Path+'/'+wav))
            elif emotion[2] == 'NEU':
                crema.append(('neutral', self.Crema_Path+'/'+wav))
            else:
                # crema.append(('unknown', Crema_Path+'/'+wav))
                raise ValueError('Invalid label in crema...')

        Crema_df = pd.DataFrame.from_dict(crema)
        Crema_df.rename(columns={0: 'Emotion', 1: 'File_Path'}, inplace=True)

        return Crema_df

    def get_ravdess_df(self):
        ravdess = []
        for directory in os.listdir(self.Ravdess_Path):
            actors = os.listdir(os.path.join(self.Ravdess_Path, directory))
            for wav in actors:
                emotion = wav.partition('.wav')[0].split('-')
                emotion_number = int(emotion[2])
                ravdess.append(
                    (emotion_number, os.path.join(self.Ravdess_Path, directory, wav)))
        Ravdess_df = pd.DataFrame.from_dict(ravdess)
        Ravdess_df.rename(columns={0: 'Emotion', 1: 'File_Path'}, inplace=True)
        Ravdess_df['Emotion'].replace({1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad',
                                       5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)

        return Ravdess_df

    def get_savee_df(self):
        savee = []
        for wav in os.listdir(self.Savee_Path):
            emo = wav.partition('.wav')[0].split('_')[1].replace(r'[0-9]', '')
            emotion = re.split(r'[0-9]', emo)[0]
            if emotion == 'a':
                savee.append(('angry', self.Savee_Path+'/'+wav))
            elif emotion == 'd':
                savee.append(('disgust', self.Savee_Path+'/'+wav))
            elif emotion == 'f':
                savee.append(('fear', self.Savee_Path+'/'+wav))
            elif emotion == 'h':
                savee.append(('happy', self.Savee_Path+'/'+wav))
            elif emotion == 'n':
                savee.append(('neutral', self.Savee_Path+'/'+wav))
            elif emotion == 'sa':
                savee.append(('sad', self.Savee_Path+'/'+wav))
            elif emotion == 'su':
                savee.append(('surprise', self.Savee_Path+'/'+wav))
        Savee_df = pd.DataFrame.from_dict(savee)
        Savee_df.rename(columns={0: 'Emotion', 1: 'File_Path'}, inplace=True)
        return Savee_df

    def get_tess_df(self):
        tess = []
        for directory in os.listdir(self.Tess_Path):
            for wav in os.listdir(os.path.join(self.Tess_Path, directory)):
                emotion = wav.partition('.wav')[0].split('_')
                if emotion[2] == 'ps':
                    tess.append(
                        ('surprise', os.path.join(self.Tess_Path, directory, wav)))
                else:
                    tess.append(
                        (emotion[2], os.path.join(self.Tess_Path, directory, wav)))
        Tess_df = pd.DataFrame.from_dict(tess)
        Tess_df.rename(columns={0: 'Emotion', 1: 'File_Path'}, inplace=True)
        return Tess_df

    def get_all_df(self):
        Crema_df = self.get_crema_df()
        Tess_df = self.get_tess_df()
        Ravdess_df = self.get_ravdess_df()
        Savee_df = self.get_savee_df()

        Crema_df['Database'] = ['crema'] * Crema_df.shape[0]
        Tess_df['Database'] = ['tess'] * Tess_df.shape[0]
        Ravdess_df['Database'] = ['ravdess'] * Ravdess_df.shape[0]
        Savee_df['Database'] = ['savee'] * Savee_df.shape[0]

        main_df = pd.concat([Crema_df, Ravdess_df, Savee_df, Tess_df], axis=0)

        return main_df

    def __get_ds(self, ds: str):
        if ds == 'crema':
            return self.get_crema_df()
        elif ds == 'ravdess':
            return self.get_ravdess_df()
        elif ds == 'savee':
            return self.get_savee_df()
        elif ds == 'tess':
            return self.get_tess_df()
        elif ds == 'all':
            return self.get_all_df()
        else:
            raise ValueError('Invalid dataset name...')

    def split_numpy(self, data, label, ratio_train=0.7, ratio_val=0.15, ratio_test=0.15):
        if ratio_train + ratio_val + ratio_test != 1:
            raise ValueError('Train, validation and test ratios must sum to 1')

        # Produces test split.
        remaining_data,  test_data, remaining_label, test_label = train_test_split(
            data, label, test_size=ratio_test, stratify=label, shuffle=True)

        # Adjusts val ratio, w.r.t. remaining dataset.
        ratio_remaining = 1 - ratio_test
        ratio_val_adjusted = ratio_val / ratio_remaining

        # Produces train and val splits.
        train_data, val_data, train_label, val_label = train_test_split(
            remaining_data, remaining_label,
            test_size=ratio_val_adjusted,
            stratify=remaining_label, shuffle=True
        )

        return train_data, train_label, val_data, val_label, test_data, test_label

    def split_df(self, d: pd.DataFrame, ratio_train=0.7, ratio_val=0.15, ratio_test=0.15):

        if ratio_train + ratio_val + ratio_test != 1:
            raise ValueError('Train, validation and test ratios must sum to 1')

        # Produces test split.
        remaining, test = train_test_split(
            d, test_size=ratio_test, stratify=d['Emotion'], shuffle=True)

        # Adjusts val ratio, w.r.t. remaining dataset.
        ratio_remaining = 1 - ratio_test
        ratio_val_adjusted = ratio_val / ratio_remaining

        # Produces train and val splits.
        train, val = train_test_split(
            remaining, test_size=ratio_val_adjusted, stratify=remaining['Emotion'], shuffle=True)

        return train, val, test

    def ohe_labels(self, train_label, val_label, test_label):
        train_label = train_label.reshape(-1, 1)
        val_label = val_label.reshape(-1, 1)
        test_label = test_label.reshape(-1, 1)

        encoder = OneHotEncoder()

        train_label = encoder.fit_transform(train_label).toarray()
        val_label = encoder.transform(val_label).toarray()
        test_label = encoder.transform(test_label).toarray()

        return encoder, train_label, val_label, test_label

    def get_wav_data(self, file_name, rate=16000):
        wav_data, sr = sf.read(file_name, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        waveform = waveform.astype('float32')

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if sr != rate:
            waveform = resampy.resample(waveform, sr, rate)

        return waveform

    def pad(self, data, max_len):
        data, _ = librosa.effects.trim(data, top_db=25)
        if data.shape[0] < max_len:
            data = np.pad(
                data, (0, max_len - data.shape[0]), 'constant')
        else:
            data = data[:max_len]

    def get_numpy(self, ds: str, pad=False, max_len=42000):
        df = self.__get_ds(ds)

        X = []
        y = []

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            # data, _ = librosa.load(row['File_Path'], sr=16000)
            data = self.get_wav_data(row['File_Path'])

            if pad:
                self.pad(data, max_len)

            X.append(data)
            y.append(row['Emotion'])

        # X = np.array(X)
        # y = np.array(y)

        return X, y
