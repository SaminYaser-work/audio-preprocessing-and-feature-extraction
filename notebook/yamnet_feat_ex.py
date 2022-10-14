import os
from IPython import display
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
import re
from tqdm import tqdm


Crema_Path = '../Datasets/Crema'
Ravdess_Path = '../Datasets/Ravdess/'
Savee_Path = '../Datasets/Savee/'
Tess_Path = '../Datasets/Tess/'

crema = []
for wav in os.listdir(Crema_Path):
    emotion = wav.partition(".wav")[0].split('_')
    if emotion[2] == 'SAD':
        crema.append(('sad', Crema_Path+'/'+wav))
    elif emotion[2] == 'ANG':
        crema.append(('angry', Crema_Path+'/'+wav))
    elif emotion[2] == 'DIS':
        crema.append(('disgust', Crema_Path+'/'+wav))
    elif emotion[2] == 'FEA':
        crema.append(('fear', Crema_Path+'/'+wav))
    elif emotion[2] == 'HAP':
        crema.append(('happy', Crema_Path+'/'+wav))
    elif emotion[2] == 'NEU':
        crema.append(('neutral', Crema_Path+'/'+wav))
    else:
        # crema.append(('unknown', Crema_Path+'/'+wav))
        raise ValueError('Invalid label in crema...')

Crema_df = pd.DataFrame.from_dict(crema)
Crema_df.rename(columns={0: 'Emotion', 1: 'File_Path'}, inplace=True)
Crema_df.to_csv('./features/crema.csv', index=False)

ravdess = []
for directory in os.listdir(Ravdess_Path):
    actors = os.listdir(os.path.join(Ravdess_Path, directory))
    for wav in actors:
        emotion = wav.partition('.wav')[0].split('-')
        emotion_number = int(emotion[2])
        ravdess.append(
            (emotion_number, os.path.join(Ravdess_Path, directory, wav)))
Ravdess_df = pd.DataFrame.from_dict(ravdess)
Ravdess_df.rename(columns={0: 'Emotion', 1: 'File_Path'}, inplace=True)
Ravdess_df['Emotion'].replace({1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad',
                              5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)
Ravdess_df.to_csv('./features/ravdess.csv', index=False)

savee = []
for wav in os.listdir(Savee_Path):
    emo = wav.partition('.wav')[0].split('_')[1].replace(r'[0-9]', '')
    emotion = re.split(r'[0-9]', emo)[0]
    if emotion == 'a':
        savee.append(('angry', Savee_Path+'/'+wav))
    elif emotion == 'd':
        savee.append(('disgust', Savee_Path+'/'+wav))
    elif emotion == 'f':
        savee.append(('fear', Savee_Path+'/'+wav))
    elif emotion == 'h':
        savee.append(('happy', Savee_Path+'/'+wav))
    elif emotion == 'n':
        savee.append(('neutral', Savee_Path+'/'+wav))
    elif emotion == 'sa':
        savee.append(('sad', Savee_Path+'/'+wav))
    elif emotion == 'su':
        savee.append(('surprise', Savee_Path+'/'+wav))
Savee_df = pd.DataFrame.from_dict(savee)
Savee_df.rename(columns={0: 'Emotion', 1: 'File_Path'}, inplace=True)
Savee_df.to_csv('./features/savee.csv', index=False)

tess = []
for directory in os.listdir(Tess_Path):
    for wav in os.listdir(os.path.join(Tess_Path, directory)):
        emotion = wav.partition('.wav')[0].split('_')
        if emotion[2] == 'ps':
            tess.append(('surprise', os.path.join(Tess_Path, directory, wav)))
        else:
            tess.append((emotion[2], os.path.join(Tess_Path, directory, wav)))
Tess_df = pd.DataFrame.from_dict(tess)
Tess_df.rename(columns={0: 'Emotion', 1: 'File_Path'}, inplace=True)
Tess_df.to_csv('./features/tess.csv', index=False)

Crema_df['Database'] = ['crema'] * Crema_df.shape[0]
Tess_df['Database'] = ['tess'] * Tess_df.shape[0]
Ravdess_df['Database'] = ['ravdess'] * Ravdess_df.shape[0]
Savee_df['Database'] = ['savee'] * Savee_df.shape[0]

main_df = pd.concat([Crema_df, Ravdess_df, Savee_df, Tess_df], axis=0)
main_df.to_csv('./features/all.csv', index=False)

# main_df = main_df[main_df['Emotion'] != 'surprise']

# classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# main_df['Emotion'] = main_df['Emotion'].apply(
#     lambda emotion: classes.index(emotion))
