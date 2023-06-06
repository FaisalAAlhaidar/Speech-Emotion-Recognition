import json
import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def remove_silence(voice_file_path):
    a, b = librosa.load(voice_file_path)
    s = librosa.effects.split(a, top_db=25)
    r = []
    for d in s:
        r.extend(a[d[0]:d[1]])

    # Overwrite the original voice file with the removed silence one.
    sf.write(voice_file_path, r, b)


def convert_voice_files_to_JSON_dataset(voice_file_path="EmotionData", file_name="TrainingSet.json"):
    # Expected hierarchy:
    #   Emotions
    #       Angry
    #       Happy
    #       Neutral
    #       Sad
    #       Surprised
    dataset = {
        "classes": [],
        "labels": [],
        "ZCR": [],
        "Flatness": [],
        "SpectralCentroid": [],
        "F0": [],
        "Tempo": [],
        "Kurtosis": [],
        "Contrast": [],
        "MFCCs": [],
        "Chroma": [],
        "Mel": []
    }

    classes = ["Angry", "Surprise", "Happy", "Neutral", "Sad"]
    for i, (a, b, c) in enumerate(os.walk(voice_file_path)):
        if a is not voice_file_path and a is not file_name:
            j = 0
            for w in classes:
                if w in a:
                    j = classes.index(w)
            dataset["classes"].append(classes[j])
            for voice_file in c:
                path = a + "/" + voice_file
                remove_silence(path)
                MFCCs = 40
                feature_vector = voice_file_to_feature_vector(voice_file_path=path, number_of_MFCCs=MFCCs)
                dataset["labels"].append(i - 1)
                dataset["ZCR"].append(str(feature_vector[0]))
                dataset["Flatness"].append(str(feature_vector[1]))
                dataset["SpectralCentroid"].append(str(feature_vector[2]))
                dataset["F0"].append(str(feature_vector[3]))
                dataset["Tempo"].append(str(feature_vector[4]))
                dataset["Kurtosis"].append(str(feature_vector[5]))
                dataset["Contrast"].append(feature_vector[6:13])
                dataset["MFCCs"].append(feature_vector[13:54])
                dataset["Chroma"].append(feature_vector[54:67])
                dataset["Mel"].append(feature_vector[67:76])
    with open(file_name, "w") as writer:
        json.dump(dataset, writer, indent=4)


def voice_file_to_feature_vector(voice_file_path, number_of_MFCCs=40):
    voice_samples, sample_rate = librosa.load(voice_file_path)

    stft = np.abs(librosa.stft(voice_samples))
    MFCCs = number_of_MFCCs
    mfcc = np.mean(librosa.feature.mfcc(voice_samples, sr=sample_rate, n_mfcc=MFCCs).T, axis=0)
    flatness = np.mean(librosa.feature.spectral_flatness(voice_samples))
    mel = np.mean(librosa.feature.melspectrogram(y=voice_samples, sr=sample_rate).T, axis=0)
    mel_reduced = []

    # Reducing the number of mel values by averaging.
    for c in range(8):
        s = 0
        for j in range(16):
            s += mel[(16 * c) + j]
        mel_reduced.append(s / 16)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(voice_samples))
    temp = librosa.feature.spectral_centroid(voice_samples, sr=sample_rate)
    spectral_centroid = np.mean(temp)
    normalizer = StandardScaler()
    pitch = np.mean(
        normalizer.fit_transform(np.array(librosa.yin(voice_samples, fmin=50, fmax=500)).reshape(-1, 1)))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    tempo = np.mean(librosa.beat.tempo(voice_samples))
    kurtosis = np.mean(librosa.feature.spectral_bandwidth(voice_samples))
    feature_vector = [zcr, flatness, spectral_centroid, pitch, tempo, kurtosis]

    # Storing the string values instead of the numerical ones.
    feature_vector = [str(i) for i in feature_vector]

    # Appending the remaining features manually because they are represented as vectors.
    for i in [contrast, mfcc, chroma, mel_reduced]:
        for j in i:
            feature_vector.append(str(j))

    return feature_vector
