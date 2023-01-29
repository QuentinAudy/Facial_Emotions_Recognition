import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy import signal
import os
import librosa
import librosa.display
import helpers
from IPython.display import clear_output, display
import soundfile as sf
import random



myAudio = "Test2.wav"



def saveWavSpectrogram(directory,Emotion):

    for myAudio in os.listdir(directory+'/'+Emotion):

        if myAudio != ".DS_Store":
            audioName = myAudio[0:-4]
            file_path = directory + '/' + Emotion + '/' + myAudio
            print(file_path)

            # Windowing
            n_fft = 2048
            hop_length = 512

            # Load audio file
            y, sr = librosa.load(file_path)

            yt, index = librosa.effects.trim(y, top_db=15)

            # Normalize between -1 and 1
            normalized_y = librosa.util.normalize(yt)

            # Compute STFT
            stft = librosa.core.stft(normalized_y, n_fft=n_fft, hop_length=hop_length)

            # Convert sound intensity to log amplitude:
            stft_db = librosa.amplitude_to_db(abs(stft))

            # Plot spectrogram from STFT
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(stft_db)
            plt.tight_layout()
            finalDirectory = "C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/Audioset/Audioset/Spectres/Train"
            plt.savefig(finalDirectory + "/" + Emotion + "/" + audioName + ".jpg", format = 'jpg')
            plt.close()


def saveWavSpectrogramMFCC(directory,Emotion):

    features = []
    labels = []

    for myAudio in os.listdir(directory+'/'+Emotion):

        if myAudio != ".DS_Store":
            audioName = myAudio[0:-4]
            file_path = directory + '/' + Emotion + '/' + myAudio
            print(file_path)

            # Windowing
            n_fft = 2048
            hop_length = 512

            # Load audio file
            y, sr = librosa.load(file_path)

            # Normalize between -1 and 1
            normalized_y = librosa.util.normalize(y)

            # Compute STFT
            stft = librosa.core.stft(normalized_y, n_fft=n_fft, hop_length=hop_length)

            n_mels = 128

            # Generate mel scaled spectrogram
            mel = librosa.feature.melspectrogram(S=stft, n_mels=n_mels)

            # Convert sound intensity to log amplitude:
            mel_db = librosa.amplitude_to_db(abs(mel))

            # Normalize between -1 and 1
            normalized_mel = librosa.util.normalize(mel_db)
            # Plot spectrogram from STFT
            # plt.figure(figsize=(12, 4))
            # librosa.display.specshow(mel_db)
            # plt.tight_layout()
            # finalDirectory = "C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/Audioset/Audioset/Spectres MEL/Train"
            # plt.savefig(finalDirectory + "/" + Emotion + "/" + audioName + ".jpg", format = 'jpg')
            # plt.close()

            # Iterate through all audio files and extract MFCC

            frames_max = 0
            counter = 0
            total_samples = 1251
            n_mels = 40


            # Extract Log-Mel Spectrograms (do not add padding)

            # Save current frame count
            num_frames = mel.shape[1]

            # Add row (feature / label)
            features.append(mel)
            labels.append(Emotion)



            # Update frames maximum
            if (num_frames > frames_max):
                    frames_max = num_frames

    X = np.array(features)
    y = np.array(labels)
    print(X,y)
    print(frames_max)
    np.save("C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/ferproject/data/X-mel_spec.txt", X)
    np.save("C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/ferproject/data/y-mel_spec.txt", y)


def MELcoefficients(directory):

    features = []
    labels = []
    frames_max = 228
    counter = 0
    total_samples = 0
    n_mels = 40

    for Emotion in os.listdir(directory):
        if Emotion != ".DS_Store":

            for file in os.listdir(directory+"/"+Emotion):
                if file != ".DS_Store":

                    total_samples+=1

                    file_path = (directory+"/"+Emotion+"/"+file)

                    class_label = Emotion

                    # Extract Log-Mel Spectrograms (do not add padding)
                    mels = helpers.get_mel_spectrogram(file_path, 0, n_mels=n_mels)

                    # Save current frame count
                    num_frames = mels.shape[1]

                    # Add row (feature / label)
                    features.append(mels)
                    labels.append(class_label)

                    # Update frames maximum
                    if (num_frames > frames_max):
                        frames_max = num_frames

                    # Notify update every N files
                    if (counter % 500 == 0):
                        print("Status: {}/{}".format(counter + 1, total_samples))


                    counter += 1
    print("Finished: {}/{}".format(counter, total_samples))

    # Add padding to features with less than frames than frames_max
    padded_features = helpers.add_padding(features, frames_max)

    # Verify shapes
    print("Raw features length: {}".format(len(features)))
    print("Padded features length: {}".format(len(padded_features)))
    print("Feature labels length: {}".format(len(labels)))

    # Convert features (X) and labels (y) to Numpy arrays
    X = np.array(padded_features)
    y = np.array(labels)

    np.save("data/X-mel_spec_augmented", X)
    np.save("data/y-mel_spec_augmented", y)


def timeStretching(directory):



    rates = [0.81, 1.07]
    total = 1214 * len(rates) / 3
    count = 0

    # Set your path to the dataset
    us8k_path = os.path.abspath('./UrbanSound8K')
    audio_path = os.path.join(us8k_path, 'audio')
    augmented_path = os.path.join(audio_path, 'augmented')

    # Metadata
    path = "C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/Audioset/Audioset/Full/Complete"
    augmented_path = "C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/Audioset/Audioset/Full/Augmented"

    verifier=0

    for Emotion in os.listdir(directory):
        if Emotion != ".DS_Store":

            for file in os.listdir(directory + "/" + Emotion):
                if file != ".DS_Store":
                    verifier+=1
                    if verifier%3 ==0:
                        for rate in rates:

                            curr_file_path = directory + '/' + Emotion + '/' + file
                            output_path = augmented_path + '/' + Emotion + '/' + file[0:-4] + "stretched-rate" + str(rate) + ".wav"


                            y, sr = librosa.load(curr_file_path)
                            y_changed = librosa.effects.time_stretch(y, rate=rate)
                            #librosa.output.write_wav(output_path, y_changed, sr)
                            sf.write(output_path, y_changed, sr)
                            count += 1

                            clear_output(wait=True)
                            print("Progress: {}/{}".format(count, total))

def pitchShifting(directory):

    tone_steps = [-1, 2]
    total = 1214 * len(tone_steps) / 3
    count = 0

    verifier = 0

    path = "C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/Audioset/Audioset/Full/Complete"
    augmented_path = "C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/Audioset/Audioset/Full/Augmented"


    for Emotion in os.listdir(directory):
        if Emotion != ".DS_Store":

            for file in os.listdir(directory + "/" + Emotion):
                if file != ".DS_Store":

                    verifier+=1
                    if verifier%3 ==1:

                        for tone_step in tone_steps:

                            curr_file_path = directory + '/' + Emotion + '/' + file
                            output_path = augmented_path + '/' + Emotion + '/' + file[0:-4] + "pitched-tone-step" + str(tone_step) + ".wav"


                            y, sr = librosa.load(curr_file_path)
                            y_changed = librosa.effects.pitch_shift(y, sr, n_steps=tone_step)
                            #librosa.output.write_wav(output_path, y_changed, sr)
                            sf.write(output_path, y_changed, sr)
                            count += 1

                            clear_output(wait=True)
                            print("Progress: {}/{}".format(count, total))


def add_noise(data):
    noise = np.random.rand(len(data))
    noise_amp = random.uniform(0.005, 0.008)
    data_noise = data + (noise_amp * noise)
    return data_noise

def noiseAdding(directory):

    total = 1214 * 2 / 3
    count = 0

    verifier = 0

    path = "C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/Audioset/Audioset/Full/Complete"
    augmented_path = "C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/Audioset/Audioset/Full/Augmented"

    for Emotion in os.listdir(directory):
        if Emotion != ".DS_Store":

            for file in os.listdir(directory + "/" + Emotion):
                if file != ".DS_Store":

                    verifier += 1
                    if verifier % 3 == 2:

                        for noise in range(2):
                            curr_file_path = directory + '/' + Emotion + '/' + file
                            output_path = augmented_path + '/' + Emotion + '/' + file[0:-4] + "noise" + str(
                                noise) + ".wav"

                            y, sr = librosa.load(curr_file_path)
                            y_changed = add_noise(y)
                            # librosa.output.write_wav(output_path, y_changed, sr)
                            sf.write(output_path, y_changed, sr)
                            count += 1

                            clear_output(wait=True)
                            print("Progress: {}/{}".format(count, total))

def main():

    # for i in range(10,25):
    #     directory = 'C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Ravdess database/Video_Speech_Actor_{}/Actor_{}'.format(i,i)
    #     for filename in os.listdir(directory) :
    #         print(filename)
    #         shutil.move(directory+'/'+str(filename),'C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Ravdess database')

    directory = 'C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/Audioset/Audioset/Full/Augmented'

    # for Emotion in os.listdir(directory):
    #     if Emotion != ".DS_Store":
    #         print(Emotion)
    #         saveWavSpectrogramMFCC(directory, Emotion)
    # return
    MELcoefficients(directory)
    # timeStretching(directory)
    # pitchShifting(directory)
    # noiseAdding(directory)

if __name__ == '__main__':
    main()