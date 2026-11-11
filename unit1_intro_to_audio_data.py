import librosa
import matplotlib.pyplot as plt
import librosa.display

# Loading the audio:
array, sampling_rate = librosa.load(librosa.ex("trumpet"))

# Forme d'onde:
plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr= sampling_rate)
# plt.show()


# Spectre Frequentiel:
import numpy as np

dft_input = array[: 4096]

#  calculate the DFT:
window = np.hanning(len(dft_input))
windowed_input = dft_input * window
dft = np.fft.rfft(windowed_input)

# Get the amplitude in db:
amplitude = np.abs(dft)
amplitude_db = librosa.amplitude_to_db(amplitude, ref = np.max)

# Get the frequencies:
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))

plt.figure().set_figwidth(12)
plt.plot(frequency, amplitude_db)
plt.xlabel("Freq (Hz)")
plt.ylabel("Amplitude (dB)")
plt.xscale("log")

#  Spectrogram:
D = librosa.stft(array)
S_db = librosa.amplitude_to_db(np.abs(D), ref = np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(S_db, x_axis = "time", y_axis = "hz")
plt.colorbar()


# Mel Spectrogram:
S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels= 128 , fmax= 8000 ) 
S_dB = librosa.power_to_db(S, ref=np. max ) 

plt.figure().set_figwidth( 12 ) 
librosa.display.specshow(S_dB, x_axis= "time" , y_axis= "mel" , sr=sampling_rate, fmax= 8000 ) 
plt.colorbar()

plt.show()