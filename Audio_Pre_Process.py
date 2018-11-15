# Script to process audio data and convert to X matrix (X_Data)
# for CSCI 550 final project
# cd /Users/JeremyTate/Documents/School/CSCI/CSCI\ 550/Final\ Project/
# python Audio_Pre_Process.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.io import wavfile

filename = "Audio2.wav"

# Sampling rate and amplitude data for L and R channels
samp_rate, amplitude = wavfile.read(filename)

# Convert amplitude data to actual number of samples from video
# Taking 1 channel only
num_obs = 623 # to match number of video observations

# number of attributes
width = 100
num_raw_obs = int(amplitude.shape[0]/width)
X_Raw = np.zeros((num_raw_obs, width))
# Samples signal many times to obtain frequencies via FFT
for i in range(num_raw_obs):
    start = i*width
    end = start+width
    temp = amplitude[start:end,0]
    X_Raw[i] = np.abs(fftpack.fft(temp))

# X_Raw has way more observations than the video data so we need to
# condense the observations down to only num_obs
X = np.zeros((num_obs, width))
avg_chunk_w = float(num_raw_obs/num_obs)
for i in range(num_obs):
    start = int(i*avg_chunk_w)
    end = int((i+1)*avg_chunk_w)
    for j in range(width):
        X[i, j] = np.average(X_Raw[start:end, j])

# Export data as csv
X_Data = pd.DataFrame(X)
X_Data.to_csv("X_Data.csv", header=False, index=False)
