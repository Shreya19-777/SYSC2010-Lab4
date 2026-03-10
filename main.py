import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

#**********************Section 5 : Low Pass Filtering*****************************

#Base data read from the file
df = pd.read_csv("ecg_signal.csv")
t = np.array(df['time'])
signal_data = np.array(df['signal'])

#Values used for filtering
fs = 250
cutoff_freq = 40
nyquist_f = 0.5 * fs
normal_cutoff = cutoff_freq / nyquist_f

'''
#Unfiltered plot
plt.figure()
plt.title("Unfiltered Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t, signal_data, label="Noisy Signal")
plt.show()

#Low pass filter
b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)
lp_filtered_ecg = signal.filtfilt(b, a, signal_data)

plt.figure()
plt.title("Filtered signal using low pass filter")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t, lp_filtered_ecg)
plt.show()

#Overlay of plots
plt.figure()
plt.plot(t, signal_data, label='Noisy signal', alpha=0.4, color='red')
plt.plot(t, lp_filtered_ecg, label='Low Pass Filtered signal', alpha=0.4, color='blue')
plt.title("Comparison of filtered and non-filtered signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

#**********************Section 6 : High Pass Filtering*****************************
hp_cutoff_freq = 0.5

#High pass filter
normal_cutoff = hp_cutoff_freq/nyquist_f
b, a = signal.butter(5, normal_cutoff, btype='high', analog=False)
hp_filtered_ecg = signal.filtfilt(b, a, signal_data)

plt.figure()
plt.title("Filtered signal using high pass filter")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t, hp_filtered_ecg)
plt.show()
'''

#**********************Section 7 : Band Pass Filtering*****************************
#0.5 to 40 Hz
minimum = 0.5
high = 40

sos = signal.butter(10, [minimum, high], 'band', analog=False)
bp_filtered_ecg = signal.sosfiltfilt(sos, signal_data)

plt.figure()
plt.title("Filtered signal using high pass filter")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t, bp_filtered_ecg, color='green')
plt.show()