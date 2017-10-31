# Z

This repository includes a Matlab class and a Python module which implement several methods/functions for audio signal processing.

- [z Matlab class](#z-matlab-class)
- [z Python module](#z-python-module)
- [Author](#author)

## z Matlab class

z This class implements several methods for audio signal processing.

z Methods:
- [stft - Short-time Fourier transform (STFT)](#stft-short-time-fourier-transform-stft)
- [istft - Inverse STFT](#istft-inverse-short-time-fourier-transform-stft)
- [cqtkernel - Constant-Q transform (CQT) kernel](#cqtkernel-constant-q-transform-cqt-kernel)
- [cqtspectrogram - CQT spectrogram using a CQT kernel](#cqtspectrogram-constant-q-transform-cqt-spectrogram-using-a-cqt-kernel)
- [cqtchromagram - CQT chromagram using a CQT kernel](#cqtchromagram-constant-q-transform-cqt-chromagram-using-a-cqt-kernel)
- [mfcc - Mel frequency cepstrum coefficients (MFCCs)](#mfcc-mel-frequency-cepstrum-coefficients-mfccs)
- [dct - Discrete cosine transform (DCT) using the fast Fourier transform (FFT)](#dct-discrete-cosine-transform-dct-using-the-fast-fourier-transform-fft)
- [dst - Discrete sine transform (DST) using the FFT](#dst-discrete-sine-transform-dst-using-the-fast-fourier-transform-fft)
- [mdct - Modified discrete cosine transform (MDCT)](#mdct-modified-discrete-cosine-transform-mdct-using-the-dct-iv)
- [imdct - Inverse MDCT using the DCT-IV](#imdct-inverse-modified-discrete-cosine-transform-mdct-using-the-dct-iv)

### stft [Short-time Fourier transform (STFT)](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
`audio_stft = z.stft(audio_signal,window_function,step_length);`
    
Arguments:
```
audio_signal: audio signal [number_samples,1]
window_function: window function [window_length,1]
step_length: step length in samples
audio_stft: audio stft [window_length,number_frames]
```

Example: Compute and display the spectrogram of an audio file
```
% Audio signal averaged over its channels and sample rate in Hz
[audio_signal,sample_rate] = audioread('audio_file.wav');
audio_signal = mean(audio_signal,2);

% Window duration in seconds (audio is stationary around 40 milliseconds)
window_duration = 0.04;

% Window length in samples (power of 2 for fast FFT and constant overlap-add (COLA))
window_length = 2^nextpow2(window_duration*sample_rate);

% Window function (periodic Hamming window for COLA)
window_function = hamming(window_length,'periodic');

% Step length in samples (half the window length for COLA)
step_length = window_length/2;

% Magnitude spectrogram (without the DC component and the mirrored frequencies)
audio_stft = z.stft(audio_signal,window_function,step_length);
audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));

% Spectrogram displayed in dB, s, and kHz
figure
imagesc(db(audio_spectrogram))
axis xy
colormap(jet)
title('Spectrogram (dB)')
xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
xticklabels(1:floor(length(audio_signal)/sample_rate))
xlabel('Time (s)')
yticks(round((1e3:1e3:sample_rate/2)/sample_rate*window_length))
yticklabels(1:sample_rate/2*1e-3)
ylabel('Frequency (kHz)')
```

### istft Inverse short-time Fourier transform (STFT)
`audio_signal = z.istft(audio_stft,window_function,step_length);`

Arguments:
```
audio_stft: audio STFT [window_length,number_frames]
window_function: window function [window_length,1]
step_length: step length in samples
audio_signal: audio signal [number_samples,1]
```

Example: Estimate the center and sides signals of a stereo audio file
```
% Stereo signal and sample rate in Hz
[audio_signal,sample_rate] = audioread('audio_file.wav');

% Parameters for the STFT
window_duration = 0.04;
window_length = 2^nextpow2(window_duration*sample_rate);
window_function = hamming(window_length,'periodic');
step_length = window_length/2;

% STFT of the left and right channels
audio_stft1 = z.stft(audio_signal(:,1),window_function,step_length);
audio_stft2 = z.stft(audio_signal(:,2),window_function,step_length);

% Magnitude spectrogram (with DC component) of the left and right channels
audio_spectrogram1 = abs(audio_stft1(1:window_length/2+1,:));
audio_spectrogram2 = abs(audio_stft2(1:window_length/2+1,:));

% Time-frequency mask of the left and right channels of the center signal
center_mask1 = min(audio_spectrogram1,audio_spectrogram2)./audio_spectrogram1;
center_mask2 = min(audio_spectrogram1,audio_spectrogram2)./audio_spectrogram2;

% STFT of the left and right channels of the center signal (with extension to mirrored frequencies)
center_stft1 = cat(1,center_mask1,flipud(center_mask1(2:end-1,:))).*audio_stft1;
center_stft2 = cat(1,center_mask2,flipud(center_mask2(2:end-1,:))).*audio_stft2;

% Synthesized signals of the left and right channels of the center signal
center_signal1 = z.istft(center_stft1,window_function,step_length);
center_signal2 = z.istft(center_stft2,window_function,step_length);

% Finalized stereo center and sides signals
center_signal = cat(2,center_signal1,center_signal2);
center_signal = center_signal(1:length(audio_signal),:);
sides_signal = audio_signal-center_signal;

% Synthesized center and side signals
audiowrite('center_signal.wav',center_signal,sample_rate);
audiowrite('sides_signal.wav',sides_signal,sample_rate);
```

### cqtkernel [Constant-Q transform (CQT)](https://en.wikipedia.org/wiki/Constant-Q_transform) kernel
`cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);`

Arguments:
```
sample_rate: sample rate in Hz
frequency_resolution: frequency resolution in number of frequency channels per semitone
minimum_frequency: minimum frequency in Hz
maximum_frequency: maximum frequency in Hz
cqt_kernel: CQT kernel [number_frequencies,fft_length]
```

Example: Compute and display the CQT kernel
```
% CQT kernel parameters
sample_rate = 44100;
frequency_resolution = 2;
minimum_frequency = 55;
maximum_frequency = sample_rate/2;

% CQT kernel
cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);

% Magnitude CQT kernel displayed
figure
imagesc(abs(cqt_kernel))
axis xy
colormap(jet)
title('Magnitude CQT kernel')
xlabel('FFT length')
ylabel('CQT frequency')
```

### cqtspectrogram Constant-Q transform (CQT) spectrogram using a CQT kernel
`audio_spectrogram = z.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);`

Arguments:
```
audio_signal: audio signal [number_samples,1]
sample_rate: sample rate in Hz
time_resolution: time resolution in number of time frames per second
cqt_kernel: CQT kernel [number_frequencies,fft_length]
audio_spectrogram: audio spectrogram in magnitude [number_frequencies,number_times]
```
Example: Compute and display the CQT spectrogram
```
% Audio file averaged over the channels and sample rate in Hz
[audio_signal,sample_rate] = audioread('audio_file.wav');
audio_signal = mean(audio_signal,2);

% CQT kernel
frequency_resolution = 2;
minimum_frequency = 55;
maximum_frequency = 3520;
cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);

% CQT spectrogram
time_resolution = 25;
audio_spectrogram = z.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);

% CQT spectrogram displayed in dB, s, and semitones
figure
imagesc(db(audio_spectrogram))
axis xy
colormap(jet)
title('CQT spectrogram (dB)')
xticks(round((1:floor(length(audio_signal)/sample_rate))*time_resolution))
xticklabels(1:floor(length(audio_signal)/sample_rate))
xlabel('Time (s)')
yticks(1:12*frequency_resolution:6*12*frequency_resolution)
yticklabels({'A1 (55 Hz)','A2 (110 Hz)','A3 (220 Hz)','A4 (440 Hz)','A5 (880 Hz)','A6 (1760 Hz)'})
ylabel('Frequency (semitones)')
```

### cqtchromagram Constant-Q transform (CQT) [chromagram](https://en.wikipedia.org/wiki/Chroma_feature) using a CQT kernel
`audio_chromagram = z.cqtchromagram(audio_signal,sample_rate,time_resolution,frequency_resolution,cqt_kernel);`

Arguments:
```
audio_signal: audio signal [number_samples,1]
sample_rate: sample rate in Hz
time_resolution: time resolution in number of time frames per second
frequency_resolution: frequency resolution in number of frequency channels per semitones
cqt_kernel: CQT kernel [number_frequencies,fft_length]
audio_chromagram: audio chromagram [number_chromas,number_times]
```

Example: Compute and display the CQT chromagram
```
% Audio file averaged over the channels and sample rate in Hz
[audio_signal,sample_rate] = audioread('audio_file.wav');
audio_signal = mean(audio_signal,2);

% CQT kernel
frequency_resolution = 2;
minimum_frequency = 55;
maximum_frequency = 3520;
cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);

% CQT chromagram
time_resolution = 25;
audio_chromagram = z.cqtchromagram(audio_signal,sample_rate,time_resolution,frequency_resolution,cqt_kernel);

% CQT chromagram displayed in dB, s, and chromas
figure
imagesc(db(audio_chromagram))
axis xy
colormap(jet)
title('CQT chromagram (dB)')
xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/time_resolution))
xticklabels(1:floor(length(audio_signal)/sample_rate))
xlabel('Time (s)')
yticks(1:frequency_resolution:12*frequency_resolution)
yticklabels({'A','A#','B','C','C#','D','D#','E','F','F#','G','G#'})
ylabel('Chroma')
```

### mfcc [Mel frequency cepstrum coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
`audio_mfcc = z.mfcc(audio_signal,sample_rate,number_filters,number_coefficients);`

Arguments:
```
audio_signal: audio signal [number_samples,1]
sample_rate: sample rate in Hz
number_filters: number of filters
number_coefficients: number of coefficients (without the 0th coefficient)
audio_mfcc: audio MFCCs [number_times,number_coefficients]
```

Example: Compute and display the MFCCs, delta MFCCs, and delta-detla MFCCs
```
% Audio signal averaged over its channels and sample rate in Hz
[audio_signal,sample_rate] = audioread('audio_file.wav');
audio_signal = mean(audio_signal,2);

% MFCCs for a given number of filters and coefficients
number_filters = 40;
number_coefficients = 20;
audio_mfcc = z.mfcc(audio_signal,sample_rate,number_filters,number_coefficients);

% Delta and delta-delta MFCCs
audio_deltamfcc = diff(audio_mfcc,1,2);
audio_deltadeltamfcc = diff(audio_deltamfcc,1,2);

% MFCCs, delta MFCCs, and delta-delta MFCCs displayed in s
step_length = (2^nextpow2(0.04*sample_rate))/2;
figure
subplot(3,1,1)
plot(audio_mfcc')
title('MFCCs')
xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
xticklabels(1:floor(length(audio_signal)/sample_rate))
xlabel('Time (s)')
axis tight
subplot(3,1,2)
plot(audio_deltamfcc')
title('Delta MFCCs')
xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
xticklabels(1:floor(length(audio_signal)/sample_rate))
xlabel('Time (s)')
axis tight
subplot(3,1,3)
plot(audio_deltadeltamfcc')
title('Delta-delta MFCCs')
xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
xticklabels(1:floor(length(audio_signal)/sample_rate))
xlabel('Time (s)')
axis tight
```

### dct [Discrete cosine transform (DCT)](https://en.wikipedia.org/wiki/Discrete_cosine_transform) using the fast Fourier transform (FFT)
`audio_dct = z.dct(audio_signal,dct_type);`

Arguments:
```
audio_signal: audio signal [number_samples,number_frames]
dct_type: dct type (1, 2, 3, or 4)
audio_dct: audio DCT [number_frequencies,number_frames]
```

Example: Compute the 4 different DCTs and compare them to Matlab's DCTs
```
% Audio signal averaged over its channels and sample rate in Hz
[audio_signal,sample_rate] = audioread('audio_file.wav');
audio_signal = mean(audio_signal,2);

% Audio signal for a given window length, and one frame
window_length = 1024;
audio_signal = audio_signal(1:window_length);

% DCT-I, II, III, and IV
audio_dct1 = z.dct(audio_signal,1);
audio_dct2 = z.dct(audio_signal,2);
audio_dct3 = z.dct(audio_signal,3);
audio_dct4 = z.dct(audio_signal,4);

% Matlab's DCT-I, II, III, and IV
matlab_dct1 = dct(audio_signal,'Type',1);
matlab_dct2 = dct(audio_signal,'Type',2);
matlab_dct3 = dct(audio_signal,'Type',3);
matlab_dct4 = dct(audio_signal,'Type',4);

% DCT-I, II, III, and IV, Matlab's versions, and their differences displayed
figure
subplot(4,3,1), plot(audio_dct1), axis tight, title('DCT-I')
subplot(4,3,2), plot(matlab_dct1), axis tight, title('Maltab''s DCT-I')
subplot(4,3,3), plot(audio_dct1-matlab_dct1), axis tight, title('Differences')
subplot(4,3,4), plot(audio_dct2), axis tight, title('DCT-II')
subplot(4,3,5), plot(matlab_dct2),axis tight, title('Maltab''s DCT-II')
subplot(4,3,6), plot(audio_dct2-matlab_dct2), axis tight, title('Differences')
subplot(4,3,7), plot(audio_dct3), axis tight, title('DCT-III')
subplot(4,3,8), plot(matlab_dct3), axis tight, title('Maltab''s DCT-III')
subplot(4,3,9), plot(audio_dct3-matlab_dct3), axis tight, title('Differences')
subplot(4,3,10), plot(audio_dct4), axis tight, title('DCT-IV')
subplot(4,3,11), plot(matlab_dct4), axis tight, title('Maltab''s DCT-IV')
subplot(4,3,12), plot(audio_dct4-matlab_dct4), axis tight, title('Differences')
```

### dst [Discrete sine transform (DST)](https://en.wikipedia.org/wiki/Discrete_sine_transform) using the fast Fourier transform (FFT)
`audio_dst = z.dst(audio_signal,dst_type);`

Arguments:
```
audio_signal: audio signal [number_samples,number_frames]
dst_type: DST type (1, 2, 3, or 4)
audio_dst: audio DST [number_frequencies,number_frames]
```

Example: Compute the 4 different DSTs and compare them to their respective inverses
```
% Audio signal averaged over its channels and sample rate in Hz
[audio_signal,sample_rate] = audioread('audio_file.wav');
audio_signal = mean(audio_signal,2);

% Audio signal for a given window length, and one frame
window_length = 1024;
audio_signal = audio_signal(1:window_length);

% DST-I, II, III, and IV
audio_dst1 = z.dst(audio_signal,1);
audio_dst2 = z.dst(audio_signal,2);
audio_dst3 = z.dst(audio_signal,3);
audio_dst4 = z.dst(audio_signal,4);

% Respective inverses, i.e., DST-I, II, III, and IV
audio_idst1 = z.dst(audio_dst1,1);
audio_idst2 = z.dst(audio_dst2,3);
audio_idst3 = z.dst(audio_dst3,2);
audio_idst4 = z.dst(audio_dst4,4);

% DST-I, II, III, and IV, respective inverses, and differences with the original signal displayed
figure
subplot(4,3,1), plot(audio_dst1), axis tight, title('DST-I')
subplot(4,3,2), plot(audio_idst1), axis tight, title('Inverse DST-I = DST-I')
subplot(4,3,3), plot(audio_idst1-audio_signal), axis tight, title('Differences with signal')
subplot(4,3,4), plot(audio_dst2), axis tight, title('DST-II')
subplot(4,3,5), plot(audio_idst2), axis tight, title('Inverse DST-II = DST-III')
subplot(4,3,6), plot(audio_idst2-audio_signal), axis tight, title('Differences with signal')
subplot(4,3,7), plot(audio_dst3), axis tight, title('DST-III')
subplot(4,3,8), plot(audio_idst3), axis tight, title('Inverse DST-III = DST-II')
subplot(4,3,9), plot(audio_idst3-audio_signal), axis tight, title('Differences with signal')
subplot(4,3,10), plot(audio_dst4), axis tight, title('DST-IV')
subplot(4,3,11), plot(audio_idst4), axis tight, title('Inverse DST-IV = DST-IV')
subplot(4,3,12), plot(audio_idst4-audio_signal), axis tight, title('Differences with signal')
```

### mdct [Modified discrete cosine transform (MDCT)](https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform) using the DCT-IV
`audio_mdct = z.mdct(audio_signal,window_function);`

Arguments:
```
audio_signal: audio signal [number_samples,1]
window_function: window function [window_length,1]
audio_mdct: audio MDCT [number_frequencies,number_times]
```

Example: Compute and display the MDCT as used in the AC-3 audio coding format
```
% Audio file averaged over the channels and sample rate in Hz
[audio_signal,sample_rate] = audioread('audio_file.wav');
audio_signal = mean(audio_signal,2);

% Kaiser-Bessel-derived (KBD) window as used in the AC-3 audio coding format
window_length = 2048;
alpha_value = 5;
window_function = kaiser(window_length/2+1,alpha_value*pi);
window_function2 = cumsum(window_function(1:window_length/2));
window_function = sqrt([window_function2;window_function2(window_length/2:-1:1)]./sum(window_function));

% MDCT
audio_mdct = z.mdct(audio_signal,window_function);

% MDCT displayed in dB, s, and kHz
figure
imagesc(db(audio_mdct))
axis xy
colormap(jet)
title('mdct (dB)')
xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/(window_length/2)))
xticklabels(1:floor(length(audio_signal)/sample_rate))
xlabel('Time (s)')
yticks(round((1e3:1e3:sample_rate/2)/sample_rate*window_length))
yticklabels(1:sample_rate/2*1e-3)
ylabel('Frequency (kHz)')
```

### imdct Inverse modified discrete cosine transform (MDCT) using the DCT-IV
`audio_signal = z.imdct(audio_mdct,window_function);`

Arguments:
```
window_function: window function [window_length,1]
audio_mdct: audio MDCT [number_frequencies,number_times]
audio_signal: audio signal [number_samples,1]
```

Example: Verify that the MDCT is perfectly invertible
```
% Audio file averaged over the channels and sample rate in Hz
[audio_signal,sample_rate] = audioread('audio_file.wav');
audio_signal = mean(audio_signal,2);

% MDCT with a slope function as used in the Vorbis audio coding format
window_length = 2048;
window_function = sin((pi/2)*sin((pi/window_length)*(1/2:(window_length-1/2))).^2)';
audio_mdct = z.mdct(audio_signal,window_function);

% Inverse MDCT and error signal
audio_signal2 = z.imdct(audio_mdct,window_function);
audio_signal2 = audio_signal2(1:length(audio_signal));
error_signal = audio_signal-audio_signal2;

% Original, resynthesized, and error signals
figure
subplot(3,1,1)
plot(audio_signal)
title('Original Signal')
xticks(sample_rate:sample_rate:length(audio_signal))
xticklabels(1:floor(length(audio_signal)/sample_rate))
xlabel('Time (s)')
axis tight
subplot(3,1,2)
plot(audio_signal2)
title('Resynthesized Signal')
xticks(sample_rate:sample_rate:length(audio_signal))
xticklabels(1:floor(length(audio_signal)/sample_rate))
xlabel('Time (s)')
axis tight
subplot(3,1,3)
plot(error_signal)
title('Error Signal')
xticks(sample_rate:sample_rate:length(audio_signal))
xticklabels(1:floor(length(audio_signal)/sample_rate))
xlabel('Time (s)')
axis tight
```

## z Python module

z This module implements several functions for audio signal processing.

z Functions:
- [stft - Short-time Fourier transform (STFT)](#stft-short-time-fourier-transform-stft-1)
- [istft - Inverse STFT](#istft-inverse-short-time-fourier-transform-stft-1)
- [cqtkernel - Constant-Q transform (CQT) kernel](#cqtkernel-constant-q-transform-cqt-kernel-1)
- [cqtspectrogram - CQT spectrogram using a CQT kernel](#cqtspectrogram-constant-q-transform-cqt-spectrogram-using-a-cqt-kernel-1)
- [cqtchromagram - CQT chromagram using a CQT kernel](#cqtchromagram-constant-q-transform-cqt-chromagram-using-a-cqt-kernel-1)

### stft Short-time Fourier transform (STFT)
```
import z
audio_stft = z.stft(audio_signal, window_function, step_length)
```
    
Arguments:
```
audio_signal: audio signal [number_samples,1]
window_function: window function [window_length,1]
step_length: step length in samples
audio_stft: audio stft [window_length,number_frames]
```

Example: Compute and display the spectrogram of an audio file
```
# Import modules
import scipy.io.wavfile
import numpy as np
import scipy.signal
import z
import matplotlib.pyplot as plt

# Audio signal (normalized) averaged over its channels and sample rate in Hz
sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
audio_signal = audio_signal/(2.0**(audio_signal.itemsize*8-1))
audio_signal = np.mean(audio_signal, 1)

# Window duration in seconds (audio is stationary around 40 milliseconds)
window_duration = 0.04

# Window length in samples (power of 2 for fast FFT and constant overlap-add (COLA))
window_length = int(np.power(2, np.ceil(np.log2(window_duration * sample_rate))))

# Window function (periodic Hamming window for COLA)
window_function = scipy.signal.hamming(window_length, False)

# Step length in samples (half the window length for COLA)
step_length = int(window_length/2)

# Magnitude spectrogram (without the DC component and the mirrored frequencies)
audio_stft = z.stft(audio_signal, window_function, step_length)
audio_spectrogram = np.absolute(audio_stft[1:int(window_length/2+1), :])

# Spectrogram displayed in dB, s, and kHz
plt.imshow(20*np.log10(audio_spectrogram), aspect='auto', cmap='jet', origin='lower')
plt.title('Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
           np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, sample_rate/2+1, 1e3)/sample_rate*window_length),
           np.arange(1, int(sample_rate/2*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.show()
```

### istft Inverse short-time Fourier transform (STFT)
```
import z
audio_signal = z.istft(audio_stft, window_function, step_length)
```

Arguments:
```
audio_stft: audio STFT [window_length,number_frames]
window_function: window function [window_length,1]
step_length: step length in samples
audio_signal: audio signal [number_samples,1]
```

Example: Estimate the center and sides signals of a stereo audio file
```
# Import modules
import scipy.io.wavfile
import numpy as np
import scipy.signal
import z

# Stereo audio signal (normalized) and sample rate in Hz
sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
audio_signal = audio_signal/(2.0**(audio_signal.itemsize*8-1))

# Parameters for the STFT
window_duration = 0.04
window_length = int(np.power(2, np.ceil(np.log2(window_duration*sample_rate))))
window_function = scipy.signal.hamming(window_length, False)
step_length = int(window_length/2)

# STFT of the left and right channels
audio_stft1 = z.stft(audio_signal[:, 0], window_function, step_length)
audio_stft2 = z.stft(audio_signal[:, 1], window_function, step_length)

# Magnitude spectrogram (with DC component) of the left and right channels
audio_spectrogram1 = np.absolute(audio_stft1[0:int(window_length/2)+1, :])
audio_spectrogram2 = np.absolute(audio_stft2[0:int(window_length/2)+1, :])

# Time-frequency masks of the left and right channels for the center signal
center_mask1 = np.minimum(audio_spectrogram1, audio_spectrogram2)/audio_spectrogram1
center_mask2 = np.minimum(audio_spectrogram1, audio_spectrogram2)/audio_spectrogram2

# STFT of the left and right channels for the center signal (with extension to mirrored frequencies)
center_stft1 = np.multiply(np.concatenate((center_mask1, center_mask1[int(window_length/2)-1:0:-1, :])),
                           audio_stft1)
center_stft2 = np.multiply(np.concatenate((center_mask2, center_mask2[int(window_length/2)-1:0:-1, :])),
                           audio_stft2)

# Synthesized signals of the left and right channels for the center signal
center_signal1 = z.istft(center_stft1, window_function, step_length)
center_signal2 = z.istft(center_stft2, window_function, step_length)

# Final stereo center and sides signals
center_signal = np.concatenate((center_signal1, center_signal2), 1)
center_signal = center_signal[0:len(audio_signal), :]
sides_signal = audio_signal-center_signal

# Synthesized center and side signals (un-normalized)
scipy.io.wavfile.write('center_signal.wav', sample_rate, center_signal)
scipy.io.wavfile.write('sides_signal.wav', sample_rate, sides_signal)
```

### cqtkernel Constant-Q transform (CQT) kernel
```
import z
cqt_kernel = z.cqtkernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency)
```

Arguments:
```
sample_rate: sample rate in Hz
frequency_resolution: frequency resolution in number of frequency channels per semitone
minimum_frequency: minimum frequency in Hz
maximum_frequency: maximum frequency in Hz
CQT kernel [number_frequencies,fft_length]
```

Example: Compute and display the CQT kernel
```
# Import modules
import z
import numpy as np
import matplotlib.pyplot as plt

# CQT kernel parameters
sample_rate = 44100
frequency_resolution = 2
minimum_frequency = 55
maximum_frequency = sample_rate/2

# CQT kernel
cqt_kernel = z.cqtkernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency)

# Magnitude CQT kernel displayed
plt.imshow(np.absolute(cqt_kernel).toarray(), aspect='auto', cmap='jet', origin='lower')
plt.title('Magnitude CQT kernel')
plt.xlabel('FFT length')
plt.ylabel('CQT frequency')
plt.show()
```

### cqtspectrogram Constant-Q transform (CQT) spectrogram using a CQT kernel
```
import z
audio_spectrogram = z.cqtspectrogram(audio_signal, sample_rate, time_resolution, cqt_kernel);
```

Arguments:
```
audio_signal: audio signal [number_samples,1]
sample_rate: sample rate in Hz
time_resolution: time resolution in number of time frames per second
cqt_kernel: CQT kernel [number_frequencies,fft_length]
audio_spectrogram: audio spectrogram in magnitude [number_frequencies,number_times]
```

Example: Compute and display the CQT spectrogram
```
# Import modules
import scipy.io.wavfile
import numpy as np
import z
import matplotlib.pyplot as plt

# Audio file (normalized) averaged over the channels and sample rate in Hz
sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
audio_signal = audio_signal / ( 2.0**(audio_signal.itemsize*8-1))
audio_signal = np.mean(audio_signal, 1)

# CQT kernel
frequency_resolution = 2
minimum_frequency = 55
maximum_frequency = 3520
cqt_kernel = z.cqtkernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency)

# CQT spectrogram
time_resolution = 25
audio_spectrogram = z.cqtspectrogram(audio_signal, sample_rate, time_resolution, cqt_kernel)

# CQT spectrogram displayed in dB, s, and semitones
plt.imshow(20*np.log10(audio_spectrogram), aspect='auto', cmap='jet', origin='lower')
plt.title('CQT spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*time_resolution),
           np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.arange(1, 6*12*frequency_resolution+1, 12*frequency_resolution),
           ('A1 (55 Hz)','A2 (110 Hz)','A3 (220 Hz)','A4 (440 Hz)','A5 (880 Hz)','A6 (1760 Hz)'))
plt.ylabel('Frequency (semitones)')
plt.show()
```

### cqtchromagram Constant-Q transform (CQT) chromagram using a CQT kernel
```
import z
audio_chromagram = z.cqtchromagram(audio_signal, sample_rate, time_resolution, frequency_resolution, cqt_kernel);
```

Arguments:
```
audio_signal: audio signal [number_samples,1]
sample_rate: sample rate in Hz
time_resolution: time resolution in number of time frames per second
frequency_resolution: frequency resolution in number of frequency channels per semitones
cqt_kernel: CQT kernel [number_frequencies,fft_length]
audio_chromagram: audio chromagram [number_chromas,number_times]
```

Example: Compute and display the CQT chromagram
```
# Import modules
import scipy.io.wavfile
import numpy as np
import z
import matplotlib.pyplot as plt

# Audio signal (normalized) averaged over its channels and sample rate in Hz
sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
audio_signal = np.mean(audio_signal, 1)

# CQT kernel
frequency_resolution = 2
minimum_frequency = 55
maximum_frequency = 3520
cqt_kernel = z.cqtkernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency)

# CQT chromagram
time_resolution = 25
audio_chromagram = z.cqtchromagram(audio_signal, sample_rate, time_resolution, frequency_resolution, cqt_kernel)

# CQT chromagram displayed in dB, s, and chromas
plt.imshow(20*np.log10(audio_chromagram), aspect='auto', cmap='jet', origin='lower')
plt.title('CQT chromagram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*time_resolution),
           np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.arange(1, 12*frequency_resolution+1, frequency_resolution),
           ('A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'))
plt.ylabel('Chroma')
plt.show()
```

# Author

- Zafar Rafii
- zafarrafii@gmail.com
- [Website](http://zafarrafii.com/)
- [CV](http://zafarrafii.com/Zafar%20Rafii%20-%20C.V..pdf)
- [Google Scholar](https://scholar.google.com/citations?user=8wbS2EsAAAAJ&hl=en)
- [LinkedIn](https://www.linkedin.com/in/zafar-rafii-b44b59123/)
