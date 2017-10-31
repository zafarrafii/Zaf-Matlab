"""
z This module implements several functions for audio signal processing.

z Functions:
stft - Short-time Fourier transform (STFT)
istft - inverse STFT
cqtkernel - Constant-Q transform (CQT) kernel
cqtspectrogram - CQT spectrogram using a CQT kernel
cqtchromagram - CQT chromagram using a CQT kernel
mfcc - Mel frequency cepstrum coefficients (MFCCs)

Zafar Rafii
zafarrafii@gmail.com
http://zafarrafii.com
https://github.com/zafarrafii
10/31/17
"""

import numpy as np
import scipy.sparse
import scipy.signal
import scipy.fftpack


def stft(audio_signal, window_function, step_length):
    """
    Short-time Fourier transform (STFT)

    :param audio_signal: audio signal [number_samples,1]
    :param window_function: window function [window_length,1]
    :param step_length: step length in samples
    :return: audio STFT [window_length,number_frames]

    Example: Compute and display the spectrogram of an audio file:

    # Import modules
    import scipy.io.wavfile
    import numpy as np
    import scipy.signal
    import z
    import matplotlib.pyplot as plt

    # Audio signal (normalized) averaged over its channels and sample rate in Hz
    sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
    audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
    audio_signal = np.mean(audio_signal, 1)

    # Window duration in seconds (audio is stationary around 40 milliseconds)
    window_duration = 0.04

    # Window length in samples (power of 2 for fast FFT and constant overlap-add (COLA))
    window_length = int(2**np.ceil(np.log2(window_duration*sample_rate)))

    # Window function (periodic Hamming window for COLA)
    window_function = scipy.signal.hamming(window_length, False)

    # Step length in samples (half the window length for COLA)
    step_length = int(window_length/2)

    # Magnitude spectrogram (without the DC component and the mirrored frequencies)
    audio_stft = z.stft(audio_signal, window_function, step_length)
    audio_spectrogram = abs(audio_stft[1:int(window_length/2+1), :])

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
    """

    # Number of samples
    number_samples = len(audio_signal)

    # Window length in samples
    window_length = len(window_function)

    # Number of time frames
    number_times = int(np.ceil((window_length-step_length+number_samples)/step_length))

    # Zero-padding at the start and end to center the windows
    audio_signal = np.pad(audio_signal, (window_length-step_length, number_times*step_length-number_samples),
                          'constant', constant_values=(0, 0))

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Window the signal
        sample_index = step_length*time_index
        audio_stft[:, time_index] = audio_signal[sample_index:window_length+sample_index] * window_function

    # Fourier transform of the frames
    audio_stft = np.fft.fft(audio_stft, window_length, 0)

    return audio_stft


def istft(audio_stft, window_function, step_length):
    """
    Inverse short-time Fourier transform (STFT)

    :param audio_stft: audio STFT [window_length,number_frames]
    :param window_function: window function [window_length,1]
    :param step_length: step length in samples
    :return: audio_signal: audio signal [number_samples,1]

    Example: Estimate the center and sides signals of a stereo audio file
    # Import modules
    import scipy.io.wavfile
    import numpy as np
    import scipy.signal
    import z

    # Stereo audio signal (normalized) and sample rate in Hz
    sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
    audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))

    # Parameters for the STFT
    window_duration = 0.04
    window_length = int(2**np.ceil(np.log2(window_duration*sample_rate)))
    window_function = scipy.signal.hamming(window_length, False)
    step_length = int(window_length/2)

    # STFT of the left and right channels
    audio_stft1 = z.stft(audio_signal[:, 0], window_function, step_length)
    audio_stft2 = z.stft(audio_signal[:, 1], window_function, step_length)

    # Magnitude spectrogram (with DC component) of the left and right channels
    audio_spectrogram1 = abs(audio_stft1[0:int(window_length/2)+1, :])
    audio_spectrogram2 = abs(audio_stft2[0:int(window_length/2)+1, :])

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
    """

    # Window length in samples and number of time frames
    window_length, number_times = np.shape(audio_stft)

    # Number of samples for the signal
    number_samples = (number_times-1)*step_length + window_length

    # Initialize the signal
    audio_signal = np.zeros(number_samples)

    # Inverse Fourier transform of the frames and real part to ensure real values
    audio_stft = np.real(np.fft.ifft(audio_stft, window_length, 0))

    # Loop over the time frames
    for time_index in range(0, number_times):
        # Constant overlap-add (if proper window and step)
        sample_index = step_length*time_index
        audio_signal[sample_index:window_length+sample_index] \
            = audio_signal[sample_index:window_length+sample_index] + audio_stft[:, time_index]

    # Remove the zero-padding at the start and end
    audio_signal = audio_signal[window_length-step_length:number_samples-(window_length-step_length)]

    # Un-apply window (just in case)
    audio_signal = audio_signal / sum(window_function[0:window_length:step_length])

    # Expand the shape from (number_samples,) to (number_samples, 1)
    audio_signal = np.expand_dims(audio_signal, 1)

    return audio_signal


def cqtkernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency):
    """
    Constant-Q transform (CQT) kernel

    :param sample_rate: sample rate in Hz
    :param frequency_resolution: frequency resolution in number of frequency channels per semitone
    :param minimum_frequency: minimum frequency in Hz
    :param maximum_frequency: maximum frequency in Hz
    :return: cqt_kernel: CQT kernel [number_frequencies,fft_length]

    Example: Compute and display the CQT kernel
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
    """

    # Number of frequency channels per octave
    octave_resolution = 12*frequency_resolution

    # Constant ratio of frequency to resolution (= fk/(fk+1-fk))
    quality_factor = 1 / (2**(1/octave_resolution)-1)

    # Number of frequency channels for the CQT
    number_frequencies = int(round(octave_resolution*np.log2(maximum_frequency/minimum_frequency)))

    # Window length for the FFT (= window length of the minimum frequency = longest window)
    fft_length = int(2**np.ceil(np.log2(quality_factor*sample_rate/minimum_frequency)))

    # Initialize the kernel
    cqt_kernel = np.zeros((number_frequencies, fft_length), dtype=complex)

    # Loop over the frequency channels
    for frequency_index in range(0, number_frequencies):

        # Frequency value (in Hz)
        frequency_value = minimum_frequency * 2**(frequency_index/octave_resolution)

        # Window length (nearest odd value because the complex exponential will have an odd length, in samples)
        window_length = 2*round(quality_factor*sample_rate/frequency_value/2) + 1

        # Temporal kernel (without zero-padding, odd and symmetric)
        temporal_kernel = np.hamming(window_length)\
            * np.exp(2*np.pi*1j*quality_factor
                     * np.arange(-(window_length-1)/2, (window_length-1)/2+1)/window_length) / window_length

        # Pre zero-padding to center FFTs (fft does post zero-padding; temporal kernel still odd but almost symmetric)
        temporal_kernel = np.pad(temporal_kernel, (int((fft_length-window_length+1)/2), 0),
                                 'constant', constant_values=0)

        # Spectral kernel (mostly real because temporal kernel almost symmetric)
        spectral_kernel = np.fft.fft(temporal_kernel, fft_length)

        # Save the spectral kernels
        cqt_kernel[frequency_index, :] = spectral_kernel

    # Energy threshold for making the kernel sparse
    energy_threshold = 0.01

    # Make the CQT kernel sparser
    cqt_kernel[np.absolute(cqt_kernel) < energy_threshold] = 0

    # Make the CQT kernel sparse
    cqt_kernel = scipy.sparse.csc_matrix(cqt_kernel)

    # From Parseval's theorem
    cqt_kernel = np.conjugate(cqt_kernel)/fft_length

    return cqt_kernel


def cqtspectrogram(audio_signal, sample_rate, time_resolution, cqt_kernel):
    """
    Constant-Q transform (CQT) spectrogram using a kernel

    :param audio_signal: audio signal [number_samples,1]
    :param sample_rate: sample rate in Hz
    :param time_resolution: time resolution in number of time frames per second
    :param cqt_kernel: CQT kernel [number_frequencies,fft_length]
    :return: audio_spectrogram: audio spectrogram in magnitude [number_frequencies,number_times]

    Example: Compute and display the CQT spectrogram
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
    """

    # Number of time samples per time frame
    step_length = round(sample_rate/time_resolution)

    # Number of time frames
    number_times = int(np.floor(len(audio_signal)/step_length))

    # Number of frequency channels and FFT length
    number_frequencies, fft_length = np.shape(cqt_kernel)

    # Zero-padding to center the CQT
    audio_signal = np.pad(audio_signal, (int(np.ceil((fft_length-step_length)/2)),
                                         int(np.floor((fft_length-step_length)/2))), 'constant', constant_values=(0, 0))

    # Initialize the spectrogram
    audio_spectrogram = np.zeros((number_frequencies, number_times))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Magnitude CQT using the kernel
        sample_index = step_length*time_index
        audio_spectrogram[:, time_index] = abs(cqt_kernel*np.fft.fft(audio_signal[sample_index:sample_index+fft_length]))

    return audio_spectrogram


def cqtchromagram(audio_signal, sample_rate, time_resolution, frequency_resolution, cqt_kernel):
    """
    Constant-Q transform (CQT) chromagram using a kernel

    :param audio_signal: audio signal [number_samples,1]
    :param sample_rate: sample rate in Hz
    :param time_resolution: time resolution in number of time frames per second
    :param frequency_resolution: frequency resolution in number of frequency channels per semitones
    :param cqt_kernel: CQT kernel [number_frequencies,fft_length]
    :return: audio_chromagram: audio chromagram [number_chromas,number_times]

    Example: Compute and display the CQT chromagram
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
    """

    # CQT spectrogram
    audio_spectrogram = cqtspectrogram(audio_signal, sample_rate, time_resolution, cqt_kernel)

    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Number of chroma bins
    number_chromas = 12*frequency_resolution

    # Initialize the chromagram
    audio_chromagram = np.zeros((number_chromas, number_times))

    # Loop over the chroma bins
    for chroma_index in range(0,number_chromas):

        # Sum the energy of the frequency channels for every chroma
        audio_chromagram[chroma_index, :] \
            = np.sum(audio_spectrogram[chroma_index:number_frequencies:number_chromas, :], 0)

    return audio_chromagram


def mfcc(audio_signal, sample_rate, number_filters, number_coefficients):
    """
    Mel frequency cepstrum coefficients (MFFCs)

    :param audio_signal: audio signal [number_samples,1]
    :param sample_rate: sample rate in Hz
    :param number_filters: number of filters
    :param number_coefficients: number of coefficients (without the 0th coefficient)
    :return: audio_mfcc: audio MFCCs [number_times,number_coefficients]

    Example: Compute and display the MFCCs, delta MFCCs, and delta-detla MFCCs
    """

    # Window duration in seconds, length in samples, and function, and step length in samples
    window_duration = 0.04
    window_length = int(2**np.ceil(np.log2(window_duration*sample_rate)))
    window_function = scipy.signal.hamming(window_length, False)
    step_length = int(window_length/2)

    # Magnitude spectrogram (without the DC component and the mirrored frequencies)
    audio_stft = stft(audio_signal, window_function, step_length)
    audio_spectrogram = abs(audio_stft[1:int(window_length/2)+1, :])

    # Minimum and maximum mel frequencies
    mininum_melfrequency = 2595 * np.log10(1+(sample_rate/window_length)/700)
    maximum_melfrequency = 2595 * np.log10(1+(sample_rate/2)/700)

    # Indices of the overlapping filters (linearly spaced in the  mel scale and logarithmically spaced in the linear scale)
    filter_width = 2*(maximum_melfrequency-mininum_melfrequency)/(number_filters+1)
    filter_indices = np.arange(mininum_melfrequency, maximum_melfrequency+1, filter_width/2)
    filter_indices = np.round(700*(np.power(10, filter_indices/2595)-1)*window_length/sample_rate).astype(int)

    # Initialize the filter bank
    filter_bank = np.zeros((number_filters, int(window_length/2)))

    # Loop over the filters
    for filter_index in range(0, number_filters):

        # Left and right sides of the triangular overlapping filters (linspace more accurate than triang or bartlett!)
        filter_bank[filter_index, filter_indices[filter_index]:filter_indices[filter_index+1]] \
            = np.linspace(0, 1, num=filter_indices[filter_index+1]-filter_indices[filter_index])
        filter_bank[filter_index, filter_indices[filter_index+1]:filter_indices[filter_index+2]] \
            = np.linspace(1, 0, num=filter_indices[filter_index+2]-filter_indices[filter_index+1])

    # Discrete cosine transform (DCT) of the log of the magnitude spectrogram mapped onto the mel scale using the filter bank
    audio_mfcc = scipy.fftpack.dct(np.log(np.dot(filter_bank, audio_spectrogram)+np.spacing(1)))

    # The first coefficients (without the 0th) represent the MFCCs
    audio_mfcc = audio_mfcc[1:number_coefficients, :]

    return audio_mfcc


def test():
    # Import modules
    import scipy.io.wavfile
    import numpy as np
    import z
    import matplotlib.pyplot as plt

    # Audio signal (normalized) averaged over its channels and sample rate in Hz
    sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
    audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
    audio_signal = np.mean(audio_signal, 1)

    # MFCCs for a given number of filters and coefficients
    number_filters = 40
    number_coefficients = 20
    audio_mfcc = z.mfcc(audio_signal, sample_rate, number_filters, number_coefficients)

    # Delta and delta-delta MFCCs
    audio_deltamfcc = np.diff(audio_mfcc, n=1, axis=1)
    audio_deltadeltamfcc = np.diff(audio_deltamfcc, n=1, axis=1)

    # MFCCs, delta MFCCs, and delta-delta MFCCs displayed in s
    step_length = 2**np.ceil(np.log2(0.04*sample_rate)) / 2
    plt.subplots(3, 1)
    plt.plot(audio_mfcc)
    plt.title('MFCCs')
    plt.subplots(3, 1)
    plt.plot(audio_deltamfcc)

    return 0
