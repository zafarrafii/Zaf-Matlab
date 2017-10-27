"""
z This module implements several functions for audio signal processing.

z Functions:
stft - Short-time Fourier transform (STFT)
istft - inverse STFT

Zafar Rafii
zafarrafii@gmail.com
http://zafarrafii.com
https://github.com/zafarrafii
10/27/17
"""

import numpy as np
import scipy.sparse


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
    audio_signal = audio_signal / ( 2.0**(audio_signal.itemsize*8-1))
    audio_signal = np.mean(audio_signal, 1)

    # Window duration in seconds (audio is stationary around 40 milliseconds)
    window_duration = 0.04

    # Window length in samples (power of 2 for fast FFT and constant overlap-add (COLA))
    window_length = int(np.power(2, np.ceil(np.log2(window_duration*sample_rate))))

    # Window function (periodic Hamming window for COLA)
    window_function = scipy.signal.hamming(window_length, False)

    # Step length in samples (half the window length for COLA)
    step_length = int(window_length/2)

    # Magnitude spectrogram (without the DC component and the mirrored frequencies)
    audio_stft = z.stft(audio_signal, window_function, step_length)
    audio_spectrogram = np.absolute(audio_stft[1:int(window_length/2+1), :])

    # Spectrogram displayed in dB, s, and kHz
    plt.imshow(20*np.log10(audio_spectrogram), cmap='jet', origin='lower')
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
    """

    # Window length in samples and number of time frames
    [window_length, number_times] = np.shape(audio_stft)

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
    audio_signal = audio_signal / np.sum(window_function[0:window_length:step_length])

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
    :return: CQT kernel [number_frequencies,fft_length]

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
    fft_length = int(np.power(2, np.ceil(np.log2(quality_factor*sample_rate/minimum_frequency))))

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


def test():
    return 0
