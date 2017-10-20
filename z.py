"""
Z This module implements few basic methods for audio signal processing

    Z Functions:
        stft - Short-time Fourier transform (STFT)
        istft - Inverse STFT
        cqtkernel - Constant-Q transform (CQT) kernel
        cqtspectrogram - CQT spectrogram using a CQT kernel
        cqtchromagram - CQT chromagram using a CQT kernel
        mfcc - Mel frequency cepstrum coefficients (MFCCs)
        dct - Discrete cosine transform (DCT) using the fast Fourier transform (FFT)
        dst - Discrete sine transform (DST) using the FFT
        mdct - Modified discrete cosine transform (MDCT) using the DCT-IV
        imdct - Inverse MDCT using the DCT-IV

    Author
        Zafar Rafii
        zafarrafii@gmail.com
        10/20/17

    See also http://zafarrafii.com
"""

import numpy as np


def stft(audio_signal, window_function, step_length):
    """
    stft Short-time Fourier transform (STFT)
        audio_stft = z.stft(audio_signal,window_function,step_length);

        Arguments:
            audio_signal: audio signal [number_samples,1]
            window_function: window function [window_length,1]
            step_length: step length in samples
            audio_stft: audio STFT [window_length,number_frames]

        Example: Compute and display the spectrogram of an audio file
            # Import modules
            import scipy.io.wavfile

            # Audio signal averaged over its channels and sample rate in Hz
            sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')

            audio_signal = mean(audio_signal,2);

            # Window duration in seconds (audio is stationary around 40 milliseconds)
            window_duration = 0.04;

            # Window length in samples (power of 2 for fast FFT and constant overlap-add (COLA))
            window_length = 2^nextpow2(window_duration*sample_rate);

            # Window function (periodic Hamming window for COLA)
            window_function = hamming(window_length,'periodic');

            # Step length in samples (half the window length for COLA)
            step_length = window_length/2;

            # Magnitude spectrogram (without the DC component and the mirrored frequencies)
            audio_stft = z.stft(audio_signal,window_function,step_length);
            audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));

            # Spectrogram displayed in dB, s, and kHz
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
    """

    # Number of samples
    number_samples = len(audio_signal)

    # Window length in samples
    window_length = len(window_function)

    # Number of time frames
    number_times = int(np.ceil((window_length - step_length + number_samples) / step_length))

    # Zero-padding at the start and end to center the windows
    audio_signal = np.pad(audio_signal, (window_length-step_length, number_times*step_length-number_samples),
                          'constant', constant_values=(0, 0))

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Window the signal
        sample_index = step_length*time_index
        audio_stft[:, time_index] = audio_signal[sample_index:window_length+sample_index]*window_function

    # Fourier transform of the frames
    audio_stft = np.fft.fft(audio_stft, window_length, 0)

    return audio_stft


def test():

    # Import modules
    import scipy.io.wavfile
    import scipy.signal
    import matplotlib.pyplot as plt

    # Audio signal averaged over its channels and sample rate in Hz
    sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
    audio_signal = np.mean(audio_signal, 1)

    # Window duration in seconds (audio is stationary around 40 milliseconds)
    window_duration = 0.04

    # Window length in samples (power of 2 for fast FFT and constant overlap-add (COLA))
    window_length = int(np.power(2, np.ceil(np.log2(window_duration * sample_rate))))

    # Window function (periodic Hamming window for COLA)
    window_function = scipy.signal.hamming(window_length, False)

    # Step length in samples(half the window length for COLA)
    step_length = int(window_length/2)

    # Magnitude spectrogram (without the DC component and the mirrored frequencies)
    audio_stft = stft(audio_signal, window_function, step_length)
    audio_spectrogram = np.absolute(audio_stft[1:int(window_length/2+1), :])

    plt.imshow(audio_spectrogram, extent=[0, 1, 0, 1])
    plt.show()
    return 0

