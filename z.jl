"""
`z` This module implements several functions for audio signal processing.

`z` Functions:
- `stft` - Short-time Fourier transform (STFT)
- `istft` - inverse STFT
- `cqtkernel` - Constant-Q transform (CQT) kernel
- `cqtspectrogram` - CQT spectrogram using a CQT kernel
- `cqtchromagram` - CQT chromagram using a CQT kernel
- `mfcc` - Mel frequency cepstrum coefficients (MFCCs)
- `dct` - Discrete cosine transform (DCT) using the fast Fourier transform (FFT)
- `dst` - Discrete sine transform (DST) using the FFT
- `mdct` - Modified discrete cosine transform (MDCT) using the FFT
- `imdct` - Inverse MDCT using the FFT

Author:
- Zafar Rafii
- zafarrafii@gmail.com
- http://zafarrafii.com
- https://github.com/zafarrafii
- https://www.linkedin.com/in/zafarrafii/
- 05/21/18
"""
module z

# Public
export stft, istft, test

"""
    audio_stft = z.stft(audio_signal, window_function, step_length);

Compute the short-time Fourier transform (STFT)

# Arguments:
- `audio_signal::Float`: the audio signal [number_samples, 1]
- `window_function::Integer`: the window function [window_length, 1]
- `step_length::Integer`: the step length in samples
- `audio_stft::Float`: the audio STFT [window_length, number_frames]

# Example: Compute the spectrogram of an audio file
```
# Audio signal averaged over its channels and sample rate in Hz
Pkg.add("WAV")
using WAV
audio_signal, sample_rate = wavread("audio_file.wav");
audio_signal = mean(audio_signal, 2);

# Window duration in seconds (audio is stationary around 40 milliseconds)
window_duration = 0.04;

# Window length in samples (power of 2 for fast FFT and constant overlap-add (COLA))
window_length = nextpow2(convert(Int64, window_duration*sample_rate));

# Window function (periodic Hamming window for COLA)
window_function = 0.54 - 0.46*cos.(2*pi*(0:window_length-1)/window_length);

# Step length in samples (half the window length for COLA)
step_length = convert(Int64, window_length/2);

# Magnitude spectrogram (without the DC component and the mirrored frequencies)
include("z.jl")
using z
audio_stft = stft(audio_signal, window_function, step_length);
audio_spectrogram = abs.(audio_stft[2:convert(Int64, window_length/2)+1,:]);

# Spectrogram displayed in dB, s, and kHz
Pkg.add("Plots")
using Plots
plotly()
x_labels = [string(round(i*step_length/sample_rate, 2)) for i = 1:size(audio_spectrogram, 2)];
y_labels = [string(round(i*sample_rate/window_length/1000, 2)) for i = 1:size(audio_spectrogram, 1)];
heatmap(x_labels, y_labels, 20*log10.(audio_spectrogram))
```
"""
function stft(audio_signal, window_function, step_length)

    # Number of samples and window length
    number_samples = length(audio_signal)
    window_length = length(window_function)

    # Number of time number_frames
    number_times = ceil(Int64, (window_length-step_length+number_samples)/step_length)

    # Zero-padding at the start and end to center the windows
    audio_signal = [zeros(window_length-step_length,1); audio_signal;
    zeros(number_times*step_length-number_samples,1)]

    # Initialize the STFT
    audio_stft = zeros(window_length, number_times)

    # Loop over the time frames
    for time_index = 1:number_times

        # Window the signal
        sample_index = step_length*(time_index-1)
        audio_stft[:, time_index] = audio_signal[1+sample_index:window_length+sample_index].*window_function

    end

    # Fourier transform of the frames
    audio_stft = fft(audio_stft, 1)

end

"""
    audio_istft = z.istft(audio_signal, window_function, step_length);

Compute the inverse short-time Fourier transform (STFT)

# Arguments:
- `audio_stft::Float`: the audio STFT [window_length, number_frames]
- `window_function::Integer`: the window function [window_length, 1]
- `step_length::Integer`: the step length in samples
- `audio_signal::Float`: the audio signal [number_samples, 1]

# Example: Estimate the center and sides signals of a stereo audio file
```
# Stereo audio signal and sample rate in Hz
#Pkg.add("WAV")
using WAV
audio_signal, sample_rate = wavread("audio_file.wav");

# Parameters for the STFT
window_duration = 0.04;
window_length = nextpow2(convert(Int64, window_duration*sample_rate));
window_function = 0.54 - 0.46*cos.(2*pi*(0:window_length-1)/window_length);
step_length = convert(Int64, window_length/2);

# STFT of the left and right channels
include("z.jl")
using z
audio_stft1 = z.stft(audio_signal[:,1], window_function, step_length);
audio_stft2 = z.stft(audio_signal[:,2], window_function, step_length);

# Magnitude spectrogram (with DC component) of the left and right channels
audio_spectrogram1 = abs.(audio_stft1[1:Int(window_length/2)+1, :]);
audio_spectrogram2 = abs.(audio_stft2[1:Int(window_length/2)+1, :]);

# Time-frequency masks of the left and right channels for the center signal
center_mask1 = min(audio_spectrogram1, audio_spectrogram2)./audio_spectrogram1;
center_mask2 = min(audio_spectrogram1, audio_spectrogram2)./audio_spectrogram2;

# STFT of the left and right channels for the center signal (with extension to mirrored frequencies)
center_stft1 = cat(1, center_mask1, center_mask1[Int(window_length/2):-1:2,:]).*audio_stft1;
center_stft2 = cat(1, center_mask2, center_mask2[Int(window_length/2):-1:2,:]).*audio_stft2;

# Synthesized signals of the left and right channels for the center signal
center_signal1 = z.istft(center_stft1, window_function, step_length);
center_signal2 = z.istft(center_stft2, window_function, step_length);

# Final stereo center and sides signals
center_signal = cat(2, center_signal1, center_signal2);
center_signal = center_signal[1:size(audio_signal, 1), :];
sides_signal = audio_signal-center_signal;

# Synthesized center and side signals
wavwrite(center_signal, "center_signal.wav", Fs=sample_rate);
wavwrite(sides_signal, "sides_signal.wav", Fs=sample_rate);

# Spectrogram displayed in dB, s, and kHz
#Pkg.add("Plots")
using Plots
plotly()
p1 = plot(audio_signal, xlabel="Time (s)", title="Original Signal");
p2 = plot(center_signal, xlabel="Time (s)", title="Center Signal");
p3 = plot(sides_signal, xlabel="Time (s)", title="Sides Signal");
plot(p1, p2, p3,layout=(3,1))
```
"""
function istft(audio_stft, window_function, step_length)

    # Window length in samples and number of time frames
    window_length, number_times = size(audio_stft);

    # Number of samples for the signal
    number_samples = (number_times-1)*step_length+window_length;

    # Initialize the signal
    audio_signal = zeros(number_samples, 1);

    # Inverse Fourier transform of the frames and real part to ensure real values
    audio_stft = real(ifft(audio_stft));

    # Loop over the time frames
    for time_index = 1:number_times

        # Constant overlap-add (if proper window and step)
        sample_index = step_length*(time_index-1);
        audio_signal[1+sample_index:window_length+sample_index] = audio_signal[1+sample_index:window_length+sample_index] + audio_stft[:,time_index];

    end

    # Remove the zero-padding at the start and end
    audio_signal = audio_signal[window_length-step_length+1:number_samples-(window_length-step_length)];

    # Un-apply window (just in case)
    audio_signal = audio_signal/sum(window_function[1:step_length:window_length]);

end

end
