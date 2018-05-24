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
- 05/24/18
"""
module z

# Public
#export stft, istft, cqtkernel

"""
    audio_stft = z.stft(audio_signal, window_function, step_length);

Compute the short-time Fourier transform (STFT)

# Arguments:
- `audio_signal::Float`: the audio signal [number_samples, 1]
- `window_function::Integer`: the window function [window_length, 1]
- `step_length::Integer`: the step length in samples
- `audio_stft::Complex`: the audio STFT [window_length, number_frames]

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
window_length = nextpow2(ceil(Int64, window_duration*sample_rate));

# Window function (periodic Hamming window for COLA)
include("z.jl")
using z
window_function = z.hamming(window_length, "periodic");

# Step length in samples (half the window length for COLA)
step_length = convert(Int64, window_length/2);

# Magnitude spectrogram (without the DC component and the mirrored frequencies)
audio_stft = z.stft(audio_signal, window_function, step_length);
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
- `audio_stft::Complex`: the audio STFT [window_length, number_frames]
- `window_function::Integer`: the window function [window_length, 1]
- `step_length::Integer`: the step length in samples
- `audio_signal::Float`: the audio signal [number_samples, 1]

# Example: Estimate the center and sides signals of a stereo audio file
```
# Stereo audio signal and sample rate in Hz
Pkg.add("WAV")
using WAV
audio_signal, sample_rate = wavread("audio_file.wav");

# Parameters for the STFT
include("z.jl")
using z
window_duration = 0.04;
window_length = nextpow2(ceil(Int64, window_duration*sample_rate));
window_function = z.hamming(window_length,"periodic");
step_length = convert(Int64, window_length/2);

# STFT of the left and right channels
audio_stft1 = z.stft(audio_signal[:,1], window_function, step_length);
audio_stft2 = z.stft(audio_signal[:,2], window_function, step_length);

# Magnitude spectrogram (with DC component) of the left and right channels
audio_spectrogram1 = abs.(audio_stft1[1:Int(window_length/2)+1, :]);
audio_spectrogram2 = abs.(audio_stft2[1:Int(window_length/2)+1, :]);

# Time-frequency masks of the left and right channels for the center signal
center_mask1 = min.(audio_spectrogram1, audio_spectrogram2)./audio_spectrogram1;
center_mask2 = min.(audio_spectrogram1, audio_spectrogram2)./audio_spectrogram2;

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
Pkg.add("Plots")
using Plots
plotly()
time_signal = (1:size(audio_signal, 1))/sample_rate;
audio_plot = plot(time_signal, audio_signal, xlabel="Time (s)", title="Original Signal");
center_plot = plot(time_signal, center_signal, xlabel="Time (s)", title="Center Signal");
sides_plot = plot(time_signal, sides_signal, xlabel="Time (s)", title="Sides Signal");
plot(audio_plot, center_plot, sides_plot, layout=(3,1), legend=false)
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
    audio_stft = real(ifft(audio_stft, 1));

    # Loop over the time frames
    for time_index = 1:number_times

        # Constant overlap-add (if proper window and step)
        sample_index = step_length*(time_index-1);
        audio_signal[1+sample_index:window_length+sample_index] = audio_signal[1+sample_index:window_length+sample_index]
        + audio_stft[:,time_index];

    end

    # Remove the zero-padding at the start and end
    audio_signal = audio_signal[window_length-step_length+1:number_samples-(window_length-step_length)];

    # Un-apply window (just in case)
    audio_signal = audio_signal/sum(window_function[1:step_length:window_length]);

end

"""
    cqt_kernel = z.cqtkernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency);

Compute the constant-Q transform (CQT) kernel

# Arguments:
- `sample_rate::Float` sample rate in Hz
- `frequency_resolution::Float` frequency resolution in number of frequency channels per semitone
- `minimum_frequency::Float`: minimum frequency in Hz
- `maximum_frequency::Float`: maximum frequency in Hz
- `cqt_kernel::Float`: CQT kernel [number_frequencies, fft_length]

# Example: Compute and display the CQT kernel
```
# CQT kernel parameters
sample_rate = 44100;
frequency_resolution = 2;
minimum_frequency = 55;
maximum_frequency = sample_rate/2;

# CQT kernel
cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);

# Magnitude CQT kernel displayed
figure
imagesc(abs(cqt_kernel))
axis xy
colormap(jet)
title('Magnitude CQT kernel')
xlabel('FFT length')
ylabel('CQT frequency')
set(gca,'FontSize',30)
```
"""
function cqtkernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency)

    # Number of frequency channels per octave
    octave_resolution = 12*frequency_resolution;

    # Constant ratio of frequency to resolution (= fk/(fk+1-fk))
    quality_factor = 1/(2^(1/octave_resolution)-1);

    # Number of frequency channels for the CQT
    number_frequencies = round(Int64, octave_resolution*log2(maximum_frequency/minimum_frequency));

    # Window length for the FFT (= window length of the minimum frequency = longest window)
    fft_length = nextpow2(ceil(Int64, quality_factor*sample_rate/minimum_frequency));

    # Initialize the kernel
    cqt_kernel = zeros(Complex128,number_frequencies, fft_length);

    # Loop over the frequency channels
    #for frequency_index = 1:number_frequencies

        # Frequency value (in Hz)
        #frequency_value = minimum_frequency*2^((frequency_index-1)/octave_resolution);

        # Window length (nearest odd value because the complex exponential will have an odd length, in samples)
        #window_length = 2*round(Int64, quality_factor*sample_rate/frequency_value/2)+1;

        # Temporal kernel (without zero-padding, odd and symmetric)
        #temporal_kernel = hamming(window_length, "symmetric").*exp.(2*pi*im*quality_factor*(-(window_length-1)/2:(window_length-1)/2)/window_length)/window_length;

        # Pre and post zero-padding to center FFTs
        #temporal_kernel = cat(2, zeros(1, convert(Int64, (fft_length-window_length+1)/2)), temporal_kernel', zeros(1, convert(Int64, (fft_length-window_length-1)/2)));

        # Spectral kernel (mostly real because temporal kernel almost symmetric)
        #spectral_kernel = fft(temporal_kernel, 1);

        # Save the spectral kernels
        #cqt_kernel[frequency_index, :] = spectral_kernel;

    #end

    # Energy threshold for making the kernel sparse
    #energy_threshold = 0.01;

    # Make the CQT kernel sparser
    #cqt_kernel(abs(cqt_kernel)<energy_threshold) = 0;

    # Make the CQT kernel sparse
    #cqt_kernel = sparse(cqt_kernel);

    # From Parseval's theorem
    #cqt_kernel = conj(cqt_kernel)/fft_length;

end

"Compute the Hamming window"
function hamming(window_length,window_sampling="symmetric")

    if window_sampling == "symmetric"
        window_function = 0.54 - 0.46*cos.(2*pi*(0:window_length-1)/(window_length-1))
    elseif window_sampling == "periodic"
        window_function = 0.54 - 0.46*cos.(2*pi*(0:window_length-1)/window_length)
    else
        error("Window sampling must be either 'symmetric or 'periodic'.")
    end

end

end
