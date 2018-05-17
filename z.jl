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
- 05/17/18
"""
module z

# Public
export stft, istft, test

"""
    audio_stft = z.stft(audio_signal, window_function, step_length)

Compute the short-time Fourier transform (STFT)

# Arguments
- `audio_signal::Float`: the audio signal [number_samples, 1]
- `window_function::Integer`: the window function [window_length, 1]
- `step_length::Integer`: the step length in samples
- `audio_stft::Float`: the audio STFT [window_length, number_frames]

# Example

```jldoctest
julia> a = [1 2; 3 4]
2×2 Array{Int64,2}:
 1  2
 3  4
```
"""
function stft(audio_signal,window_function,step_length)

    # Number of samples and window length
    number_samples = length(audio_signal)
    window_length = length(window_function)

    # Number of time number_frames
    number_times = ceil(Int64,(window_length-step_length+number_samples)/step_length)

    # Zero-padding at the start and end to center the windows
    audio_signal = [zeros(window_length-step_length,1);audio_signal;
        zeros(number_times*step_length-number_samples,1)]

    # Initialize the STFT
    audio_stft = zeros(window_length,number_times)

    # Loop over the time frames
    for time_index = 1:number_times

        # Window the signal
        sample_index = step_length*(time_index-1)
        audio_stft[:,time_index] = audio_signal[1+sample_index:window_length+sample_index].*window_function

    end

    # Fourier transform of the frames
    audio_stft = fft(audio_stft,1)

end

"Compute the inverse short-time Fourier transform (STFT)"
function istft()
   println("this is stft")
end

"Compute the Hamming window"
function hamming(window_length,window_sampling="symmetric")

    if window_sampling == "symmetric"
        window_function = 0.54 - 0.46*cos(2*pi*(0:window_length-1)/(window_length-1))
    elseif window_sampling == "periodic"
        window_function = 0.54 - 0.46*cos(2*pi*(0:window_length-1)/window_length)
    else
        error("Window sampling must be either 'symmetric or 'periodic'.")
    end

end


#Pkg.add("WAV")
using WAV

function test()

    audio_signal, sample_rate = wavread("audio_file.wav")
    audio_signal = mean(audio_signal,2)

    window_length = 2048
    step_length = convert(Int64,window_length/2)
    window_function = hamming(window_length,"periodic")
    audio_stft = stft(audio_signal,window_function,step_length)

end

end
