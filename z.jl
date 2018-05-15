"""
This module implements several functions for audio signal processing.

Functions:
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
- 05/15/18
"""
module z

# Public
export stft, istft, test

"""
    audio_stft = z.stft(audio_signal, window_function, step_length)

Compute the short-time Fourier transform (STFT)

# Arguments
- `audio_signal`: audio signal [number_samples, 0]
- `window_function`: window function [window_length, 0]
- `step_length`: step length in samples
- `audio_stft`: audio STFT [window_length, number_frames]

# Example Compute and display the spectrogram of an audio file
```
```
"""
function stft(audio_signal,window_funciton,step_length)

    # Number of samples and window length
   number_samples = length(audio_signal)
   window_length = length(window_function)

   # Number of time number_frames
   number_times = ceil((window_length-step_length+number_samples)/step_length)

end

"istft Inverse short-time Fourier transform (STFT)"
function istft()
   println("this is stft")
end

function test()
    a = a+2
    a*b

end

end
