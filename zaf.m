 classdef zaf
    % zaf This Matlab class implements a number of functions for audio signal analysis.
    %
    % zaf Methods:
	%   stft - Compute the short-time Fourier transform (STFT).
	%   istft - Compute the inverse STFT.
    %   melfilterbank - Compute the mel filterbank.
    %   melspectrogram - Compute the mel spectrogram using a mel filterbank.
    %   mfcc - Compute the mel frequency cepstrum coefficients (MFCCs) using a mel filterbank.
	%   cqtkernel - Compute the constant-Q transform (CQT) kernel.
	%   cqtspectrogram - Compute the CQT spectrogram using a CQT kernel.
	%   cqtchromagram - Compute the CQT chromagram using a CQT kernel.
	%   dct - Compute the discrete cosine transform (DCT) using the fast Fourier transform (FFT).
	%   dst - Compute the discrete sine transform (DST) using the FFT.
	%   mdct - Compute the modified discrete cosine transform (MDCT) using the FFT.
	%   imdct - Compute the inverse MDCT using the FFT.
    %
	% zaf Other:
	%   sigplot - Plot a signal in seconds.
	%   specshow - Display an spectrogram in dB, seconds, and Hz.
    %	melspecshow - Display a mel spectrogram in dB, seconds, and Hz.
    %   mfccshow - Display MFCCs in seconds.
	%   cqtspecshow - Display a CQT spectrogram in dB, seconds, and Hz.
	%   cqtchromshow - Display a CQT chromagram in seconds.
	%
    % Author:
    %   Zafar Rafii
    %   zafarrafii@gmail.com
    %   http://zafarrafii.com
    %   https://github.com/zafarrafii
    %   https://www.linkedin.com/in/zafarrafii/
    %   03/09/21
    
    methods (Static = true)
        
        function audio_stft = stft(audio_signal,window_function,step_length)
            % stft Compute the short-time Fourier transform (STFT).
            %   audio_stft = zaf.stft(audio_signal,window_function,step_length)
            %   
            %   Inputs:
            %       audio_signal: audio signal [number_samples,1]
            %       window_function: window function [window_length,1]
            %       step_length: step length in samples
            %   Output:
            %       audio_stft: audio STFT [window_length,number_frames]
            %   
            %   Example: Compute and display the spectrogram from an audio file.
            %       % Read the audio signal with its sampling frequency in Hz, and average it over its channels
            %       [audio_signal,sampling_frequency] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            %
            %       % Set the window duration in seconds (audio is stationary around 40 milliseconds)
            %       window_duration = 0.04;
            %
            %       % Derive the window length in samples (use powers of 2 for faster FFT and constant overlap-add (COLA))
            %       window_length = 2^nextpow2(window_duration*sampling_frequency);
            %
            %       % Compute the window function (periodic Hamming window for COLA)
            %       window_function = hamming(window_length,'periodic');
            %
            %       % Set the step length in samples (half of the window length for COLA)
            %       step_length = window_length/2;
            %
            %       % Compute the STFT
            %       audio_stft = zaf.stft(audio_signal,window_function,step_length);
            %
            %       % Derive the magnitude spectrogram (without the DC component and the mirrored frequencies)
            %       audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));
            %
            %       % Display the spectrogram in dB, seconds, and Hz
            %       xtick_step = 1;
            %       ytick_step = 1000;
            %       figure
            %       zaf.specshow(audio_spectrogram, length(audio_signal), sampling_frequency, xtick_step, ytick_step);
            %       title('Spectrogram (dB)')
            
            % Get the number of samples and the window length in samples
            number_samples = length(audio_signal);
            window_length = length(window_function);
            
            % Derive the zero-padding length at the start and at the end of the signal to center the windows
            padding_length = floor(window_length/2);
            
            % Compute the number of time frames given the zero-padding at the start and at the end of the signal
            number_times = ceil(((number_samples+2*padding_length)-window_length)/step_length)+1;
            
            % Zero-pad the start and the end of the signal to center the windows
            audio_signal = [zeros(padding_length,1);audio_signal; ...
                zeros((number_times*step_length+(window_length-step_length)-padding_length)-number_samples,1)];
            
            % Initialize the STFT
            audio_stft = zeros(window_length,number_times);
            
            % Loop over the time frames
            i = 0;
            for j = 1:number_times
                
                % Window the signal
                audio_stft(:,j) = audio_signal(i+1:i+window_length).*window_function;
                i = i+step_length;
                
            end
            
            % Compute the Fourier transform of the frames using the FFT
            audio_stft = fft(audio_stft);
            
        end
        
        function audio_signal = istft(audio_stft,window_function,step_length)
            % istft Compute the inverse short-time Fourier transform (STFT).
            %   audio_signal = zaf.istft(audio_stft,window_function,step_length)
            %   
            %   Inputs:
            %       audio_stft: audio STFT [window_length,number_frames]
            %       window_function: window function [window_length,1]
            %       step_length: step length in samples
            %   Output:
            %       audio_signal: audio signal [number_samples,1]
            %   
            %   Example: Estimate the center and sides signals from a stereo audio file.
            %       % Read the (stereo) audio signal with its sampling frequency in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            % 
            %       % Set the parameters for the STFT
            %       window_duration = 0.04;
            %       window_length = 2^nextpow2(window_duration*sample_rate);
            %       window_function = hamming(window_length,'periodic');
            %       step_length = window_length/2;
            % 
            %       % Compute the STFTs for the left and right channels
            %       audio_stft1 = zaf.stft(audio_signal(:,1),window_function,step_length);
            %       audio_stft2 = zaf.stft(audio_signal(:,2),window_function,step_length);
            % 
            %       % Derive the magnitude spectrograms (with DC component) for the left and right channels
            %       audio_spectrogram1 = abs(audio_stft1(1:window_length/2+1,:));
            %       audio_spectrogram2 = abs(audio_stft2(1:window_length/2+1,:));
            % 
            %       % Estimate the time-frequency masks for the left and right channels for the center
            %       center_mask1 = min(audio_spectrogram1,audio_spectrogram2)./audio_spectrogram1;
            %       center_mask2 = min(audio_spectrogram1,audio_spectrogram2)./audio_spectrogram2;
            % 
            %       % Derive the STFTs for the left and right channels for the center (with mirrored frequencies)
            %       center_stft1 = [center_mask1;center_mask1(window_length/2:-1:2,:)].*audio_stft1;
            %       center_stft2 = [center_mask2;center_mask2(window_length/2:-1:2,:)].*audio_stft2;
            % 
            %       % Synthesize the signals for the left and right channels for the center
            %       center_signal1 = zaf.istft(center_stft1,window_function,step_length);
            %       center_signal2 = zaf.istft(center_stft2,window_function,step_length);
            % 
            %       % Derive the final stereo center and sides signals
            %       center_signal = [center_signal1,center_signal2];
            %       center_signal = center_signal(1:length(audio_signal),:);
            %       sides_signal = audio_signal-center_signal;
            % 
            %       % Write the center and sides signals
            %       audiowrite('center_signal.wav',center_signal,sample_rate);
            %       audiowrite('sides_signal.wav',sides_signal,sample_rate);
            % 
            %       % Display the original, center, and sides signals in seconds
            %       xtick_step = 1;
            %       figure
            %       subplot(3,1,1)
            %       zaf.sigplot(audio_signal, sampling_frequency, xtick_step)
            %       ylim([-1,1]), title("Original signal")
            %       subplot(3,1,2)
            %       zaf.sigplot(center_signal, sampling_frequency, xtick_step)
            %       ylim([-1,1]), title("Center signal")
            %       subplot(3,1,3)
            %       zaf.sigplot(sides_signal, sampling_frequency, xtick_step)
            %       ylim([-1,1]), title("Sides signal")

            % Get the window length in samples and the number of time frames
            [window_length,number_times] = size(audio_stft);
            
            % Compute the number of samples for the signal
            number_samples = number_times*step_length+(window_length-step_length);
            
            % Initialize the signal
            audio_signal = zeros(number_samples,1);
            
            % Compute the inverse Fourier transform of the frames and real part to ensure real values
            audio_stft = real(ifft(audio_stft));
            
            % Loop over the time frames
            i = 0;
            for j = 1:number_times
                
                % Perform a constant overlap-add (COLA) of the signal 
                % (with proper window function and step length)
                audio_signal(i+1:i+window_length) ...
                    = audio_signal(i+1:i+window_length)+audio_stft(:,j);
                i = i+step_length;
                
            end
            
            % Remove the zero-padding at the start and at the end of the signal
            audio_signal = audio_signal(window_length-step_length+1:number_samples-(window_length-step_length));
            
            % Normalize the signal by the gain introduced by the COLA (if any)
            audio_signal = audio_signal/sum(window_function(1:step_length:window_length));
            
        end
        
        function mel_filterbank = melfilterbank(sampling_frequency, window_length, number_filters)
            % melfilterbank Compute the mel filterbank.
            %   audio_mfcc = zaf.melfilterbank(audio_signal,sampling_frequency,number_filters,number_coefficients)
            %   
            %   Inputs:
            %       sampling_frequency: sampling frequency in Hz
            %       window_length: window length for the Fourier analysis in samples
            %       number_mels: number of mel filters
            %   Output:
            %       mel_filterbank: mel filterbank (sparse) [number_mels,number_frequencies]
            %   
            %   Example: Compute and display the mel filterbank.
            
            % Compute the minimum and maximum mels
            mininum_melfrequency = 2595*log10(1+(sampling_frequency/window_length)/700);
            maximum_melfrequency = 2595*log10(1+(sampling_frequency/2)/700);
            
            % Derive the width of the half-overlapping filters in the mel scale (constant)
            filter_width = 2*(maximum_melfrequency-mininum_melfrequency)/(number_filters+1);
            
            % Compute the start and end indices of the overlapping filters in the mel scale (linearly spaced)
            filter_indices = mininum_melfrequency:filter_width/2:maximum_melfrequency;
            
            % Derive the indices of the filters in the linear frequency scale (log spaced)
            filter_indices = round(700*(10.^(filter_indices/2595)-1)*window_length/sampling_frequency);
            
            % Initialize the mel filterbank
            mel_filterbank = zeros(number_filters,window_length/2);
            
            % Loop over the filters
            for i = 1:number_filters
                                
                % Compute the left and right sides of the triangular filters
                % (this is more accurate than creating triangular filters directly)
                mel_filterbank(i,filter_indices(i):filter_indices(i+1)) ...
                    = linspace(0,1,filter_indices(i+1)-filter_indices(i)+1);
                mel_filterbank(i,filter_indices(i+1):filter_indices(i+2)) ...
                    = linspace(1,0,filter_indices(i+2)-filter_indices(i+1)+1);
            end
            
            % Make the mel filterbank sparse
            mel_filterbank = sparse(mel_filterbank);
            
        end
        
        function mel_spectrogram = melspectrogram(audio_signal, window_function, step_length, mel_filterbank)
            % melspectrogram Compute the mel spectrogram using a mel filterbank.
            %   mel_spectrogram = zaf.melspectrogram(audio_signal, window_function, step_length, mel_filterbank)
            %   
            %   Inputs:
            %       audio_signal: audio signal [number_samples,1]
            %       window_function: window function [window_length,1]
            %       step_length: step length in samples
            %       mel_filterbank: mel filterbank [number_mels,number_frequencies]
            %   Output:
            %       mel_spectrogram: mel spectrogram [number_mels,number_times]
            %   
            %   Example: Compute and display the mel spectrogram.
            
            % Compute the magnitude spectrogram (without the DC component and the mirrored frequencies)
            audio_stft = zaf.stft(audio_signal,window_function,step_length);
            audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));
            
            % Compute the mel spectrogram by using the filterbank
            mel_spectrogram = mel_filterbank*audio_spectrogram;
    
        end
        
        function audio_mfcc = mfcc(audio_signal,window_function, step_length, mel_filterbank,number_coefficients)
            % mfcc Compute the mel frequency cepstrum coefficients (MFFCs) using a mel filterbank.
            %   audio_mfcc = zaf.mfcc(audio_signal,sampling_frequency,number_filters,number_coefficients)
            %   
            %   Inputs:
            %       audio_signal: audio signal [number_samples,1]
            %       window_function: window function [window_length,1]
            %       step_length: step length in samples
            %       mel_filterbank: mel filterbank [number_mels,number_frequencies]
            %       number_coefficients: number of coefficients (without the 0th coefficient)
            %   Output:
            %       audio_mfcc: audio MFCCs [number_coefficients, number_times]
            %   
            %   Example: Compute and display the MFCCs, delta MFCCs, and delta-detla MFCCs.
            
            % Compute the magnitude spectrogram (without the DC component and the mirrored frequencies)
            audio_stft = zaf.stft(audio_signal,window_function,step_length);
            audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));
            
            % Compute the discrete cosine transform of the log magnitude spectrogram 
            % mapped onto the mel scale using the filter bank
            audio_mfcc = dct(log(mel_filterbank*audio_spectrogram+eps));
            
            % Keep only the first coefficients (without the 0th)
            audio_mfcc = audio_mfcc(2:number_coefficients+1,:);
            
        end
        
        function cqt_kernel = cqtkernel(sampling_frequency,frequency_resolution,minimum_frequency,maximum_frequency)
            % cqtkernel Compute the constant-Q transform (CQT) kernel.
            %   cqt_kernel = zaf.cqtkernel(sampling_frequency,frequency_resolution,minimum_frequency,maximum_frequency)
            %   
            %   Inputs:
            %       sampling_frequency: sample frequency in Hz
            %       frequency_resolution: frequency resolution in number of frequency channels per semitone
            %       minimum_frequency: minimum frequency in Hz
            %       maximum_frequency: maximum frequency in Hz
            %   Output:
            %       cqt_kernel: CQT kernel (sparse) [number_frequencies,fft_length]
            %   
            %   Example: Compute and display the CQT kernel.
            %       % Set the parameters for the CQT kernel
            %       sampling_frequency = 44100;
            %       frequency_resolution = 2;
            %       minimum_frequency = 55;
            %       maximum_frequency = sampling_frequency/2;
            % 
            %       % Compute the CQT kernel
            %       cqt_kernel = zaf.cqtkernel(sampling_frequency,frequency_resolution,minimum_frequency,maximum_frequency);
            % 
            %       % Display the magnitude CQT kernel
            %       figure
            %       imagesc(abs(cqt_kernel))
            %       axis xy
            %       colormap(jet)
            %       title('Magnitude CQT kernel')
            %       xlabel('FFT length')
            %       ylabel('CQT frequency')

            % Derive the umber of frequency channels per octave
            octave_resolution = 12*frequency_resolution;
            
            % Compute the constant ratio of frequency to resolution (= fk/(fk+1-fk))
            quality_factor = 1/(2^(1/octave_resolution)-1);
            
            % Compute the number of frequency channels for the CQT
            number_frequencies = round(octave_resolution*log2(maximum_frequency/minimum_frequency));
            
            % Compute the window length for the FFT (= longest window for the minimum frequency)
            fft_length = 2^nextpow2(quality_factor*sampling_frequency/minimum_frequency);
            
            % Initialize the CQT kernel
            cqt_kernel = zeros(number_frequencies,fft_length);
            
            % Loop over the frequency channels
            for i = 1:number_frequencies
                
                % Derive the frequency value in Hz
                frequency_value = minimum_frequency*2^((i-1)/octave_resolution);
                
                % Compute the window length in samples 
                % (nearest odd value to center the temporal kernel on 0)
                window_length = 2*round(quality_factor*sampling_frequency/frequency_value/2)+1;
                
                % Compute the temporal kernel for the current frequency (odd and symmetric)
                temporal_kernel = hamming(window_length,'symmetric')' ... 
                    .*exp(2*pi*1j*quality_factor ...
                    *(-(window_length-1)/2:(window_length-1)/2)/window_length)/window_length;
                
                % Derive the pad width to center the temporal kernels
                pad_width = (fft_length-window_length+1)/2;
                
                % Save the current temporal kernel at the center
                %(the zero-padded temporal kernels are not perfectly symmetric anymore because of the even length here)
                cqt_kernel(i,pad_width+1:pad_width+window_length) = temporal_kernel;
                
            end
            
            % Derive the spectral kernels by taking the FFT of the temporal kernels
            % (the spectral kernels are almost real because the temporal kernels are almost symmetric)
             cqt_kernel = fft(cqt_kernel,[],2);
            
            % Make the CQT kernel sparser by zeroing magnitudes below a threshold
            cqt_kernel(abs(cqt_kernel)<0.01) = 0;
            
            % Make the CQT kernel sparse
            cqt_kernel = sparse(cqt_kernel);
            
            % Get the final CQT kernel by using Parseval's theorem
            cqt_kernel = conj(cqt_kernel)/fft_length;
            
        end
        
        function cqt_spectrogram = cqtspectrogram(audio_signal,sampling_frequency,time_resolution,cqt_kernel)
            % cqtspectrogram Compute the constant-Q transform (CQT) spectrogram using a CQT kernel.
            %   cqt_spectrogram = zaf.cqtspectrogram(audio_signal,sampling_frequency,time_resolution,cqt_kernel);
            %   
            %   Inputs:
            %       audio_signal: audio signal [number_samples,1]
            %       sampling_frequency: sampling frequency in Hz
            %       time_resolution: time resolution in number of time frames per second
            %       cqt_kernel: CQT kernel [number_frequencies,fft_length]
            %   Output:
            %       cqt_spectrogram: CQT spectrogram [number_frequencies,number_times]
            %   
            %   Example: Compute and display the CQT spectrogram.
            %       % Read the audio signal with its sampling frequency in Hz, and average it over its channels
            %       [audio_signal,sampling_frequency] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            % 
            %       % Compute the CQT kernel using some parameters
            %       frequency_resolution = 2;
            %       minimum_frequency = 55;
            %       maximum_frequency = 3520;
            %       cqt_kernel = zaf.cqtkernel(sampling_frequency,frequency_resolution,minimum_frequency,maximum_frequency);
            % 
            %       % Compute the (magnitude) CQT spectrogram using the kernel
            %       time_resolution = 25;
            %       cqt_spectrogram = zaf.cqtspectrogram(audio_signal,sampling_frequency,time_resolution,cqt_kernel);
            % 
            %       % Display the CQT spectrogram in dB, seconds, and Hz
            %       xtick_step = 1;
            %       figure
            %       zaf.cqtspecshow(cqt_spectrogram,time_resolution,frequency_resolution,minimum_frequency,xtick_step);
            %       title('CQT spectrogram (dB)')
            
            % Derive the number of time samples per time frame
            step_length = round(sampling_frequency/time_resolution);
            
            % Compute the number of time frames
            number_times = floor(length(audio_signal)/step_length);
            
            % Get the number of frequency channels and the FFT length
            [number_frequencies,fft_length] = size(cqt_kernel);
            
            % Zero-pad the signal to center the CQT
            audio_signal = [zeros(ceil((fft_length-step_length)/2),1); ...
                audio_signal;zeros(floor((fft_length-step_length)/2),1)];
            
            % Initialize the CQT spectrogram
            cqt_spectrogram = zeros(number_frequencies,number_times);
            
            % Loop over the time frames
            i = 0;
            for j = 1:number_times
                
                % Compute the magnitude CQT using the kernel
                cqt_spectrogram(:,j) = abs(cqt_kernel...
                    *fft(audio_signal(i+1:i+fft_length)));
                i = i+step_length;
                
            end
            
        end
        
        function cqt_chromagram = cqtchromagram(audio_signal,sampling_frequency,time_resolution,frequency_resolution,cqt_kernel)
            % cqtchromagram Compute the constant-Q transform (CQT) chromagram using a CQT kernel.
            %   cqt_chromagram = zaf.cqtchromagram(audio_signal,sampling_frequency,time_resolution,frequency_resolution,cqt_kernel)
            %   
            %   Inputs:
            %       audio_signal: audio signal [number_samples,1]
            %       sampling_frequency: sample frequency in Hz
            %       time_resolution: time resolution in number of time frames per second
            %       frequency_resolution: frequency resolution in number of frequency channels per semitones
            %       cqt_kernel: CQT kernel [number_frequencies,fft_length]
            %   Output:
            %       cqt_chromagram: CQT chromagram [number_chromas,number_times]
            %   
            %   Example: Compute and display the CQT chromagram.
            %       % Read the audio signal with its sampling frequency in Hz, and average it over its channels
            %       [audio_signal,sampling_frequency] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            % 
            %       % Compute the CQT kernel using some parameters
            %       frequency_resolution = 2;
            %       minimum_frequency = 55;
            %       maximum_frequency = 3520;
            %       cqt_kernel = zaf.cqtkernel(sampling_frequency,frequency_resolution,minimum_frequency,maximum_frequency);
            % 
            %       % Compute the CQT chromagram
            %       time_resolution = 25;
            %       cqt_chromagram = zaf.cqtchromagram(audio_signal,sampling_frequency,time_resolution,frequency_resolution,cqt_kernel);
            % 
            %       % Display the CQT chromagram in seconds
            %       xtick_step = 1;
            %       figure
            %       zaf.cqtchromshow(cqt_chromagram,time_resolution,xtick_step)
            %       title('CQT chromagram')
            
            % Compute the CQT spectrogram
            cqt_chromagram = zaf.cqtspectrogram(audio_signal,sampling_frequency,time_resolution,cqt_kernel);
            
            % Get the number of frequency channels and time frames
            [number_frequencies,number_times] = size(cqt_chromagram);
            
            % Derive the number of chroma channels
            number_chromas = 12*frequency_resolution;
            
            % Initialize the CQT chromagram
            cqt_chromagram = zeros(number_chromas,number_times);
            
            % Loop over the chroma bins
            for i = 1:number_chromas
                
                % Sum the energy of the frequency channels for every chroma
                cqt_chromagram(i,:) = sum(cqt_chromagram(i:number_chromas:number_frequencies,:),1);
                
            end
            
        end
        
        function audio_dct = dct(audio_signal,dct_type)
            % dct Compute the discrete cosine transform (DCT) using the fast Fourier transform (FFT).
            %   audio_dct = zaf.dct(audio_signal,dct_type)
            %   
            %   Inputs:
            %       audio_signal: audio signal [number_samples,1]
            %       dct_type: DCT type (1, 2, 3, or 4)
            %   Output:
            %       audio_dct: audio DCT [number_frequencies,1]
            %   
            %   Example: Compute the 4 different DCTs and compare them to MATLAB's DCTs.
            %       % Read the audio signal with its sampling frequency in Hz, and average it over its channels
            %       [audio_signal,sampling_frequency] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            % 
            %       % Get an audio segment for a given window length
            %       window_length = 1024;
            %       audio_segment = audio_signal(1:window_length);
            % 
            %       % Compute the DCT-I, II, III, and IV
            %       audio_dct1 = zaf.dct(audio_segment,1);
            %       audio_dct2 = zaf.dct(audio_segment,2);
            %       audio_dct3 = zaf.dct(audio_segment,3);
            %       audio_dct4 = zaf.dct(audio_segment,4);
            % 
            %       % Compute MATLAB's DCT-I, II, III, and IV
            %       matlab_dct1 = dct(audio_segment,'Type',1);
            %       matlab_dct2 = dct(audio_segment,'Type',2);
            %       matlab_dct3 = dct(audio_segment,'Type',3);
            %       matlab_dct4 = dct(audio_segment,'Type',4);
            % 
            %       % Plot the DCT-I, II, III, and IV, MATLAB's versions, and their differences
            %       figure
            %       subplot(3,4,1), plot(audio_dct1), xlim([0,window_length]), title('DCT-I')
            %       subplot(3,4,2), plot(audio_dct2), xlim([0,window_length]), title('DCT-II')
            %       subplot(3,4,3), plot(audio_dct3), xlim([0,window_length]), title('DCT-III')
            %       subplot(3,4,4), plot(audio_dct4), xlim([0,window_length]), title('DCT-IV')
            %       subplot(3,4,5), plot(matlab_dct1), xlim([0,window_length]), title('MATLAB''s DCT-I')
            %       subplot(3,4,6), plot(matlab_dct2), xlim([0,window_length]), title('MATLAB''s DCT-II')
            %       subplot(3,4,7), plot(matlab_dct3), xlim([0,window_length]), title('MATLAB''s DCT-III')
            %       subplot(3,4,8), plot(matlab_dct4), xlim([0,window_length]), title('MATLAB''s DCT-IV')
            %       subplot(3,4,9), plot(audio_dct1-matlab_dct1), xlim([0,window_length]), title('DCT-I - MATLAB''s DCT-I')
            %       subplot(3,4,10), plot(audio_dct2-matlab_dct2), xlim([0,window_length]), title('DCT-II - MATLAB''s DCT-II')
            %       subplot(3,4,11), plot(audio_dct3-matlab_dct3), xlim([0,window_length]), title('DCT-III - MATLAB''s DCT-III')
            %       subplot(3,4,12), plot(audio_dct4-matlab_dct4), xlim([0,window_length]), title('DCT-IV - MATLAB''s DCT-IV')
            
            % Check if the DCT type is I, II, III, or IV
            switch dct_type
                case 1
                    
                    % Get the number of samples
                    window_length = length(audio_signal);
                    
                    % Pre-process the signal to make the DCT-I matrix orthogonal
                    audio_signal([1,end]) = audio_signal([1,end])*sqrt(2);

                    % Compute the DCT-I using the FFT
                    audio_dct = [audio_signal;audio_signal(end-1:-1:2)];
                    audio_dct = fft(audio_dct);
                    audio_dct = real(audio_dct(1:window_length))/2;
                    
                    % Post-process the results to make the DCT-I matrix orthogonal
                    audio_dct([1,end]) = audio_dct([1,end])/sqrt(2);
                    audio_dct = audio_dct*sqrt(2/(window_length-1));
                    
                case 2
                    
                    % Get the number of samples
                    window_length = length(audio_signal);
                    
                    % Compute the DCT-II using the FFT
                    audio_dct = zeros(4*window_length,1);
                    audio_dct(2:2:2*window_length) = audio_signal;
                    audio_dct(2*window_length+2:2:4*window_length) = audio_signal(end:-1:1);
                    audio_dct = fft(audio_dct);
                    audio_dct = real(audio_dct(1:window_length))/2;

                    % Post-process the results to make the DCT-II matrix orthogonal
                    audio_dct(1) = audio_dct(1)/sqrt(2);
                    audio_dct = audio_dct*sqrt(2/window_length);
                    
                case 3
                    
                    % Get the number of samples
                    window_length = length(audio_signal);
                    
                    % Pre-process the signal to make the DCT-III matrix orthogonal
                    audio_signal(1) = audio_signal(1)*sqrt(2);
                    
                    % Compute the DCT-III using the FFT
                    audio_dct = zeros(4*window_length,1);
                    audio_dct(1:window_length) = audio_signal;
                    audio_dct(window_length+2:2*window_length+1) = -audio_signal(end:-1:1,:);
                    audio_dct(2*window_length+2:3*window_length) = -audio_signal(2:end,:);
                    audio_dct(3*window_length+2:4*window_length) = audio_signal(end:-1:2,:);
                    audio_dct = fft(audio_dct);
                    audio_dct = real(audio_dct(2:2:2*window_length))/4;
                    
                    % Post-process the results to make the DCT-III matrix orthogonal
                    audio_dct = audio_dct*sqrt(2/window_length);
                    
                case 4
                    
                    % Get the number of samples
                    window_length = length(audio_signal);
                    
                    % Compute the DCT-IV using the FFT
                    audio_dct = zeros(8*window_length,1);
                    audio_dct(2:2:2*window_length) = audio_signal;
                    audio_dct(2*window_length+2:2:4*window_length) = -audio_signal(end:-1:1,:);
                    audio_dct(4*window_length+2:2:6*window_length) = -audio_signal;
                    audio_dct(6*window_length+2:2:8*window_length) = audio_signal(end:-1:1,:);
                    audio_dct = fft(audio_dct);
                    audio_dct = real(audio_dct(2:2:2*window_length))/4;
                    
                    % Post-process the results to make the DCT-IV matrix orthogonal
                    audio_dct = audio_dct*sqrt(2/window_length);
                    
            end
            
        end
        
        function audio_dst = dst(audio_signal,dst_type)
            % dst Compute the discrete sine transform (DST) using the fast Fourier transform (FFT).
            %   audio_dst = zaf.dst(audio_signal,dst_type)
            %   
            %   Inputs:
            %       audio_signal: audio signal [number_samples,1]
            %       dst_type: DST type (1, 2, 3, or 4)
            %   Outputs:
            %       audio_dst: audio DST [number_frequencies,1]
            %   
            %   Example: Compute the 4 different DSTs and compare their respective inverses with the original audio.
            %       % Read the audio signal with its sampling frequency in Hz, and average it over its channels
            %       [audio_signal,sampling_frequency] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            % 
            %       % Get an audio segment for a given window length
            %       window_length = 1024;
            %       audio_segment = audio_signal(1:window_length);
            % 
            %       % Compute the DST-I, II, III, and IV
            %       audio_dst1 = zaf.dst(audio_segment,1);
            %       audio_dst2 = zaf.dst(audio_segment,2);
            %       audio_dst3 = zaf.dst(audio_segment,3);
            %       audio_dst4 = zaf.dst(audio_segment,4);
            % 
            %       % Compute their respective inverses, i.e., DST-I, II, III, and IV
            %       audio_idst1 = zaf.dst(audio_dst1,1);
            %       audio_idst2 = zaf.dst(audio_dst2,3);
            %       audio_idst3 = zaf.dst(audio_dst3,2);
            %       audio_idst4 = zaf.dst(audio_dst4,4);
            % 
            %       % Plot the DST-I, II, III, and IV, their respective inverses, and their differences with the original audio segment
            %       figure
            %       subplot(3,4,1), plot(audio_dst1), xlim([0,window_length]), title('DST-I')
            %       subplot(3,4,2), plot(audio_dst2), xlim([0,window_length]), title('DST-II')
            %       subplot(3,4,3), plot(audio_dst3), xlim([0,window_length]), title('DST-III')
            %       subplot(3,4,4), plot(audio_dst4), xlim([0,window_length]), title('DST-IV')
            %       subplot(3,4,5), plot(audio_idst1), xlim([0,window_length]), title('Inverse DST-I (DST-I)')
            %       subplot(3,4,6), plot(audio_idst2), xlim([0,window_length]), title('Inverse DST-II (DST-III)')
            %       subplot(3,4,7), plot(audio_idst3), xlim([0,window_length]), title('Inverse DST-III (DST-II)')
            %       subplot(3,4,8), plot(audio_idst4), xlim([0,window_length]), title('Inverse DST-IV (DST-IV)')
            %       subplot(3,4,9), plot(audio_idst1-audio_segment), xlim([0,window_length]), title('Inverse DST-I - audio segment')
            %       subplot(3,4,10), plot(audio_idst2-audio_segment), xlim([0,window_length]), title('Inverse DST-II - audio segment')
            %       subplot(3,4,11), plot(audio_idst3-audio_segment), xlim([0,window_length]), title('Inverse DST-III - audio segment')
            %       subplot(3,4,12), plot(audio_idst4-audio_segment), xlim([0,window_length]), title('Inverse DST-IV - audio segment')
            
            % Check if the DST type is I, II, III, or IV
            switch dst_type
                case 1
                    
                    % Get the number of samples
                    window_length = length(audio_signal);
                    
                    % Compute the DST-I using the FFT
                    audio_dst = zeros(2*window_length+2,1);
                    audio_dst(2:window_length+1) = audio_signal;
                    audio_dst(window_length+3:end) = -audio_signal(end:-1:1);
                    audio_dst = fft(audio_dst);
                    audio_dst = -imag(audio_dst(2:window_length+1))/2;
                    
                    % Post-process the results to make the DST-I matrix orthogonal
                    audio_dst = audio_dst*sqrt(2/(window_length+1));
                    
                case 2
                    
                    % Get the number of samples
                    window_length = length(audio_signal);
                    
                    % Compute the DST-II using the FFT
                    audio_dst = zeros(4*window_length,1);
                    audio_dst(2:2:2*window_length) = audio_signal;
                    audio_dst(2*window_length+2:2:4*window_length) = -audio_signal(end:-1:1);
                    audio_dst = fft(audio_dst);
                    audio_dst = -imag(audio_dst(2:window_length+1))/2;
                    
                    % Post-process the results to make the DST-II matrix orthogonal
                    audio_dst(end) = audio_dst(end)/sqrt(2);
                    audio_dst = audio_dst*sqrt(2/window_length);
                    
                case 3
                    
                    % Get the number of samples
                    window_length = length(audio_signal);
                    
                    % Pre-process the signal to make the DST-III matrix orthogonal
                    audio_signal(end) = audio_signal(end)*sqrt(2);
                    
                    % Compute the DST-III using the FFT
                    audio_dst = zeros(4*window_length,1);
                    audio_dst(2:window_length+1) = audio_signal;
                    audio_dst(window_length+2:2*window_length) = audio_signal(end-1:-1:1);
                    audio_dst(2*window_length+2:3*window_length+1) = -audio_signal;
                    audio_dst(3*window_length+2:4*window_length) = -audio_signal(end-1:-1:1);
                    audio_dst = fft(audio_dst);
                    audio_dst = -imag(audio_dst(2:2:2*window_length))/4;
                    
                    % Post-process the results to make the DST-III matrix orthogonal
                    audio_dst = audio_dst*sqrt(2/window_length);
                    
                case 4
                    
                    % Get the number of samples
                    window_length = length(audio_signal);
                    
                    % Compute the DST-IV using the FFT
                    audio_dst = zeros(8*window_length,1);
                    audio_dst(2:2:2*window_length) = audio_signal;
                    audio_dst(2*window_length+2:2:4*window_length) = audio_signal(end:-1:1);
                    audio_dst(4*window_length+2:2:6*window_length) = -audio_signal;
                    audio_dst(6*window_length+2:2:8*window_length) = -audio_signal(end:-1:1);
                    audio_dst = fft(audio_dst);
                    audio_dst = -imag(audio_dst(2:2:2*window_length))/4;
                    
                    % Post-process the results to make the DST-IV matrix orthogonal
                    audio_dst = audio_dst*sqrt(2/window_length);
                    
            end
            
        end
        
        function audio_mdct = mdct(audio_signal,window_function)
            % mdct Compute the modified discrete cosine transform (MDCT) using the FFT.
            %   audio_mdct = zaf.mdct(audio_signal,window_function)
            %   
            %   Inputs:
            %       audio_signal: audio signal [number_samples,1]
            %       window_function: window function [window_length,1]
            %   Output:
            %       audio_mdct: audio MDCT [number_frequencies,number_times]
            %   
            %   Example: Compute and display the MDCT as used in the AC-3 audio coding format.
            %       % Read the audio signal with its sampling frequency in Hz, and average it over its channels
            %       [audio_signal,sampling_frequency] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            % 
            %       % Compute the Kaiser-Bessel-derived (KBD) window as used in the AC-3 audio coding format
            %       window_length = 512;
            %       alpha_value = 5;
            %       window_function = kaiser(window_length/2+1,alpha_value*pi);
            %       window_function2 = cumsum(window_function(1:window_length/2));
            %       window_function = sqrt([window_function2; window_function2(window_length/2:-1:1)]/sum(window_function));
            % 
            %       % Compute the MDCT
            %       audio_mdct = zaf.mdct(audio_signal,window_function);
            % 
            %       % Display the MDCT in dB, seconds, and Hz
            %       xtick_step = 1;
            %       ytick_step = 1000;
            %       figure
            %       zaf.specshow(abs(audio_mdct),length(audio_signal),sampling_frequency,xtick_step,ytick_step)
            %       title('MDCT (dB)')
            
            % Get the number of samples and the window length in samples
            number_samples = length(audio_signal);
            window_length = length(window_function);
            
            % Derive the step length and the number of frequencies (for clarity)
            step_length = window_length/2;
            number_frequencies = window_length/2;
            
            % Derive the number of time frames
            number_times = ceil(number_samples/step_length)+1;
            
            % Zero-pad the start and the end of the signal to center the windows
            audio_signal = [zeros(step_length,1);audio_signal; ...
                zeros((number_times+1)*step_length-number_samples,1)];
            
            % Initialize the MDCT
            audio_mdct = zeros(number_frequencies,number_times);
            
            % Prepare the pre-processing and post-processing arrays
            preprocessing_array = exp(-1j*pi/window_length*(0:window_length-1)');
            postprocessing_array = exp(-1j*pi/window_length ...
                *(window_length/2+1)*(0.5:window_length/2-0.5)');
            
            % Loop over the time frames
            % (do the pre and post-processing, and take the FFT in the loop to avoid storing twice longer frames)
            i = 0;
            for j = 1:number_times
                
                % Window the signal
                audio_segment = audio_signal(i+1:i+window_length).*window_function;
                i = i+step_length;
                
                % Compute the Fourier transform of the windowed segment using the FFT after pre-processing
                audio_segment = fft(audio_segment.*preprocessing_array);

                % Truncate to the first half before post-processing (and take the real to ensure real values)
                audio_mdct(:,j) = real(audio_segment(1:number_frequencies).*postprocessing_array);
                
            end
            
        end
        
        function audio_signal = imdct(audio_mdct,window_function)
            % imdct Compute the inverse modified discrete cosine transform (MDCT) using the FFT.
            %   audio_signal = zaf.imdct(audio_mdct,window_function)
            %   
            %   Inputs:
            %       audio_mdct: audio MDCT [number_frequencies,number_times]
            %       window_function: window function [window_length,1]
            %   Output:
            %       audio_signal: audio signal [number_samples,1]
            %   
            %   Example: Verify that the MDCT is perfectly invertible.
            %       % Read the audio signal with its sampling frequency in Hz, and average it over its channels
            %       [audio_signal,sampling_frequency] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            % 
            %       % Compute the MDCT with a slope function as used in the Vorbis audio coding format
            %       window_length = 2048;
            %       window_function = sin(pi/2*(sin(pi/window_length*(0.5:window_length-0.5)').^2));
            %       audio_mdct = zaf.mdct(audio_signal,window_function);
            % 
            %       % Compute the inverse MDCT
            %       audio_signal2 = zaf.imdct(audio_mdct,window_function);
            %       audio_signal2 = audio_signal2(1:length(audio_signal));
            % 
            %       % Compute the differences between the original signal and the resynthesized one
            %       audio_differences = audio_signal-audio_signal2;
            %       y_max = max(abs(audio_differences));
            % 
            %       % Display the original and resynthesized signals, and their differences in seconds
            %       xtick_step = 1;
            %       figure
            %       subplot(3,1,1)
            %       zaf.sigplot(audio_signal,sampling_frequency,xtick_step)
            %       ylim([-1,1]), title('Original signal')
            %       subplot(3,1,2)
            %       zaf.sigplot(audio_signal2,sampling_frequency,xtick_step)
            %       ylim([-1,1]), title('Resyntesized signal')
            %       subplot(3,1,3)
            %       zaf.sigplot(audio_differences,sampling_frequency,xtick_step)
            %       ylim([-y_max,y_max]), title('Original - resyntesized signal')
            
            % Get the number of frequency channels and time frames
            [number_frequencies,number_times] = size(audio_mdct);
            
            % Derive the window length and the step length in samples (for clarity)
            window_length = 2*number_frequencies;
            step_length = number_frequencies;
            
            % Derive the number of samples for the signal
            number_samples = step_length*(number_times+1);
            
            % Initialize the audio signal
            audio_signal = zeros(number_samples,1);
            
            % Prepare the pre-processing and post-processing arrays
            preprocessing_array = exp(-1j*pi/(2*number_frequencies) ...
                *(number_frequencies+1)*(0:number_frequencies-1)');
            postprocessing_array = exp(-1j*pi/(2*number_frequencies) ...
                *(0.5+number_frequencies/2:2*number_frequencies+number_frequencies/2-0.5)') ...
                /number_frequencies;
            
            % Compute the Fourier transform of the frames using the FFT after pre-processing
            % (zero-pad to get twice the length)
            audio_mdct = fft(audio_mdct.*preprocessing_array,2*number_frequencies,1);
            
            % Apply the window function to the frames after post-processing
            % (take the real to ensure real values)
            audio_mdct = 2*real(audio_mdct.*postprocessing_array).*window_function;
            
            % Loop over the time frames
            i = 0;
            for j = 1:number_times
                
                % Recover the signal with the time-domain aliasing cancellation (TDAC) principle
                audio_signal(i+1:i+window_length) ...
                    = audio_signal(i+1:i+window_length)+audio_mdct(:,j);
                i = i+step_length;
                
            end
            
            % Remove the zero-padding at the start and at the end of the signal
            audio_signal = audio_signal(step_length+1:end-step_length);
            
        end
        
        function sigplot(audio_signal, sampling_frequency, xtick_step)
            % sigplot Plot a signal in seconds.
            %   zaf.sigplot(audio_signal, sampling_frequency, xtick_step)
            %   
            %   Inputs:
            %       audio_signal: audio signal [number_samples, number_channels]
            %       sampling_frequency: sampling frequency from the original signal in Hz
            %       xtick_step: step for the x-axis ticks in seconds (default: 1 second)
            
            % Set the default values for xtick_step
            if nargin <= 3
                xtick_step = 1;
            end
            
            % Get the number of samples
            number_samples = size(audio_signal,1);
            
            % Prepare the tick locations and labels for the x-axis
            xtick_locations = xtick_step*sampling_frequency:xtick_step*sampling_frequency:number_samples;
            xtick_labels = xtick_step:xtick_step:number_samples/sampling_frequency;
            
            % Plot the signal in seconds
            plot(audio_signal)
            xlim([0,number_samples])
            xticks(xtick_locations)
            xticklabels(xtick_labels)
            xlabel('Time (s)')
            
        end
        
        function specshow(audio_spectrogram, number_samples, sampling_frequency, xtick_step, ytick_step)
            % specshow Display a spectrogram in dB, seconds, and Hz.
            %   zaf.specshow(audio_spectrogram, number_samples, sampling_frequency, xtick_step, ytick_step)
            %   
            %   Inputs:
            %       audio_spectrogram: audio spectrogram (without DC and mirrored frequencies) [number_frequencies, number_times]
            %       number_samples: number of samples from the original signal
            %       sampling_frequency: sampling frequency from the original signal in Hz
            %       xtick_step: step for the x-axis ticks in seconds (default: 1 second)
            %       ytick_step: step for the y-axis ticks in Hz (default: 1000 Hz)
            
            % Set the default values for xtick_step and ytick_step
            if nargin <= 3
                xtick_step = 1;
                ytick_step = 1000;
            end
            
            % Get the number of frequency channels and time frames
            [number_frequencies,number_times] = size(audio_spectrogram);
            
            % Derive the number of Hertz and seconds
            number_hertz = sampling_frequency/2;
            number_seconds = number_samples/sampling_frequency;
            
            % Derive the number of time frames per second and the number of frequency channels per Hz
            time_resolution = number_times/number_seconds;
            frequency_resolution = number_frequencies/number_hertz;
            
            % Prepare the tick locations and labels for the x-axis
            xtick_locations = xtick_step*time_resolution:xtick_step*time_resolution:number_times;
            xtick_labels = xtick_step:xtick_step:number_seconds;
            
            % Prepare the tick locations and labels for the y-axis
            ytick_locations = ytick_step*frequency_resolution:ytick_step*frequency_resolution:number_frequencies;
            ytick_labels = ytick_step:ytick_step:number_hertz;
            
            % Display the spectrogram in dB, seconds, and Hz
            imagesc(db(audio_spectrogram))
            axis xy
            colormap(jet)
            xticks(xtick_locations)
            xticklabels(xtick_labels)
            yticks(ytick_locations)
            yticklabels(ytick_labels)
            xlabel('Time (s)')
            ylabel('Frequency (Hz)')
            
        end
        
        function cqtspecshow(audio_spectrogram,time_resolution,frequency_resolution,minimum_frequency,xtick_step)
            % cqtspecshow Display a CQT spectrogram in dB and seconds, and Hz.
            %   zaf.cqtspecshow(audio_spectrogram,time_resolution,frequency_resolution,minimum_frequency,maximum_frequency,xtick_step)
            %   
            %   Inputs:
            %       audio_spectrogram: CQT audio spectrogram (without DC and mirrored frequencies) [number_frequencies, number_times]
            %       time_resolution: time resolution in number of time frames per second
            %       frequency_resolution: frequency resolution in number of frequency channels per semitone
            %       minimum_frequency: minimum frequency in Hz
            %       xtick_step: step for the x-axis ticks in seconds (default: 1 second)
            
            % Set the default values for xtick_step
            if nargin <= 5
                xtick_step = 1;
            end
            
            % Get the number of frequency channels and time frames
            [number_frequencies,number_times] = size(audio_spectrogram);
            
            % Derive the octave resolution
            octave_resolution = 12*frequency_resolution;
            
            % Prepare the tick locations and labels for the x-axis
            xtick_locations = xtick_step*time_resolution:xtick_step*time_resolution:number_times;
            xtick_labels = xtick_step:xtick_step:number_times/time_resolution;
            
            % Prepare the tick locations and labels for the y-axis
            ytick_locations = 0:octave_resolution:number_frequencies;
            ytick_labels = minimum_frequency*2.^(ytick_locations/octave_resolution);
            
            % Display the spectrogram in dB, seconds, and Hz
            imagesc(db(audio_spectrogram))
            axis xy
            colormap(jet)
            xticks(xtick_locations)
            xticklabels(xtick_labels)
            yticks(ytick_locations)
            yticklabels(ytick_labels)
            xlabel('Time (s)')
            ylabel('Frequency (Hz)')
            
        end
        
        function cqtchromshow(audio_chromagram, time_resolution, xtick_step)
            % cqtchromshow Display a CQT chromagram in seconds.
            %   zaf.cqtchromshow(audio_chromagram, time_resolution, xtick_step)
            %   
            %   Inputs:
            %       audio_chromagram: CQT audio chromagram [number_chromas, number_times]
            %       time_resolution: time resolution in number of time frames per second
            %       xtick_step: step for the x-axis ticks in seconds (default: 1 second)
            
            % Set the default values for xtick_step
            if nargin <= 3
                xtick_step = 1;
            end
            
            % Get the number of time frames
            number_times = size(audio_chromagram,2);
            
            % Prepare the tick locations and labels for the x-axis
            xtick_locations = xtick_step*time_resolution:xtick_step*time_resolution:number_times;
            xtick_labels = xtick_step:xtick_step:number_times/time_resolution;
            
            % Display the chromagram in seconds
            imagesc(audio_chromagram)
            axis xy
            colormap(jet)
            xticks(xtick_locations)
            xticklabels(xtick_labels)
            xlabel('Time (s)')
            ylabel('Chroma')
            
        end
        
    end
end
