classdef z
    % Z This class implements few basic methods for audio signal processing
    %
    % Z Methods:
    %   stft - Short-Time Fourier Transform (STFT)
    %   istft - Inverse STFT
    %   mfcc - Mel Frequency Cepstrum Coefficients (MFCCs)
    %   cqtkernel - Constant Q Transform (CQT) kernel
    %   cqtspectrogram - CQT spectrogram
    %   chromagram - Chromagram
    %
    % See also http://zafarrafii.com
    %
    % Author
    %   Zafar Rafii
    %   zafarrafii@gmail.com
    %   08/21/17
    
    methods (Static = true)
        
        function audio_stft = stft(audio_signal,window_function,step_length)
            % stft Short-Time Fourier Transform (STFT)
            %   audio_stft = z.stft(audio_signal,window_function,step_length);
            %   
            %   Arguments:
            %       audio_signal: single-channel audio signal [number_samples,1]
            %       window_function: one-column window function [window_length,1]
            %       step_length: step length (in samples)
            %       audio_stft: audio STFT [window_length,number_frames]
            %   
            %   Example: Compute and display the spectrogram of an audio file
            %       % Stereo signal and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       
            %       % Window duration in seconds (audio is stationary around 40 milliseconds)
            %       window_duration = 0.04;
            %       
            %       % Window length in samples (power of 2 for fast FFT and constant overlap-add (COLA))
            %       window_length = 2^nextpow2(window_duration*sample_rate);
            %       
            %       % Window function (periodic Hamming window for COLA)
            %       window_function = hamming(window_length,'periodic');
            %       
            %       % Step length in samples (half the window length for COLA)
            %       step_length = window_length/2;
            %       
            %       % STFT of the average over the channels
            %       audio_stft = z.stft(mean(audio_signal,2),window_function,step_length);
            %       
            %       % Magnitude spectrogram (without the DC component and the mirrored frequencies)
            %       audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));
            %       
            %       % Spectrogram displayed in dB, seconds, and kHz
            %       figure
            %       imagesc(db(audio_spectrogram))
            %       axis xy
            %       colormap(jet)
            %       title('Spectrogram (dB)')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       yticks(round((1e3:1e3:sample_rate/2)/sample_rate*window_length))
            %       yticklabels(1:sample_rate/2*1e-3)
            %       ylabel('Frequency (kHz)')
            %
            %   See also fft, istft, spectrogram
            
            % Number of samples
            number_samples = length(audio_signal);
            
            % Window length in samples
            window_length = length(window_function);
            
            % Number of time frames
            number_times = ceil((window_length-step_length+number_samples)/step_length);
            
            % Zero-padding at the start and end to center the windows
            audio_signal = [zeros(window_length-step_length,1);audio_signal; ...
                zeros(number_times*step_length-number_samples,1)];
            
            % Initialize the STFT
            audio_stft = zeros(window_length,number_times);
            
            % Loop over the time frames
            for time_index = 1:number_times
                
                % Window the signal
                sample_index = step_length*(time_index-1);
                audio_stft(:,time_index) ...
                    = audio_signal(1+sample_index:window_length+sample_index).*window_function;
                
            end
            
            % Fourier transform of the frames
            audio_stft = fft(audio_stft);
            
        end
        
        function audio_signal = istft(audio_stft,window_function,step_length)
            % istft Inverse Short-Time Fourier Transform
            %   audio_signal = z.istft(audio_stft,window_function,step_length);
            %   
            %   Arguments:
            %       audio_stft: audio STFT [window_length,number_frames]
            %       window_function: one-column window function [window_length,1]
            %       step_length: step length (in samples)
            %       audio_signal: single-channel audio signal [number_samples,1]
            %   
            %   Example: Generate time-frequency masks to synthesize the center and side channels of a stereo audio file
            %       % Stereo signal and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       
            %       % Parameters for the STFT (see also stft)
            %       window_duration = 0.04;
            %       window_length = 2^nextpow2(window_duration*sample_rate);
            %       window_function = hamming(window_length,'periodic');
            %       step_length = window_length/2;
            %       
            %       % STFT of the left and right channels
            %       audio_stft1 = z.stft(audio_signal(:,1),window_function,step_length);
            %       audio_stft2 = z.stft(audio_signal(:,1),window_function,step_length);
            %       
            %       % Magnitude spectrogram (with DC component) of the left and right channels
            %       audio_spectrogram1 = abs(audio_stft1(1:window_length/2+1,:));
            %       audio_spectrogram2 = abs(audio_stft2(1:window_length/2+1,:));
            %       
            %       % Time-frequency mask of the center channel for the left and right channels
            %       audio_mask1 = min(audio_spectrogram1,audio_spectrogram2)./audio_spectrogram1;
            %       audio_mask2 = min(audio_spectrogram1,audio_spectrogram2)./audio_spectrogram2;
            %       
            %       % STFT of the center channel for the left and right channels (with extension to mirrored frequencies)
            %       audio_stft1 = cat(1,audio_mask1,flipud(audio_mask1(2:end-1,:))).*audio_stft1;
            %       audio_stft2 = cat(1,audio_mask2,flipud(audio_mask2(2:end-1,:))).*audio_stft2;
            %       
            %       % Synthesized signal of the center channel for the left and right channels 
            %       audio_signal1 = z.istft(audio_stft1,window_function,step_length);
            %       audio_signal2 = z.istft(audio_stft2,window_function,step_length);
            %       
            %       % Finalized stereo signal of the center channel
            %       audiowrite('center_signal.wav',cat(2,audio_signal1,audio_signal2),sample_rate);
            %
            %   See also ifft, stft
            
            % Number of time frames
            [~,number_times] = size(audio_stft);
            
            % Window length in samples
            window_length = length(window_function);
            
            % Number of samples for the signal
            number_samples = (number_times-1)*step_length+window_length;
            
            % Initialize the signal
            audio_signal = zeros(number_samples,1);
            
            % Inverse Fourier transform of the frames and real part to
            % ensure real values
            audio_stft = real(ifft(audio_stft));
            
            % Loop over the time frames
            for time_index = 1:number_times
                
                % Inverse Fourier transform of the signal (normalized
                % overlap-add if proper window and step)
                sample_index = step_length*(time_index-1);
                audio_signal(1+sample_index:window_length+sample_index) ...
                    = audio_signal(1+sample_index:window_length+sample_index)+audio_stft(:,time_index);
            end
            
            % Remove the zero-padding at the start and the end
            audio_signal = audio_signal(window_length-step_length+1:number_samples-(window_length-step_length));
            
            % Un-window the signal (just in case)
            audio_signal = audio_signal/sum(window_function(1:step_length:window_length));
            
        end
        
        function audio_mfcc = mfcc(audio_signal,sample_rate,number_filters,number_coefficients)
            % mfcc Mel Frequency Cepstrum Coefficients (MFFCs)
            %   audio_mfcc = z.mfcc(audio_signal,sample_rate,number_filters,number_coefficients);
            %   
            %   Arguments:
            %       audio_signal: single-channel audio signal [number_samples,1]
            %       sample_rate: sample rate (in Hz)
            %       number_filters: number of filters 
            %       number_coefficients: number of mel-frequency cepstrum coefficients (without the 0th coefficient)
            %       audio_signal: vector of size [number_samples,1]
            %   
            %   Example: Compute and display the MFCCs, delta MFCCs, and delta-detla MFCCs
            %       % Stereo signal and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       
            %       % MFCC of the average over the channels for a given number of filters and coefficients
            %       number_filters = 40;
            %       number_coefficients = 20;
            %       audio_mfcc = z.mfcc(mean(audio_signal,2),sample_rate,number_filters,number_coefficients);
            %       
            %       % Delta and delta-delta MFCCs
            %       audio_deltamfcc = diff(audio_mfcc,1,2);
            %       audio_deltadeltamfcc = diff(audio_deltamfcc,1,2);
            %       
            %       % Display the MFCCs, delta MFCCs, and delta-delta MFCCs
            %       figure
            %       subplot(3,1,1)
            %       plot(audio_mfcc')
            %       axis tight
            %       title('MFCC')
            %       xlabel('Time frames')
            %       subplot(3,1,2)
            %       plot(audio_deltamfcc')
            %       axis tight
            %       title('Delta MFCC')
            %       xlabel('Time frames')
            %       subplot(3,1,3)
            %       plot(audio_deltadeltamfcc')
            %       axis tight
            %       title('Delta-delta MFCC')
            %       xlabel('Time frames')
            %   
            %   See also z.stft, dct
            
            %%% Compute the spectrogram
            % Window duration in seconds (audio is stationary around 40 
            % milliseconds)
            window_duration = 0.04;
            
            % Window length in samples (power of 2 for fast FFT and 
            % constant overlap-add (COLA))
            window_length = 2^nextpow2(window_duration*sample_rate);
            
            % Window function (periodic Hamming window for COLA)
            window_function = hamming(window_length,'periodic');
            
            % Step length in samples (half the window length for COLA)
            step_length = window_length/2;
            
            % STFT of the average over the channels
            audio_stft = z.stft(mean(audio_signal,2),window_function,step_length);
            
            % Magnitude spectrogram (without the DC component and the 
            % mirrored frequencies)
            audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));
            
            %%% Compute the filter bank
            % Minimum mel frequency
            mininum_melfrequency = 2595*log10(1+(sample_rate/window_length)/700);
            
            % Maximum mel frequency
            maximum_melfrequency = 2595*log10(1+(sample_rate/2)/700);
            
            % Width of a filter in the mel scale
            filter_width = 2*(maximum_melfrequency-mininum_melfrequency)/(number_filters+1);
            
            % Indices of the overlapping filters (linearly spaced in the 
            % mel scale)
            filter_indices = mininum_melfrequency:filter_width/2:maximum_melfrequency;
            
            % Indices of the overlapping filters (logarithmically spaced in 
            % the linear scale)
            filter_indices = round((700*(10.^(filter_indices/2595)-1))*window_length/sample_rate);
            
            % Filter bank
            filter_bank = zeros(number_filters,window_length/2);
            
            % Loop over the filters
            for filter_index = 1:number_filters
                                
                % Left side of the triangular overlapping filter (linspace 
                % gives a cleaner filter bank than triang or bartlett!)
                filter_bank(filter_index,filter_indices(filter_index):filter_indices(filter_index+1)) ...
                    = linspace(0,1,filter_indices(filter_index+1)-filter_indices(filter_index)+1);
                
                % Right side of the triangular overlapping filter
                filter_bank(filter_index,filter_indices(filter_index+1):filter_indices(filter_index+2)) ...
                    = linspace(1,0,filter_indices(filter_index+2)-filter_indices(filter_index+1)+1);
            end
            
            %%% Compute the MFCCs
            % Discrete cosine transform of the log of the magnitude 
            % spectrogram mapped onto the mel scale using the filter bank
            audio_mfcc = dct(log(filter_bank*audio_spectrogram));
            
            % The first coefficients (without the 0th) represent the MFCCs
            audio_mfcc = audio_mfcc(2:number_coefficients+1,:);
            
        end
        
        function cqt_kernel = cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency)
            % cqtkernel Constant Q Transform (CQT) kernel
            %   cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);
            %   
            %   Arguments:
            %       sample_rate: sample rate (in Hz)
            %       frequency_resolution: frequency resolution in number of frequency channels per semitones
            %       minimum_frequency: minimum frequency in Hz
            %       maximum_frequency: maximum frequency in Hz
            %       cqt_kernel: CQT kernel [number_frequencies,fft_length]
            %   
            %   Example: Compute and display the CQT kernel
            %       % CQT kernel parameters
            %       sample_rate = 44100;
            %       frequency_resolution = 2;
            %       minimum_frequency = 55;
            %       maximum_frequency = sample_rate/2;
            %       
            %       % CQT kernel
            %       cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);
            %       
            %       % CQT kernel displayed
            %       figure
            %       imagesc(abs(cqt_kernel))
            %       colormap(jet)
            %       title('CQT kernel (magnitude)')
            %       xlabel('FFT length')
            %       ylabel('CQT frequency')
            %
            %   See also z.cqt, fft
            
            % Number of frequency channels per octave
            octave_resolution = 12*frequency_resolution;
            
            % (constant) ratio of frequency to resolution (= fk/(fk+1-fk))
            quality_factor = 1/(2^(1/octave_resolution)-1);
            
            % Number of frequency channels for the CQT
            number_frequencies = round(octave_resolution*log2(maximum_frequency/minimum_frequency));
            
            % Window length for the FFT (= window length of the mininim frequency = longuest window)
            fft_length = 2^nextpow2(quality_factor*sample_rate/minimum_frequency);
            
            % Energy threshold for making the kernel sparse
            energy_threshold = 0.01;
            
            % Initialize the kernel
            cqt_kernel = zeros(number_frequencies,fft_length);
            
            % Loop over the frequency channels
            for frequency_index = 1:number_frequencies
                
                % Frequency value (in Hz)
                frequency_value = minimum_frequency*2^((frequency_index-1)/octave_resolution);
                
                % Window length (nearest odd value because the complex 
                % exponential will have an odd length, in samples)
                window_length = 2*round(quality_factor*sample_rate/frequency_value/2)+1;
                
                % Temporal kernel (without zero-padding, odd and symmetric)
                temporal_kernel = hamming(window_length,'symmetric')' ... 
                    .*exp(2*pi*1j*quality_factor*(-(window_length-1)/2:(window_length-1)/2)/window_length)/window_length;
                
                % Pre zero-padding to center the FFT's (FFT does post zero-
                % padding; temporal kernel still odd but almost symmetric)
                temporal_kernel = cat(2,zeros(1,(fft_length-window_length+1)/2),temporal_kernel);
                
                % Spectral kernel (mostly real because temporal kernel
                % almost symmetric)
                spectral_kernel = fft(temporal_kernel,fft_length);
                
                % Make the spectral kernel sparser
                spectral_kernel(abs(spectral_kernel)<energy_threshold) = 0;
                
                % Save the spectral kernels
                cqt_kernel(frequency_index,:) = spectral_kernel;
            end
            
            % Make the CQT kernel sparse
            cqt_kernel = sparse(cqt_kernel);
            
            % From Parseval's theorem
            cqt_kernel = conj(cqt_kernel)/fft_length;
            
        end
        
        function audio_spectrogram = cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel)
            % cqtspectrogram CQT spectrogram
            %   audio_spectrogram = z.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);
            %   
            %   Read ...
            %   
            %   Arguments:
            %       audio_signal: single-channel audio signal [number_samples,1]
            %       sample_rate: sample rate (in Hz)
            %       time_resolution: time resolution in number of time frames per second
            %       cqt_kernel: CQT kernel [number_frequencies,fft_length]
            %   
            %   Example: Compute and display the CQT spectrogram
            %       % CQT kernel
            %       sample_rate = 44100;
            %       frequency_resolution = 2;
            %       minimum_frequency = 55;
            %       maximum_frequency = sample_rate/2;
            %       cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);
            %       
            %       % CQT spectrogram
            %       audio_spectrogram = z.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);
            %       
            %       % CQT kernel displayed (in dB, s, and semitones)
            %       figure
            %       imagesc(db(audio_spectrogram))
            %       axis xy
            %       colormap(jet)
            %       title('CQT spectrogram (dB)')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/time_resolution))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       yticks(1:frequency_resolution:frequency_resolution*12)
            %       yticklabels({'A1','A#1','B1','C1','C#1','D1','D#1','E1','F1','F#1','G1','G#1'})
            %       ylabel('Frequency (semitones)')
            %
            %   See also z.cqt, fft
            
            % Number of time samples per time frame
            step_length = round(sample_rate/time_resolution);
            
            % Number of time frames
            number_times = floor(length(audio_signal)/step_length);
            
            % Number of frequency channels and FFT length
            [number_frequencies,fft_length] = size(cqt_kernel);
            
            % Zero-padding to center the CQT
            audio_signal = [zeros(ceil((fft_length-step_length)/2),1); ...
                audio_signal;zeros(floor((fft_length-step_length)/2),1)];
            
            % Initialize the spectrogram
            audio_spectrogram = zeros(number_frequencies,number_times);
            
            % Loop over the time frames
            for time_index = 1:number_times
                
                % CQT with kernel (magnitude)
                sample_index = step_length*(time_index-1);
                audio_spectrogram(:,time_index) = abs(cqt_kernel...
                    *fft(audio_signal(sample_index+1:sample_index+fft_length)));
            end
            
        end
        
        function audio_chromagram = chromagram(audio_signal,sample_rate,time_resolution,frequency_resolution,cqt_kernel)
            % cqtspectrogram CQT spectrogram
            %   audio_spectrogram = z.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);
            %   
            %   Arguments:
            %       audio_signal: single-channel audio signal [number_samples,1]
            %       sample_rate: sample rate (in Hz)
            %       time_resolution: time resolution in number of time frames per second
            %       cqt_kernel: CQT kernel [number_frequencies,fft_length]
            %   
            %   Example: Compute and display the chromagram
            %       
            %
            %   See also z.cqt
            
            % CQT spectrogram
            audio_spectrogram = z.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);
            
            % Number of frequency channels and time frames
            [number_frequencies,number_times] = size(audio_spectrogram);
            
            % Number of chroma bins
            number_chromas = 12*frequency_resolution;
            
            % Chromagram initialized
            audio_chromagram = zeros(number_chromas,number_times);
            
            % Loop over the chroma bins
            for chroma_index = 1:number_chromas
                
                % Sum the energy of the frequency channels for every chroma
                audio_chromagram(chroma_index,:) = sum(audio_spectrogram(chroma_index:number_chromas:number_frequencies,:));
            end
        end
        
    end
end
