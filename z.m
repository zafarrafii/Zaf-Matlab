classdef z
    % Z This class implements few basic methods for audio signal processing
    %
    % Z Methods:
    %   stft - Short-time Fourier transform (STFT)
    %   istft - Inverse STFT
    %   cqtkernel - Constant-Q transform (CQT) kernel
    %   cqtspectrogram - CQT spectrogram using a kernel
    %   cqtchromagram - CQT chromagram using a kernel
    %   mfcc - Mel frequency cepstrum coefficients (MFCCs)
    %   mdct - Modified discrete cosine transform (MDCT)
    %   imdct - Inverse MDCT
    %
    % Author
    %   Zafar Rafii
    %   zafarrafii@gmail.com
    %   08/24/17
    %   
    % See also http://zafarrafii.com
    
    methods (Static = true)
        
        function audio_stft = stft(audio_signal,window_function,step_length)
            % stft Short-time Fourier transform (STFT)
            %   audio_stft = z.stft(audio_signal,window_function,step_length);
            %   
            %   Arguments:
            %       audio_signal: audio signal [number_samples,1]
            %       window_function: window function [window_length,1]
            %       step_length: step length in samples
            %       audio_stft: audio STFT [window_length,number_frames]
            %   
            %   Example: Compute and display the spectrogram of an audio file
            %       % Audio signal averaged over its channels and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
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
            %       % Magnitude spectrogram (without the DC component and the mirrored frequencies)
            %       audio_stft = z.stft(audio_signal,window_function,step_length);
            %       audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));
            %       
            %       % Spectrogram displayed in dB, s, and kHz
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
            %   See also fft, z.istft, spectrogram
            
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
            % istft Inverse short-time Fourier transform (STFT)
            %   audio_signal = z.istft(audio_stft,window_function,step_length);
            %   
            %   Arguments:
            %       audio_stft: audio STFT [window_length,number_frames]
            %       window_function: window function [window_length,1]
            %       step_length: step length in samples
            %       audio_signal: audio signal [number_samples,1]
            %   
            %   Example: Estimate the center and sides signals of a stereo audio file
            %       % Stereo signal and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       
            %       % Parameters for the STFT
            %       window_duration = 0.04;
            %       window_length = 2^nextpow2(window_duration*sample_rate);
            %       window_function = hamming(window_length,'periodic');
            %       step_length = window_length/2;
            %       
            %       % STFT of the left and right channels
            %       audio_stft1 = z.stft(audio_signal(:,1),window_function,step_length);
            %       audio_stft2 = z.stft(audio_signal(:,2),window_function,step_length);
            %       
            %       % Magnitude spectrogram (with DC component) of the left and right channels
            %       audio_spectrogram1 = abs(audio_stft1(1:window_length/2+1,:));
            %       audio_spectrogram2 = abs(audio_stft2(1:window_length/2+1,:));
            %       
            %       % Time-frequency mask of the left and right channels of the center signal
            %       center_mask1 = min(audio_spectrogram1,audio_spectrogram2)./audio_spectrogram1;
            %       center_mask2 = min(audio_spectrogram1,audio_spectrogram2)./audio_spectrogram2;
            %       
            %       % STFT of the left and right channels of the center signal (with extension to mirrored frequencies)
            %       center_stft1 = cat(1,center_mask1,flipud(center_mask1(2:end-1,:))).*audio_stft1;
            %       center_stft2 = cat(1,center_mask2,flipud(center_mask2(2:end-1,:))).*audio_stft2;
            %       
            %       % Synthesized signals of the left and right channels of the center signal
            %       center_signal1 = z.istft(center_stft1,window_function,step_length);
            %       center_signal2 = z.istft(center_stft2,window_function,step_length);
            %       
            %       % Finalized stereo center and sides signals
            %       center_signal = cat(2,center_signal1,center_signal2);
            %       center_signal = center_signal(1:length(audio_signal),:);
            %       sides_signal = audio_signal-center_signal;
            %       
            %       % Synthesized center and side signals
            %       audiowrite('center_signal.wav',center_signal,sample_rate);
            %       audiowrite('sides_signal.wav',sides_signal,sample_rate);
            %
            %   See also ifft, z.stft
            
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
            
            % Remove the zero-padding at the start and end
            audio_signal = audio_signal(window_length-step_length+1:number_samples-(window_length-step_length));
            
            % Un-window the signal (just in case)
            audio_signal = audio_signal/sum(window_function(1:step_length:window_length));
            
        end
        
        function cqt_kernel = cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency)
            % cqtkernel Constant-Q transform (CQT) kernel
            %   cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);
            %   
            %   Arguments:
            %       sample_rate: sample rate in Hz
            %       frequency_resolution: frequency resolution in number of frequency channels per semitone
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
            %       % Magnitude CQT kernel displayed
            %       figure
            %       imagesc(abs(cqt_kernel))
            %       colormap(jet)
            %       title('Magnitude CQT kernel')
            %       xlabel('FFT length')
            %       ylabel('CQT frequency')
            %
            %   See also z.cqt, fft
            
            % Number of frequency channels per octave
            octave_resolution = 12*frequency_resolution;
            
            % Constant ratio of frequency to resolution (= fk/(fk+1-fk))
            quality_factor = 1/(2^(1/octave_resolution)-1);
            
            % Number of frequency channels for the CQT
            number_frequencies = round(octave_resolution*log2(maximum_frequency/minimum_frequency));
            
            % Window length for the FFT (= window length of the mininim 
            % frequency = longuest window)
            fft_length = 2^nextpow2(quality_factor*sample_rate/minimum_frequency);
            
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
                
                % Save the spectral kernels
                cqt_kernel(frequency_index,:) = spectral_kernel;
            end
            
            % Energy threshold for making the kernel sparse
            energy_threshold = 0.01;
            
            % Make the CQT kernel sparser
            cqt_kernel(abs(cqt_kernel)<energy_threshold) = 0;
            
            % Make the CQT kernel sparse
            cqt_kernel = sparse(cqt_kernel);
            
            % From Parseval's theorem
            cqt_kernel = conj(cqt_kernel)/fft_length;
            
        end
        
        function audio_spectrogram = cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel)
            % cqtspectrogram Constant-Q transform (CQT) spectrogram using a kernel
            %   audio_spectrogram = z.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);
            %   
            %   Arguments:
            %       audio_signal: audio signal [number_samples,1]
            %       sample_rate: sample rate in Hz
            %       time_resolution: time resolution in number of time frames per second
            %       cqt_kernel: CQT kernel [number_frequencies,fft_length]
            %       audio_spectrogram: audio spectrogram in magnitude [number_frequencies,number_times]
            %   
            %   Example: Compute and display the CQT spectrogram
            %       % Audio file averaged over the channels and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            %       
            %       % CQT kernel
            %       frequency_resolution = 2;
            %       minimum_frequency = 55;
            %       maximum_frequency = 3520;
            %       cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);
            %       
            %       % CQT spectrogram
            %       time_resolution = 25;
            %       audio_spectrogram = z.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);
            %       
            %       % CQT spectrogram displayed in dB, s, and semitones
            %       figure
            %       imagesc(db(audio_spectrogram))
            %       axis xy
            %       colormap(jet)
            %       title('CQT spectrogram (dB)')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/time_resolution))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       yticks(1:12*frequency_resolution:6*12*frequency_resolution)
            %       yticklabels({'A1 (55 Hz)','A2 (110 Hz)','A3 (220 Hz)','A4 (440 Hz)','A5 (880 Hz)','A6 (1760 Hz)'})
            %       ylabel('Frequency (semitones)')
            %
            %   See also z.cqtkernel, fft
            
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
                
                % Magnitude CQT using the kernel
                sample_index = step_length*(time_index-1);
                audio_spectrogram(:,time_index) = abs(cqt_kernel...
                    *fft(audio_signal(sample_index+1:sample_index+fft_length)));
            end
            
        end
        
        function audio_chromagram = cqtchromagram(audio_signal,sample_rate,time_resolution,frequency_resolution,cqt_kernel)
            % cqtchromagram Constant-Q transform (CQT) chromagram using a kernel
            %   audio_chromagram = z.cqtchromagram(audio_signal,sample_rate,time_resolution,frequency_resolution,cqt_kernel);
            %   
            %   Arguments:
            %       audio_signal: audio signal [number_samples,1]
            %       sample_rate: sample rate in Hz
            %       time_resolution: time resolution in number of time frames per second
            %       frequency_resolution: frequency resolution in number of frequency channels per semitones
            %       cqt_kernel: CQT kernel [number_frequencies,fft_length]
            %       audio_chromagram: audio chromagram [number_chromas,number_times]
            %   
            %   Example: Compute and display the CQT chromagram
            %       % Audio file averaged over the channels and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            %       
            %       % CQT kernel
            %       frequency_resolution = 2;
            %       minimum_frequency = 55;
            %       maximum_frequency = 3520;
            %       cqt_kernel = z.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);
            %       
            %       % CQT chromagram
            %       time_resolution = 25;
            %       audio_chromagram = z.cqtchromagram(audio_signal,sample_rate,time_resolution,frequency_resolution,cqt_kernel);
            %       
            %       % CQT chromagram displayed in dB, s, and chromas
            %       figure
            %       imagesc(db(audio_chromagram))
            %       axis xy
            %       colormap(jet)
            %       title('CQT chromagram (dB)')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/time_resolution))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       yticks(1:frequency_resolution:12*frequency_resolution)
            %       yticklabels({'A','A#','B','C','C#','D','D#','E','F','F#','G','G#'})
            %       ylabel('Chroma')
            %
            %   See also z.cqtkernel, z.cqtspectrogram
            
            % CQT spectrogram
            audio_spectrogram = z.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);
            
            % Number of frequency channels and time frames
            [number_frequencies,number_times] = size(audio_spectrogram);
            
            % Number of chroma bins
            number_chromas = 12*frequency_resolution;
            
            % Initialize the chromagram
            audio_chromagram = zeros(number_chromas,number_times);
            
            % Loop over the chroma bins
            for chroma_index = 1:number_chromas
                
                % Sum the energy of the frequency channels for every chroma
                audio_chromagram(chroma_index,:) = sum(audio_spectrogram(chroma_index:number_chromas:number_frequencies,:));
            end
        end
        
        function audio_mfcc = mfcc(audio_signal,sample_rate,number_filters,number_coefficients)
            % mfcc Mel frequency cepstrum coefficients (MFFCs)
            %   audio_mfcc = z.mfcc(audio_signal,sample_rate,number_filters,number_coefficients);
            %   
            %   Arguments:
            %       audio_signal: audio signal [number_samples,1]
            %       sample_rate: sample rate in Hz
            %       number_filters: number of filters
            %       number_coefficients: number of coefficients (without the 0th coefficient)
            %       audio_mfcc: audio MFCCs [number_times,number_coefficients]
            %   
            %   Example: Compute and display the MFCCs, delta MFCCs, and delta-detla MFCCs
            %       % Audio signal averaged over its channels and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            %       
            %       % MFCCs for a given number of filters and coefficients
            %       number_filters = 40;
            %       number_coefficients = 20;
            %       audio_mfcc = z.mfcc(audio_signal,sample_rate,number_filters,number_coefficients);
            %       
            %       % Delta and delta-delta MFCCs
            %       audio_deltamfcc = diff(audio_mfcc,1,2);
            %       audio_deltadeltamfcc = diff(audio_deltamfcc,1,2);
            %       
            %       % MFCCs, delta MFCCs, and delta-delta MFCCs displayed in s
            %       step_length = (2^nextpow2(0.04*sample_rate))/2;
            %       figure
            %       subplot(3,1,1)
            %       plot(audio_mfcc')
            %       title('MFCCs')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       axis tight
            %       subplot(3,1,2)
            %       plot(audio_deltamfcc')
            %       title('Delta MFCCs')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       axis tight
            %       subplot(3,1,3)
            %       plot(audio_deltadeltamfcc')
            %       title('Delta-delta MFCCs')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       axis tight
            %   
            %   See also z.stft, dct
            
            % Window duration in seconds, length in samples, and function, 
            % and step length in samples
            window_duration = 0.04;
            window_length = 2^nextpow2(window_duration*sample_rate);
            window_function = hamming(window_length,'periodic');
            step_length = window_length/2;
            
            % Magnitude spectrogram (without the DC component and the 
            % mirrored frequencies)
            audio_stft = z.stft(audio_signal,window_function,step_length);
            audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));
            
            % Minimum and maximum mel frequencies
            mininum_melfrequency = 2595*log10(1+(sample_rate/window_length)/700);
            maximum_melfrequency = 2595*log10(1+(sample_rate/2)/700);
            
            % Indices of the overlapping filters (linearly spaced in the 
            % mel scale and logarithmically spaced in the linear scale)
            filter_width = 2*(maximum_melfrequency-mininum_melfrequency)/(number_filters+1);
            filter_indices = mininum_melfrequency:filter_width/2:maximum_melfrequency;
            filter_indices = round((700*(10.^(filter_indices/2595)-1))*window_length/sample_rate);
            
            % Initialize the filter bank
            filter_bank = zeros(number_filters,window_length/2);
            
            % Loop over the filters
            for filter_index = 1:number_filters
                                
                % Left and right sides of the triangular overlapping 
                % filters (linspace more accurate than triang or bartlett!)
                filter_bank(filter_index,filter_indices(filter_index):filter_indices(filter_index+1)) ...
                    = linspace(0,1,filter_indices(filter_index+1)-filter_indices(filter_index)+1);
                filter_bank(filter_index,filter_indices(filter_index+1):filter_indices(filter_index+2)) ...
                    = linspace(1,0,filter_indices(filter_index+2)-filter_indices(filter_index+1)+1);
            end
            
            % Discrete cosine transform (DCT) of the log of the magnitude 
            % spectrogram mapped onto the mel scale using the filter bank
            audio_mfcc = dct(log(filter_bank*audio_spectrogram+eps));
            
            % The first coefficients (without the 0th) represent the MFCCs
            audio_mfcc = audio_mfcc(2:number_coefficients+1,:);
            
        end
        
        function audio_mdct = mdct(audio_signal,window_function)
            % mdct Modified discrete cosine transform (MDCT) using the DCT-IV
            %   audio_mdct = z.mdct(audio_signal,window_function);
            %   
            %   Arguments:
            %       audio_signal: audio signal [number_samples,1]
            %       window_function: window function [window_length,1]
            %       audio_mdct: audio MDCT [number_frequencies,number_times]
            %   
            %   Example: Compute and display the MDCT as used in the AC-3 audio coding format
            %       % Audio file averaged over the channels and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            %       
            %       % Kaiser-Bessel-derived (KBD) window as used in the AC-3 audio coding format
            %       window_length = 2048;
            %       alpha_value = 5;
            %       window_function = kaiser(window_length/2+1,alpha_value*pi);
            %       window_function2 = cumsum(window_function(1:window_length/2));
            %       window_function = sqrt([window_function2;window_function2(window_length/2:-1:1)]./sum(window_function));
            %       
            %       % MDCT
            %       audio_mdct = z.mdct(audio_signal,window_function);
            %       
            %       % MDCT displayed in dB, s, and kHz
            %       figure
            %       imagesc(db(audio_mdct))
            %       axis xy
            %       colormap(jet)
            %       title('MDCT (dB)')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/(window_length/2)))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       yticks(round((1e3:1e3:sample_rate/2)/sample_rate*window_length))
            %       yticklabels(1:sample_rate/2*1e-3)
            %       ylabel('Frequency (kHz)')
            %
            %   See also dct, z.imdct
            
            % Window length in number of samples
            window_length = length(window_function);
            
            % Pre and post zero-padding of the signal
            audio_signal = [zeros(window_length/2,1);audio_signal;zeros(window_length/2,1)];
            
            % Transform the vector of samples into a matrix of frames, with 
            % half-overlapping
            audio_signal = buffer(audio_signal,window_length,window_length/2,'nodelay');
            
            % Apply window to every frame
            audio_signal = bsxfun(@times,audio_signal,window_function);
            
            % Time-domain aliasing cancellation (TDAC) principle
            audio_signal = [-audio_signal(3*window_length/4:-1:window_length/2+1,:)-audio_signal(3*window_length/4+1:window_length,:); ...
                audio_signal(1:window_length/4,:)-audio_signal(window_length/2:-1:window_length/4+1,:)];
            
            % DCT-IV
            audio_mdct = dct(audio_signal,'Type',4);
            
        end
        
        function audio_signal = imdct(audio_mdct,window_function)
            % imdct Inverse modified discrete cosine transform (MDCT) using the DCT-IV
            %   audio_signal = z.imdct(audio_mdct,window_function);
            %   
            %   Arguments:
            %       window_function: window function [window_length,1]
            %       audio_mdct: audio MDCT [number_frequencies,number_times]
            %       audio_signal: audio signal [number_samples,1]
            %   
            %   Example: Verify that the MDCT is perfectly invertible
            %       % Audio file averaged over the channels and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            %       
            %       % MDCT with a slope function as used in the Vorbis audio coding format
            %       window_length = 2048;
            %       window_function = sin((pi/2)*sin((pi/window_length)*(1/2:(window_length-1/2))).^2)';
            %       audio_mdct = z.mdct(audio_signal,window_function);
            %       
            %       % Inverse MDCT and error signal
            %       audio_signal2 = z.imdct(audio_mdct,window_function);
            %       audio_signal2 = audio_signal2(1:length(audio_signal));
            %       error_signal = audio_signal-audio_signal2;
            %       
            %       % Original, resynthesized, and error signals
            %       figure
            %       subplot(3,1,1)
            %       plot(audio_signal)
            %       title('Original Signal')
            %       xticks(sample_rate:sample_rate:length(audio_signal))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       axis tight
            %       subplot(3,1,2)
            %       plot(audio_signal2)
            %       title('Resynthesized Signal')
            %       xticks(sample_rate:sample_rate:length(audio_signal))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       axis tight
            %       subplot(3,1,3)
            %       plot(error_signal)
            %       title('Error Signal')
            %       xticks(sample_rate:sample_rate:length(audio_signal))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       axis tight
            %
            %   See also dct, z.mdct
            
            % Number of frequency channels and time frames
            [number_frequencies,number_times] = size(audio_mdct);
            
            % DCT-IV (which is its own inverse)
            audio_mdct = dct(audio_mdct,'Type',4);
            
            % Time-domain aliasing cancellation (TDAC) principe
            audio_mdct = [audio_mdct(number_frequencies/2+1:number_frequencies,:); ...
                -audio_mdct(number_frequencies:-1:number_frequencies/2+1,:); ...
                -audio_mdct(number_frequencies/2:-1:1,:); ...
                -audio_mdct(1:number_frequencies/2,:)];
            
            % Apply window function to every frame
            audio_mdct = bsxfun(@times,audio_mdct,window_function);
            
            % Initialize audio signal
            audio_signal = zeros(number_frequencies*(number_times+1),1);
            
            % Loop over the frames
            frame_index = 1;
            for sample_index = 1:number_frequencies:number_frequencies*number_times
                
                % Recover the signal thanks to the TDAC principle
                audio_signal(sample_index:sample_index+2*number_frequencies-1,1) ...
                    = audio_signal(sample_index:sample_index+2*number_frequencies-1,1)+audio_mdct(:,frame_index);
                frame_index = frame_index+1;
            end
            
            % Remove the pre and post zero-padding
            audio_signal = audio_signal(number_frequencies+1:end-number_frequencies);
            
        end
        
    end
end