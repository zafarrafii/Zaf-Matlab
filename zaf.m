 classdef zaf
    % zaf This Matlab class implements a number of functions for audio signal analysis.
    %
    % zaf Methods:
	%   stft - Compute the short-time Fourier transform (STFT).
	%   istft - Compute the inverse STFT.
	%   cqtkernel - Compute the constant-Q transform (CQT) kernel.
	%   cqtspectrogram - Compute the CQT spectrogram using a CQT kernel.
	%   cqtchromagram - Compute the CQT chromagram using a CQT kernel.
	%   mfcc - Compute the mel frequency cepstrum coefficients (MFCCs).
	%   dct - Compute the discrete cosine transform (DCT) using the fast Fourier transform (FFT).
	%   dst - Compute the discrete sine transform (DST) using the FFT.
	%   mdct - Compute the modified discrete cosine transform (MDCT) using the FFT.
	%   imdct - Compute the inverse MDCT using the FFT.
    %
	% zaf Other:
	%   sigplot - Plot a signal in seconds.
	%   specshow - Display an spectrogram in dB, seconds, and Hz.
	%   cqtspecshow - Display a CQT spectrogram in dB, seconds, and Hz.
	%   cqtchromshow - Display a CQT chromagram in seconds.
	%
    % Author:
    %   Zafar Rafii
    %   zafarrafii@gmail.com
    %   http://zafarrafii.com
    %   https://github.com/zafarrafii
    %   https://www.linkedin.com/in/zafarrafii/
    %   12/09/20
    
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
            % 
            
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
            % istft Inverse short-time Fourier transform (STFT)
            %   audio_signal = zaf.istft(audio_stft,window_function,step_length);
            %   
            %   Arguments:
            %       audio_stft: audio STFT [window_length,number_frames]
            %       window_function: window function [window_length,1]
            %       step_length: step length in samples
            %       audio_signal: audio signal [number_samples,1]
            %   
            %   Example: Estimate the center and sides signals of a stereo audio file
            %       % Stereo audio signal and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       
            %       % Parameters for the STFT
            %       window_duration = 0.04;
            %       window_length = 2^nextpow2(window_duration*sample_rate);
            %       window_function = hamming(window_length,'periodic');
            %       step_length = window_length/2;
            %       
            %       % STFT of the left and right channels
            %       audio_stft1 = zaf.stft(audio_signal(:,1),window_function,step_length);
            %       audio_stft2 = zaf.stft(audio_signal(:,2),window_function,step_length);
            %       
            %       % Magnitude spectrogram (with DC component) of the left and right channels
            %       audio_spectrogram1 = abs(audio_stft1(1:window_length/2+1,:));
            %       audio_spectrogram2 = abs(audio_stft2(1:window_length/2+1,:));
            %       
            %       % Time-frequency masks of the left and right channels for the center signal
            %       center_mask1 = min(audio_spectrogram1,audio_spectrogram2)./audio_spectrogram1;
            %       center_mask2 = min(audio_spectrogram1,audio_spectrogram2)./audio_spectrogram2;
            %       
            %       % STFT of the left and right channels for the center signal (with extension to mirrored frequencies)
            %       center_stft1 = [center_mask1;center_mask1(window_length/2:-1:2,:)].*audio_stft1;
            %       center_stft2 = [center_mask2;center_mask2(window_length/2:-1:2,:)].*audio_stft2;
            %       
            %       % Synthesized signals of the left and right channels for the center signal
            %       center_signal1 = zaf.istft(center_stft1,window_function,step_length);
            %       center_signal2 = zaf.istft(center_stft2,window_function,step_length);
            %       
            %       % Final stereo center and sides signals
            %       center_signal = [center_signal1,center_signal2];
            %       center_signal = center_signal(1:length(audio_signal),:);
            %       sides_signal = audio_signal-center_signal;
            %       
            %       % Synthesized center and side signals
            %       audiowrite('center_signal.wav',center_signal,sample_rate);
            %       audiowrite('sides_signal.wav',sides_signal,sample_rate);
            %       
            %       % Original, center, and sides signals displayed in s
            %       figure
            %       subplot(3,1,1), plot(audio_signal), axis tight, title('Original Signal')
            %       xticks(sample_rate:sample_rate:length(audio_signal))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)'), set(gca,'FontSize',30)
            %       subplot(3,1,2), plot(center_signal), axis tight, title('Center Signal')
            %       xticks(sample_rate:sample_rate:length(audio_signal))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)'), set(gca,'FontSize',30)
            %       subplot(3,1,3), plot(sides_signal), axis tight, title('Sides Signal')
            %       xticks(sample_rate:sample_rate:length(audio_signal))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)'), set(gca,'FontSize',30)
            %   
            %   See also ifft, zaf.stft
            
            % Window length and number of time frames
            [window_length,number_times] = size(audio_stft);
            
            % Number of samples for the signal
            number_samples = (number_times-1)*step_length+window_length;
            
            % Initialize the signal
            audio_signal = zeros(number_samples,1);
            
            % Inverse Fourier transform of the frames and real part to
            % ensure real values
            audio_stft = real(ifft(audio_stft));
            
            % Loop over the time frames
            for time_index = 1:number_times
                
                % Constant overlap-add (if proper window and step)
                sample_index = (time_index-1)*step_length;
                audio_signal(1+sample_index:window_length+sample_index) ...
                    = audio_signal(1+sample_index:window_length+sample_index)+audio_stft(:,time_index);
            end
            
            % Remove the zero-padding at the start and end
            audio_signal = audio_signal(window_length-step_length+1:number_samples-(window_length-step_length));
            
            % Un-apply window (just in case)
            audio_signal = audio_signal/sum(window_function(1:step_length:window_length));
            
        end
        
        function cqt_kernel = cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency)
            % cqtkernel Constant-Q transform (CQT) kernel
            %   cqt_kernel = zaf.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);
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
            %       cqt_kernel = zaf.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);
            %       
            %       % Magnitude CQT kernel displayed
            %       figure
            %       imagesc(abs(cqt_kernel))
            %       axis xy
            %       colormap(jet)
            %       title('Magnitude CQT kernel')
            %       xlabel('FFT length')
            %       ylabel('CQT frequency')
            %       set(gca,'FontSize',30)
            %
            %   See also zaf.cqt, fft
            
            % Number of frequency channels per octave
            octave_resolution = 12*frequency_resolution;
            
            % Constant ratio of frequency to resolution (= fk/(fk+1-fk))
            quality_factor = 1/(2^(1/octave_resolution)-1);
            
            % Number of frequency channels for the CQT
            number_frequencies = round(octave_resolution*log2(maximum_frequency/minimum_frequency));
            
            % Window length for the FFT (= window length of the minimum 
            % frequency = longest window)
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
                
                % Pre zero-padding to center FFTs (fft does post zero-
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
            %   audio_spectrogram = zaf.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);
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
            %       cqt_kernel = zaf.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);
            %       
            %       % CQT spectrogram
            %       time_resolution = 25;
            %       audio_spectrogram = zaf.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);
            %       
            %       % CQT spectrogram displayed in dB, s, and semitones
            %       figure
            %       imagesc(db(audio_spectrogram))
            %       axis xy
            %       colormap(jet)
            %       title('CQT spectrogram (dB)')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*time_resolution))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       yticks(1:12*frequency_resolution:6*12*frequency_resolution)
            %       yticklabels({'A1 (55 Hz)','A2 (110 Hz)','A3 (220 Hz)','A4 (440 Hz)','A5 (880 Hz)','A6 (1760 Hz)'})
            %       ylabel('Frequency (semitones)')
            %       set(gca,'FontSize',30)
            %
            %   See also zaf.cqtkernel, fft
            
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
                sample_index = (time_index-1)*step_length;
                audio_spectrogram(:,time_index) = abs(cqt_kernel...
                    *fft(audio_signal(sample_index+1:sample_index+fft_length)));
                
            end
            
        end
        
        function audio_chromagram = cqtchromagram(audio_signal,sample_rate,time_resolution,frequency_resolution,cqt_kernel)
            % cqtchromagram Constant-Q transform (CQT) chromagram using a kernel
            %   audio_chromagram = zaf.cqtchromagram(audio_signal,sample_rate,time_resolution,frequency_resolution,cqt_kernel);
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
            %       cqt_kernel = zaf.cqtkernel(sample_rate,frequency_resolution,minimum_frequency,maximum_frequency);
            %       
            %       % CQT chromagram
            %       time_resolution = 25;
            %       audio_chromagram = zaf.cqtchromagram(audio_signal,sample_rate,time_resolution,frequency_resolution,cqt_kernel);
            %       
            %       % CQT chromagram displayed in dB, s, and chromas
            %       figure
            %       imagesc(db(audio_chromagram))
            %       axis xy
            %       colormap(jet)
            %       title('CQT chromagram (dB)')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*time_resolution))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)')
            %       yticks(1:frequency_resolution:12*frequency_resolution)
            %       yticklabels({'A','A#','B','C','C#','D','D#','E','F','F#','G','G#'})
            %       ylabel('Chroma')
            %       set(gca,'FontSize',30)
            %
            %   See also zaf.cqtkernel, zaf.cqtspectrogram
            
            % CQT spectrogram
            audio_spectrogram = zaf.cqtspectrogram(audio_signal,sample_rate,time_resolution,cqt_kernel);
            
            % Number of frequency channels and time frames
            [number_frequencies,number_times] = size(audio_spectrogram);
            
            % Number of chroma bins
            number_chromas = 12*frequency_resolution;
            
            % Initialize the chromagram
            audio_chromagram = zeros(number_chromas,number_times);
            
            % Loop over the chroma bins
            for chroma_index = 1:number_chromas
                
                % Sum the energy of the frequency channels for every chroma
                audio_chromagram(chroma_index,:) = sum(audio_spectrogram(chroma_index:number_chromas:number_frequencies,:),1);
                
            end
            
        end
        
        function audio_mfcc = mfcc(audio_signal,sample_rate,number_filters,number_coefficients)
            % mfcc Mel frequency cepstrum coefficients (MFFCs)
            %   audio_mfcc = zaf.mfcc(audio_signal,sample_rate,number_filters,number_coefficients);
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
            %       audio_mfcc = zaf.mfcc(audio_signal,sample_rate,number_filters,number_coefficients);
            %       
            %       % Delta and delta-delta MFCCs
            %       audio_deltamfcc = diff(audio_mfcc,1,2);
            %       audio_deltadeltamfcc = diff(audio_deltamfcc,1,2);
            %       
            %       % MFCCs, delta MFCCs, and delta-delta MFCCs displayed in s
            %       step_length = (2^nextpow2(0.04*sample_rate))/2;
            %       figure
            %       subplot(3,1,1), plot(audio_mfcc'), axis tight, title('MFCCs')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)'), set(gca,'FontSize',30)
            %       subplot(3,1,2), plot(audio_deltamfcc'), axis tight, title('Delta MFCCs')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)'), set(gca,'FontSize',30)
            %       subplot(3,1,3), plot(audio_deltadeltamfcc'), axis tight, title('Delta-delta MFCCs')
            %       xticks(round((1:floor(length(audio_signal)/sample_rate))*sample_rate/step_length))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)'), set(gca,'FontSize',30)
            %   
            %   See also zaf.stft, dct
            
            % Window duration in seconds, length in samples, and function, 
            % and step length in samples
            window_duration = 0.04;
            window_length = 2^nextpow2(window_duration*sample_rate);
            window_function = hamming(window_length,'periodic');
            step_length = window_length/2;
            
            % Magnitude spectrogram (without the DC component and the 
            % mirrored frequencies)
            audio_stft = zaf.stft(audio_signal,window_function,step_length);
            audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));
            
            % Minimum and maximum mel frequencies
            mininum_melfrequency = 2595*log10(1+(sample_rate/window_length)/700);
            maximum_melfrequency = 2595*log10(1+(sample_rate/2)/700);
            
            % Indices of the overlapping filters (linearly spaced in the 
            % mel scale and logarithmically spaced in the linear scale)
            filter_width = 2*(maximum_melfrequency-mininum_melfrequency)/(number_filters+1);
            filter_indices = mininum_melfrequency:filter_width/2:maximum_melfrequency;
            filter_indices = round(700*(10.^(filter_indices/2595)-1)*window_length/sample_rate);
            
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
            
            % Discrete cosine transform of the log of the magnitude 
            % spectrogram mapped onto the mel scale using the filter bank
            audio_mfcc = dct(log(filter_bank*audio_spectrogram+eps));
            
            % The first coefficients (without the 0th) represent the MFCCs
            audio_mfcc = audio_mfcc(2:number_coefficients+1,:);
            
        end
        
        function audio_dct = dct(audio_signal,dct_type)
            % dct Discrete cosine transform (DCT) using the fast Fourier transform (FFT)
            %   audio_dct = zaf.dct(audio_signal,dct_type);
            %   
            %   Arguments:
            %       audio_signal: audio signal [number_samples,number_frames]
            %       dct_type: DCT type (1, 2, 3, or 4)
            %       audio_dct: audio DCT [number_frequencies,number_frames]
            %   
            %   Example: Compute the 4 different DCTs and compare them to Matlab's DCTs
            %       % Audio signal averaged over its channels and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            %       
            %       % Audio signal for a given window length, and one frame
            %       window_length = 1024;
            %       audio_signal = audio_signal(1:window_length,:);
            %       
            %       % DCT-I, II, III, and IV
            %       audio_dct1 = zaf.dct(audio_signal,1);
            %       audio_dct2 = zaf.dct(audio_signal,2);
            %       audio_dct3 = zaf.dct(audio_signal,3);
            %       audio_dct4 = zaf.dct(audio_signal,4);
            %       
            %       % Matlab's DCT-I, II, III, and IV
            %       matlab_dct1 = dct(audio_signal,'Type',1);
            %       matlab_dct2 = dct(audio_signal,'Type',2);
            %       matlab_dct3 = dct(audio_signal,'Type',3);
            %       matlab_dct4 = dct(audio_signal,'Type',4);
            %       
            %       % DCT-I, II, III, and IV, Matlab's versions, and errors displayed
            %       figure
            %       subplot(4,3,1), plot(audio_dct1), axis tight, title('DCT-I'), set(gca,'FontSize',30)
            %       subplot(4,3,2), plot(matlab_dct1), axis tight, title('Maltab''s DCT-I'), set(gca,'FontSize',30)
            %       subplot(4,3,3), plot(audio_dct1-matlab_dct1), axis tight, title('Error'), set(gca,'FontSize',30)
            %       subplot(4,3,4), plot(audio_dct2), axis tight, title('DCT-II'), set(gca,'FontSize',30)
            %       subplot(4,3,5), plot(matlab_dct2),axis tight, title('Maltab''s DCT-II'), set(gca,'FontSize',30)
            %       subplot(4,3,6), plot(audio_dct2-matlab_dct2), axis tight, title('Error'), set(gca,'FontSize',30)
            %       subplot(4,3,7), plot(audio_dct3), axis tight, title('DCT-III'), set(gca,'FontSize',30)
            %       subplot(4,3,8), plot(matlab_dct3), axis tight, title('Maltab''s DCT-III'), set(gca,'FontSize',30)
            %       subplot(4,3,9), plot(audio_dct3-matlab_dct3), axis tight, title('Error'), set(gca,'FontSize',30)
            %       subplot(4,3,10), plot(audio_dct4), axis tight, title('DCT-IV'), set(gca,'FontSize',30)
            %       subplot(4,3,11), plot(matlab_dct4), axis tight, title('Maltab''s DCT-IV'), set(gca,'FontSize',30)
            %       subplot(4,3,12), plot(audio_dct4-matlab_dct4), axis tight, title('Error'), set(gca,'FontSize',30)
            %   
            %   See also dct, fft
            
            switch dct_type
                case 1
                    
                    % Number of samples per frame
                    window_length = size(audio_signal,1);
                    
                    % Pre-processing to make the DCT-I matrix orthogonal
                    audio_signal([1,window_length],:) = audio_signal([1,window_length],:)*sqrt(2);

                    % Compute the DCT-I using the FFT
                    audio_dct = [audio_signal;audio_signal(window_length-1:-1:2,:)];
                    audio_dct = fft(audio_dct);
                    audio_dct = real(audio_dct(1:window_length,:))/2;
                    
                    % Post-processing to make the DCT-I matrix orthogonal
                    audio_dct([1,window_length],:) = audio_dct([1,window_length],:)/sqrt(2);
                    audio_dct = audio_dct*sqrt(2/(window_length-1));
                    
                case 2
                    
                    % Number of samples and frames
                    [window_length,number_frames] = size(audio_signal);
                    
                    % Compute the DCT-II using the FFT
                    audio_dct = zeros(4*window_length,number_frames);
                    audio_dct(2:2:2*window_length,:) = audio_signal;
                    audio_dct(2*window_length+2:2:4*window_length,:) = audio_signal(window_length:-1:1,:);
                    audio_dct = fft(audio_dct);
                    audio_dct = real(audio_dct(1:window_length,:))/2;

                    % Post-processing to make the DCT-II matrix orthogonal
                    audio_dct(1,:) = audio_dct(1,:)/sqrt(2);
                    audio_dct = audio_dct*sqrt(2/window_length);
                    
                case 3
                    
                    % Number of samples and frames
                    [window_length,number_frames] = size(audio_signal);
                    
                    % Pre-processing to make the DCT-III matrix orthogonal
                    audio_signal(1,:) = audio_signal(1,:)*sqrt(2);
                    
                    % Compute the DCT-III using the FFT
                    audio_dct = zeros(4*window_length,number_frames);
                    audio_dct(1:window_length,:) = audio_signal;
                    audio_dct(window_length+2:2*window_length+1,:) = -audio_signal(window_length:-1:1,:);
                    audio_dct(2*window_length+2:3*window_length,:) = -audio_signal(2:window_length,:);
                    audio_dct(3*window_length+2:4*window_length,:) = audio_signal(window_length:-1:2,:);
                    audio_dct = fft(audio_dct);
                    audio_dct = real(audio_dct(2:2:2*window_length,:))/4;
                    
                    % Post-processing to make the DCT-III matrix orthogonal
                    audio_dct = audio_dct*sqrt(2/window_length);
                    
                case 4
                    
                    % Number of samples and frames
                    [window_length,number_frames] = size(audio_signal);
                    
                    % Compute the DCT-IV using the FFT
                    audio_dct = zeros(8*window_length,number_frames);
                    audio_dct(2:2:2*window_length,:) = audio_signal;
                    audio_dct(2*window_length+2:2:4*window_length,:) = -audio_signal(window_length:-1:1,:);
                    audio_dct(4*window_length+2:2:6*window_length,:) = -audio_signal;
                    audio_dct(6*window_length+2:2:8*window_length,:) = audio_signal(window_length:-1:1,:);
                    audio_dct = fft(audio_dct);
                    audio_dct = real(audio_dct(2:2:2*window_length,:))/4;
                    
                    % Post-processing to make the DCT-IV matrix orthogonal
                    audio_dct = audio_dct*sqrt(2/window_length);
                    
            end
            
        end
        
        function audio_dst = dst(audio_signal,dst_type)
            % dst Discrete sine transform (DST) using the fast Fourier transform (FFT)
            %   audio_dst = zaf.dst(audio_signal,dst_type);
            %   
            %   Arguments:
            %       audio_signal: audio signal [number_samples,number_frames]
            %       dst_type: DST type (1, 2, 3, or 4)
            %       audio_dst: audio DST [number_frequencies,number_frames]
            %   
            %   Example: Compute the 4 different DSTs and compare them to their respective inverses
            %       % Audio signal averaged over its channels and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            %       
            %       % Audio signal for a given window length, and one frame
            %       window_length = 1024;
            %       audio_signal = audio_signal(1:window_length);
            %       
            %       % DST-I, II, III, and IV
            %       audio_dst1 = zaf.dst(audio_signal,1);
            %       audio_dst2 = zaf.dst(audio_signal,2);
            %       audio_dst3 = zaf.dst(audio_signal,3);
            %       audio_dst4 = zaf.dst(audio_signal,4);
            %       
            %       % Respective inverses, i.e., DST-I, III, II, and IV
            %       audio_idst1 = zaf.dst(audio_dst1,1);
            %       audio_idst2 = zaf.dst(audio_dst2,3);
            %       audio_idst3 = zaf.dst(audio_dst3,2);
            %       audio_idst4 = zaf.dst(audio_dst4,4);
            %       
            %       % DST-I, II, III, and IV, respective inverses, and errors displayed
            %       figure
            %       subplot(4,3,1), plot(audio_dst1), axis tight, title('DST-I'), set(gca,'FontSize',30)
            %       subplot(4,3,2), plot(audio_idst1), axis tight, title('Inverse DST-I = DST-I'), set(gca,'FontSize',30)
            %       subplot(4,3,3), plot(audio_signal-audio_idst1), axis tight, title('Error'), set(gca,'FontSize',30)
            %       subplot(4,3,4), plot(audio_dst2), axis tight, title('DST-II'), set(gca,'FontSize',30)
            %       subplot(4,3,5), plot(audio_idst2), axis tight, title('Inverse DST-II = DST-III'), set(gca,'FontSize',30)
            %       subplot(4,3,6), plot(audio_signal-audio_idst2), axis tight, title('Error'), set(gca,'FontSize',30)
            %       subplot(4,3,7), plot(audio_dst3), axis tight, title('DST-III'), set(gca,'FontSize',30)
            %       subplot(4,3,8), plot(audio_idst3), axis tight, title('Inverse DST-III = DST-II'), set(gca,'FontSize',30)
            %       subplot(4,3,9), plot(audio_signal-audio_idst3), axis tight, title('Error'), set(gca,'FontSize',30)
            %       subplot(4,3,10), plot(audio_dst4), axis tight, title('DST-IV'), set(gca,'FontSize',30)
            %       subplot(4,3,11), plot(audio_idst4), axis tight, title('Inverse DST-IV = DST-IV'), set(gca,'FontSize',30)
            %       subplot(4,3,12), plot(audio_signal-audio_idst4), axis tight, title('Error'), set(gca,'FontSize',30)
            %
            %   See also dct, fft
            
            switch dst_type
                case 1
                    
                    % Number of samples per frame
                    [window_length,number_frames] = size(audio_signal);
                    
                    % Compute the DST-I using the FFT
                    audio_dst = [zeros(1,number_frames);audio_signal; ...
                        zeros(1,number_frames);-audio_signal(window_length:-1:1,:)];
                    audio_dst = fft(audio_dst);
                    audio_dst = -imag(audio_dst(2:window_length+1,:))/2;
                    
                    % Post-processing to make the DST-I matrix orthogonal
                    audio_dst = audio_dst*sqrt(2/(window_length+1));
                    
                case 2
                    
                    % Number of samples per frame
                    [window_length,number_frames] = size(audio_signal);
                    
                    % Compute the DST-II using the FFT
                    audio_dst = zeros(4*window_length,number_frames);
                    audio_dst(2:2:2*window_length,:) = audio_signal;
                    audio_dst(2*window_length+2:2:4*window_length,:) = -audio_signal(window_length:-1:1,:);
                    audio_dst = fft(audio_dst);
                    audio_dst = -imag(audio_dst(2:window_length+1,:))/2;
                    
                    % Post-processing to make the DST-II matrix orthogonal
                    audio_dst(window_length,:) = audio_dst(window_length,:)/sqrt(2);
                    audio_dst = audio_dst*sqrt(2/window_length);
                    
                case 3
                    
                    % Number of samples per frame
                    [window_length,number_frames] = size(audio_signal);
                    
                    % Pre-processing to make the DST-III matrix orthogonal
                    audio_signal(window_length,:) = audio_signal(window_length,:)*sqrt(2);
                    
                    % Compute the DST-III using the FFT
                    audio_dst = zeros(4*window_length,number_frames);
                    audio_dst(2:window_length+1,:) = audio_signal;
                    audio_dst(window_length+2:2*window_length,:) = audio_signal(window_length-1:-1:1,:);
                    audio_dst(2*window_length+2:3*window_length+1,:) = -audio_signal;
                    audio_dst(3*window_length+2:4*window_length,:) = -audio_signal(window_length-1:-1:1,:);
                    audio_dst = fft(audio_dst);
                    audio_dst = -imag(audio_dst(2:2:2*window_length,:))/4;
                    
                    % Post-processing to make the DST-III matrix orthogonal
                    audio_dst = audio_dst*sqrt(2/window_length);
                    
                case 4
                    
                    % Number of samples per frame
                    [window_length,number_frames] = size(audio_signal);
                    
                    % Compute the DST-IV using the FFT
                    audio_dst = zeros(8*window_length,number_frames);
                    audio_dst(2:2:2*window_length,:) = audio_signal;
                    audio_dst(2*window_length+2:2:4*window_length,:) = audio_signal(window_length:-1:1,:);
                    audio_dst(4*window_length+2:2:6*window_length,:) = -audio_signal;
                    audio_dst(6*window_length+2:2:8*window_length,:) = -audio_signal(window_length:-1:1,:);
                    audio_dst = fft(audio_dst);
                    audio_dst = -imag(audio_dst(2:2:2*window_length,:))/4;
                    
                    % Post-processing to make the DST-IV matrix orthogonal
                    audio_dst = audio_dst*sqrt(2/window_length);
                    
            end
            
        end
        
        function audio_mdct = mdct(audio_signal,window_function)
            % mdct Modified discrete cosine transform (MDCT) using the DCT-IV
            %   audio_mdct = zaf.mdct(audio_signal,window_function);
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
            %       window_length = 512;
            %       alpha_value = 5;
            %       window_function = kaiser(window_length/2+1,alpha_value*pi);
            %       window_function2 = cumsum(window_function(1:window_length/2));
            %       window_function = sqrt([window_function2;window_function2(window_length/2:-1:1)]./sum(window_function));
            %       
            %       % MDCT
            %       audio_mdct = zaf.mdct(audio_signal,window_function);
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
            %       set(gca,'FontSize',30)
            %
            %   See also dct, zaf.imdct
            
            % Number of samples and window length
            number_samples = length(audio_signal);
            window_length = length(window_function);
            
            % Number of time frames
            number_times = ceil(2*number_samples/window_length)+1;
            
            % Pre and post zero-padding of the signal
            audio_signal = [zeros(window_length/2,1);audio_signal;zeros((number_times+1)*window_length/2-number_samples,1)];
            
            % Initialize the MDCT
            audio_mdct = zeros(window_length/2,number_times);
            
            % Loop over the time frames
            for time_index = 1:number_times
                
                % Window the signal
                sample_index = (time_index-1)*window_length/2;
                audio_segment = audio_signal(1+sample_index:window_length+sample_index).*window_function;
                
                % Time-domain aliasing cancellation (TDAC) principle
                audio_mdct(:,time_index) = [-audio_segment(3*window_length/4:-1:window_length/2+1)-audio_segment(3*window_length/4+1:window_length); ...
                    audio_segment(1:window_length/4)-audio_segment(window_length/2:-1:window_length/4+1)];
            end
            
            % DCT-IV
            audio_mdct = sqrt(window_length/4)*dct(audio_mdct,'Type',4);
            
        end
        
        function audio_signal = imdct(audio_mdct,window_function)
            % imdct Inverse modified discrete cosine transform (MDCT) using the DCT-IV
            %   audio_signal = zaf.imdct(audio_mdct,window_function);
            %   
            %   Arguments:
            %       audio_mdct: audio MDCT [number_frequencies,number_times]
            %       window_function: window function [window_length,1]
            %       audio_signal: audio signal [number_samples,1]
            %   
            %   Example: Verify that the MDCT is perfectly invertible
            %       % Audio file averaged over the channels and sample rate in Hz
            %       [audio_signal,sample_rate] = audioread('audio_file.wav');
            %       audio_signal = mean(audio_signal,2);
            %       
            %       % MDCT with a slope function as used in the Vorbis audio coding format
            %       window_length = 2048;
            %       window_function = sin((pi/2)*sin((pi/window_length)*(0.5:(window_length-0.5))).^2)';
            %       audio_mdct = zaf.mdct(audio_signal,window_function);
            %       
            %       % Inverse MDCT and error signal
            %       audio_signal2 = zaf.imdct(audio_mdct,window_function);
            %       audio_signal2 = audio_signal2(1:length(audio_signal));
            %       error_signal = audio_signal-audio_signal2;
            %       
            %       % Original, resynthesized, and error signals displayed in s
            %       figure
            %       subplot(3,1,1), plot(audio_signal), axis tight, title('Original Signal')
            %       xticks(sample_rate:sample_rate:length(audio_signal))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)'), set(gca,'FontSize',30)
            %       subplot(3,1,2), plot(audio_signal2), axis tight, title('Resynthesized Signal')
            %       xticks(sample_rate:sample_rate:length(audio_signal))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)'), set(gca,'FontSize',30)
            %       subplot(3,1,3), plot(error_signal), axis tight, title('Error Signal')
            %       xticks(sample_rate:sample_rate:length(audio_signal))
            %       xticklabels(1:floor(length(audio_signal)/sample_rate))
            %       xlabel('Time (s)'), set(gca,'FontSize',30)
            %
            %   See also dct, zaf.mdct
            
            % Number of frequency channels and time frames
            [number_frequencies,number_times] = size(audio_mdct);
            
            % Number of samples for the signal
            number_samples = number_frequencies*(number_times+1);
            
            % Initialize the audio signal
            audio_signal = zeros(number_samples,1);
            
            % DCT-IV (which is its own inverse)
            audio_mdct = dct(audio_mdct,'Type',4);
            
            % Time-domain aliasing cancellation (TDAC) principle
            audio_mdct = [audio_mdct(number_frequencies/2+1:number_frequencies,:); ...
                -audio_mdct(number_frequencies:-1:number_frequencies/2+1,:); ...
                -audio_mdct(number_frequencies/2:-1:1,:); ...
                -audio_mdct(1:number_frequencies/2,:)];
            
            % Apply the window to the frames
            audio_mdct = window_function.*audio_mdct;

            % Loop over the time frames
            for time_index = 1:number_times
                
                % Recover the signal thanks to the TDAC principle
                sample_index = (time_index-1)*number_frequencies+1;
                audio_signal(sample_index:sample_index+2*number_frequencies-1,1) ...
                    = audio_signal(sample_index:sample_index+2*number_frequencies-1,1)+audio_mdct(:,time_index);
                
            end
            
            % Remove the pre and post zero-padding
            audio_signal = audio_signal(number_frequencies+1:end-number_frequencies);
            
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
            
        end
        
    end
end
