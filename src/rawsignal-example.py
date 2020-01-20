import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plot
from scipy.fftpack import dct


sample_rate, signal = scipy.io.wavfile.read('./1234.wav')  # File assumed to be in the same directory
#signal = signal[0:int(3.5 * sample_rate)]  # Keep the only first 3.5 seconds
Time = numpy.linspace(0, len(signal) / sample_rate, len(signal)) #Time axis 

#--Pre-Emphasis--
pre_emphasis = 0.97 #typical values for the filter coefficient (α) are 0.95 or 0.97,
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

#--Framing-- 
#Typical frame sizes in speech processing range from 20 ms to 40 ms with 50% (+/-10%) overlap between consecutive frames. 
frame_size = 0.025 #Popular setting 25 ms for the frame size
frame_stride = 0.01 #10 ms stride (15 ms overlap)

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T #construct array
frames = pad_signal[indices.astype(numpy.int32, copy=False)] #just for plotting
ham_frames = pad_signal[indices.astype(numpy.int32, copy=False)]

#--Apply Hamming Window--
ham_frames *= numpy.hamming(frame_length)
# frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

#Fourier-Transform and Power Spectrum
#We can now do an N-point FFT on each frame to calculate the frequency spectrum. aka Short-Time Fourier-Transform (STFT)
#NFFT = 256 #N is typically 256 or 512
NFFT = frame_length
mag_frames = numpy.absolute(numpy.fft.rfft(ham_frames, NFFT))  # Magnitude of the FFT
freq_scale = numpy.fft.rfftfreq(frame_length, 1./sample_rate)  # Frequency scale
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

#--Filter Banks--
nfilt = 40 #number of filter banks

low_freq_mel = 0
high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = numpy.dot(pow_frames, fbank.T)
filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * numpy.log10(filter_banks)  # dB



# -- Mel-frequency Cepstral Coefficients (MFCCs) --
num_ceps = 12 #Typically, for Automatic Speech Recognition (ASR), the resulting cepstral coefficients 2-13 are retained and the rest are discarded

mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13 (num_ceps+1)

#One may apply sinusoidal liftering1 to the MFCCs to de-emphasize higher MFCCs 
#(nframes, ncoeff) = mfcc.shape
#n = numpy.arange(ncoeff)
#lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
#mfcc *= lift  #*

#-- Plots 

plot.figure(1,[12, 6]) #plot 1st frame 

plot.subplot(321)
plot.title("(1) Frame #1")
plot.xlabel("Time ")
plot.tight_layout(2)
plot.plot(frames[0])

plot.subplot(322)
plot.title("(2) Hamming applied")
plot.plot(ham_frames[0])

plot.subplot(323)
plot.title("(3) FFT spectrum")
plot.xlabel("Freq (Hz)")
plot.bar(freq_scale,mag_frames[0],freq_scale[1])

plot.subplot(324)
plot.title("(4) Power spectrum")
plot.xlabel("Freq (Hz)")
plot.bar(freq_scale,pow_frames[0],freq_scale[1])

plot.subplot(325)
plot.title("(5) Mel Filters applied ")
plot.xlabel("Freq (Hz)")
plot.bar(numpy.arange(freq_scale[-1]/nfilt,freq_scale[-1]+(freq_scale[-1]/nfilt),freq_scale[-1]/nfilt), filter_banks[0], (freq_scale[-1]/nfilt))

plot.subplot(326)
plot.title("(6) MFCC's ")
plot.xlabel(" Coefficient # ")
plot.bar(numpy.arange(1,num_ceps+1,1), mfcc[0], 1)


plot.figure(2, [12, 6])#plot full signal 

plot.subplot(321)
plot.title("(1) Raw .wav signal")
plot.tight_layout(2)
plot.plot(Time,signal)

plot.subplot(322)
plot.title("(2) Pre-emphasised signal")
plot.plot(emphasized_signal)

plot.subplot(323)
plot.title("(3) FFT")
plot.imshow(mag_frames.T,plot.cm.jet, aspect='auto')
plot.yticks([0,50,100],[int(freq_scale[0]),int(freq_scale[50]),int(freq_scale[100])])
plot.gca().invert_yaxis()
plot.ylabel("Freq (Hz)")

plot.subplot(324)
plot.title("(4) Power Spectrum (periodogram) ")
plot.imshow(pow_frames.T,plot.cm.jet, aspect='auto')
plot.yticks([0,50,100],[int(freq_scale[0]),int(freq_scale[50]),int(freq_scale[100])])
plot.gca().invert_yaxis()
plot.ylabel("Freq (Hz)")

plot.subplot(325)
plot.title("(5) Mel Filter banks applied (spectogram)")
plot.imshow(filter_banks.T,plot.cm.jet, aspect='auto')
plot.yticks([0,int(nfilt/2),int(nfilt)],[int(freq_scale[0]),int(freq_scale[50]),int(freq_scale[100])])
plot.gca().invert_yaxis()
plot.ylabel("Freq (Hz)")
plot.xlabel("Time")

plot.subplot(326)
plot.title("(6) MFCC's")
plot.imshow(mfcc.T,plot.cm.jet, aspect='auto')
#plot.yticks([0,int(nfilt/2),int(nfilt)],[int(freq_scale[0]),int(freq_scale[50]),int(freq_scale[100])])
plot.gca().invert_yaxis()
plot.ylabel("MFCC Coeffecients")
plot.xlabel("Time")

plot.show() 

#-- Print some details 
print("Sound file duration " + str(len(signal) / sample_rate) + "(secs), " + str(len(signal)) + " samples")
print("Sampling rate " + str(sample_rate) + "Hz ("+ str(sample_rate) +" samples per sec)" )
print("-Framing-")
print("Frame length: " +str(frame_length) + " samples, " + str(round(frame_size*1000)) + 'ms')
print("Frame overlapp: " +str(round((frame_size-frame_stride)*1000)) + "ms")
print("No. of frames: " +str(num_frames))
print("-Filter Banks-")
print("No. of filter banks: " + str(nfilt) + " (spectogram array size)")
print("-MCC's-")
print("No. of coeffecients: " + str(num_ceps))



