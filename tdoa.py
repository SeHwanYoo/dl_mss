import os
import librosa
import numpy as np
import audioop
import array
import matlab.engine

'''
x: The singal of the reference microhpone
Fixed second microhpone: 1 
'''
# def gcc_phat(data, channels_num):
#     sample_num = len(data[0])
#     points_num = np.pow(2, np.ceil(np.log(samples_num) / np.log(2)))
#     half = points_num / 2
    
#     win_data = np.zeros(points_num * channels_num)
#     for i in range(channels_num):
#         win[i] = np.hamming(data[i], sample_num)

#     fft_real = np.fft.fft(points_num)
#     fft_imag = np.fft.fft(points_num) 
#     for i in range(channels_num):
#         fft_real[i] = np.fft.fft(data[i], n=points_num)
#         fft_imag[i] = np.fft.fft(data[i], n=points_num)

#     corarrayreal = np.zeros(points_num * channels_n
#     corarrayimag = np.zeros(points_num * channels_num)

#     ref = 0 # reference mic is 0 location
    
#     for c in range(1, channels_num):

#         for p in range(points_num):

#             m = ref * points_num + p 
#             n = c + points_num + p

#             corarrayreal[n] = fft_real[n] * fft_real[m] + fft_imag[n] * fft_imag[m]
#             corarrayimag[n] = fft_real[n] * fft_real[m] - fft_imag[n] * fft_imag[m]

#             length = np.sqrt(corarrayreal[n] * corarrayreal[n] + corarrayimag[n] * corarrayimag[n])

#             corarrayreal[n] /= length
#             corarrayimag[n] /= length

#         inverse fft
#         fft(corarrayreal

#         re# range
#         for h in range(half):
        

def get_samples(fixed_sr=16000):
    files = [file for file in os.listdir('./data/mics/mic1') if file.endswith('.wav')]
    return files

def get_mic_files(sample, channel_num, fixed_sr=16000):
    mic_files = []
    for m in range(channel_num):
        data, _ = librosa.load('./data/mics/mic%d/%s'%(m+1, sample), sr=fixed_sr)
        mic_files.append(data)

    return mic_files

# def get_audio_signals(mic_files, channel_num, max_tau):
    # for m in mic_files:
    #     buff = np.fromstring(m, dtype='int16')
    #     mono = buff[0::channel_num].tostring()
    #     tau, _ = gd_phat(buff[0::channel_num] * win, buff[1::channel_num] * win, max_tau)
    #     theta = math.asin(tau / max_tau) *180 / math.pi
    # return buff, mono

# def get_fft_size(samples_len):
#     fft_size = 2

#     while fft_size < 2 * samples_len:
#         fft_size = fft_size * 2;

#     return np.floor(fft_size / 2)

# def gcc_phat(data, refData, fixed_sr=16000):
#     N = 2 * len(data) - 1
#     N = np.ceil(np.log2(np.abs(N)))
#     N = int(N)
#     df = fixed_sr / N
#     sample_index = np.r_[-(N/2):(N/2)-1]
#     f = sample_index * df

#     refSig = np.fft.fft(refData, n=N)
#     ref = 0 
#     sig = np.fft.fft(data, n=N)
#     r = sig * np.conj(refSig)
#     c = r / np.abs(r)
#     ic = np.fft.fftshift(np.fft.irfft(c)) # cross-corelation
    # ind = np.max(np.abs(ic))
    # cc = np.concatenate((cc[

    # SIG = np.fft.rfft(sig, n=n)
    # REFSIG = np.fft.rfft(refsig, n=n)
    # R = SIG * np.conj(REFSIG)

    # cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    # max_shift = int(interp * n / 2)
    # if max_tau:
    #     max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    # cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # # find max cross correlation index
    # shift = np.argmax(np.abs(cc)) - max_shift

    # tau = shift / float(interp * fs)
    
    # return tau, cc

def gcc_phat(sig, refSig):
    eng = matlab.engine.start_matlab()
    tau = eng.tdoa_func(sig, refSig)
    # tau = eng.gccphat(sig, refSig)
    return tau

def gcc_phat2(samples):
    eng = matlab.engine.start_matlab()
    tau = eng.tdoa_func(samples)
    return tau 

channel_num = 5 
samples = get_samples()

taus= gcc_phat2(samples[0])
