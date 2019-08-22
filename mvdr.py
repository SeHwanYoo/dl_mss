import numpy as np
import os
import librosa

from scipy import signal
# from scipy.fftpack import ifft  

import numpy.matlib

DATA_PATH = './data/'

TXT_PATH = DATA_PATH + 'txt/'
MIC_PATH = DATA_PATH + 'mics/'

def steering_vector_estimation(sound_speed, mic_angles, channel_num, distance, direct=0, fixed_sr=16000, fft_len=512):
    freq = np.linspace(0, fixed_sr, fft_len) 
    steering = np.ones((len(freq), channel_num), dtype=np.complex64)

    for f, fr in enumerate(freq):        
        for m, angle in enumerate(mic_angles):
            steering[f, m] = np.complex(np.exp(-1j * ((2 * np.pi * fr) / sound_speed) * (distance / 2) * np.cos(np.deg2rad(direct) - np.deg2rad(angle))))

    steering = np.conj(steering).T 
    nomalized_steering = normalize(steering)

    return nomalized_steering[0 : np.int(fft_len / 2) + 1, :]

def normalize(steering, fft_len=512, fft_shift=512):

    for n in range(fft_len):
        w = np.matmul(np.conj(steering[:, n]).T, steering[:, n])
        steering[:, n] = steering[:, n] / w

    return steering

def spatial_correlation_model(data, mic_angles, channel_num, fft_len=512, fixed_sr=16000):
    # return 0
    # number_of_mic = len(mic_angles)
    forward_frames = 10
    backward_frames = 10 
                                        
    freq = np.linspace(0, fixed_sr, fft_len)
    freq = freq[0:np.int(fft_len / 2) + 1]
    start_idx = 0
    end_idx = start_idx + fft_len
    # data_len, number_of_channels = np.shape(data)
    data_len = len(data)
    R_mean = np.zeros((channel_num, channel_num, len(freq)), dtype=np.complex64)
    # frames_num = 0
    frames_num = 0 

    # forward
    for f1 in range(forward_frames):
        data_cut = data[start_idx:end_idx, :]
        complex_signal = np.fft.fft(data_cut, n=fft_len, axis=0)
                                        
        for f2 in range(len(freq)):
                R_mean[:, :, f2] = R_mean[:, :, f2] + np.multiply.outer(complex_signal[f2, :], np.conj(complex_signal[f2, :]).T)
                
        frames_num += 1
        start_idx = start_idx + fft_shift
        end_idx = end_idx + fft_shift
                                        
        if (data_len <= start_idx) or (data_len <= end_idx):
            frames_num -= 1
            break

    # backward 
    end_idx = data_len 
    start_idx = end_idx - fft_len
                                        
    for f1 in range(backward_frames):
        data_cut = data[start_idx:end_idx, :]
        complex_signal = np.fft.fft(data_cut, n=fft_len, axis=0)
                                        
        for f2 in range(len(freq)):
            R_mean[:, :, f2] = R_mean [:, :, f2] + np.multiply.outer(complex_signal[f2, :], np.conj(complex_signal[f2, :]).T)

        frames_num += 1
        start_idx -= fft_shift
        end_idx -= fft_shift
                                        
        if  start_idx < 1 or end_idx < 1:
            frames_num -= 1
            break                    

    return R_mean / frames_num  


def mvdr_beamform(steering, spatials, mic_angles, channel_num, fft_len=512, fixed_sr=16000):
    # return 0
    # number_of_mic = len(mic_angles)
    
    freq = np.linspace(0, fixed_sr, fft_len)
    freq = freq[0:np.int(fft_len / 2) + 1]        
    beamform = np.ones((channel_num, len(freq)), dtype=np.complex64)
    
    for f in range(len(freq)):
        spatials_cut = np.reshape(spatials[:, :, f], [channel_num, channel_num])
        inv_spatials = np.linalg.pinv(spatials_cut)
        a = np.matmul(np.conj(steering[:, f]).T, inv_spatials)
        b = np.matmul(a, steering[:, f])
        b = np.reshape(b, [1, 1])
        beamform[:, f] = np.matmul(inv_spatials, steering[:, f]) / b # number_of_mic *1   = number_of_mic *1 vector/scalar
        
    return beamform


# def apply_beamform(self):
#     # return 0
#     number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)        
#     enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
#     for f in range(0, number_of_bins):
#         enhanced_spectrum[:, f] = np.matmul(np.conj(beamformer[:, f]).T, complex_spectrum[:, :, f])
#     return util.spec2wav(enhanced_spectrum, self.sr, fft_len, fft_len, fft_shift)

# def spectrum_3dim(mic_files):

def speech_enhance(beamform, complex_spectrum, fft_len, fft_shift, fixed_sr=16000):

    channel_num, number_of_frames, number_of_bins = np.shape(complex_spectrum)
    e_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)

    for b in range(number_of_bins):
        e_spectrum[:, b] = np.matmul(np.conj(beamform[:, b]).T, complex_spectrum[:, :, b])


    hanning = signal.hann(fft_len + 1)[: - 1]
    sample_data = np.zeros(fft_len, dtype=np.float64)

    freq = np.zeros(fixed_sr * 60 * 5, dtype=np.float32)
    start = 0
    end = start + fft_len

    for f in range(number_of_frames):
        half_spectrum = e_spectrum[f, :]
        sample_data[:np.int(fft_len / 2) + 1] = half_spectrum.T
        sample_data[np.int(fft_len / 2) + 1:] = np.flip(np.conj(half_spectrum[1:np.int(fft_len / 2)]), axis=0)

        sample_data_real = np.real(np.fft.ifft(sample_data, n=fft_len))
        # print(np.shape(sample_data))
        # print(np.shape(hanning))
        # print(np.shape(sample_data_real))

        freq[start:end] += np.real(sample_data_real * hanning.T)

        start += fft_shift
        end += fft_shift


    # wav_freq = np.max(np.abs(freq[:end-fft_shift])) * 0.65 
    # return wav_freq
    return freq[:end-fft_shift]
    


# call the files list, but one file differnt 
def beamform_mic_array(mic_nums, fixed_sr=16000):

    files = [file for file in os.listdir('./data/mics/mic1') if file.endswith('.wav')]
    wav, _ = librosa.load('./data/mics/mic1/%s'%files[0], sr=fixed_sr)
    mic_files = np.zeros((len(wav), mic_nums), dtype='float32')
    mic_files[:, 0] = wav 

    for idx in range(1, mic_nums):
        mic_files[:, idx], _ = librosa.load('./data/mics/mic%d/%s'%(idx+1, files[0]), sr=fixed_sr)

    return mic_files

# def mixed_files(f):
#     f = ('mic')
         
#     f = f.strip().split('-')
#     # return

def spectrum_3dim(data, channel_num, fft_len=256, fft_shift=256, fixed_sr=16000):
    # len_sample, channel_num = np.shape(wav_data)
    sample_size = len(data) 
    dump_wav = data.T
    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    window = signal.hann(fft_len + 1)[: - 1]
    multi_window = np.matlib.repmat(window, channel_num, 1)    
    st = 0
    ed = fft_len
    frame_size = np.int((sample_size - fft_len) / fft_shift)
    spectrums = np.zeros((channel_num, frame_size, np.int(fft_len / 2) + 1), dtype=np.complex64)

    for ii in range(frame_size):       
        spectrums[:, ii, :] = np.fft.fft(dump_wav[:, st:ed], n=fft_len, axis=1)[:, 0:np.int(fft_len / 2) + 1]
        st += fft_shift
        ed += fft_shift
        
    return spectrums, sample_size

def create_dir():

    if not os.path.isdir('./data/output/'):
        os.mkdir('./data/output/')


sr = 16000
fft_len = 256 
fft_shift = 256
output_path = './data/output/mvdr.wav'
# mic_nums = 6
# mic_nums = 5
sound_speed = 343 
mic_angles = [90, 270] 
direct = 0
# mic_diametters = 1.5
distance = 1.5 
channel_num = 5

create_dir() 

mic_files = beamform_mic_array(channel_num) 
complex_spectrum, sample_size = spectrum_3dim(mic_files, channel_num) 
steering = steering_vector_estimation(sound_speed, mic_angles, channel_num, distance, direct)
spatials = spatial_correlation_model(mic_files, mic_angles, channel_num, fft_len=512)
beamform = mvdr_beamform(steering, spatials, mic_angles, channel_num)
e_beamform = speech_enhance(beamform, complex_spectrum, fft_len, fft_shift)

# print(enhance_beamformer)
output_signal = e_beamform / np.max(np.max(e_beamform)) * 0.65
librosa.output.write_wav(output_path, output_signal, sr)
