import numpy as np 
import matplotlib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

import tensorflow.keras.backend as k

from scipy import signal
from scipy.io import wavfile

import matplotlib.pyplot as plt
import librosa.display
import librosa.feature

def display_sigal(y, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)


def display_spectrogram(y, sr):
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

def display_mel_spectrogram(sr=16000):
    # sr = 16000

    files = './data/mics/mic5/beamformed_50069-50092.wav'
    # files1 = [file for file in os.listdir(TRAIN_PATH) if file.endswith('.wav')]
    # print(files1[0])
    y, _ = librosa.load(files, sr=sr)
    X = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=X)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    # print(librosa.power_to_db(S, ref=np.max))
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()
    

def mel_spectrogram(y, sr, n_fft=512, hop_length=1024, win_length=None, window='hann'):
    return librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hann')

def STFT_spectrum(data):
    D = np.abs(librosa.stft(y))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='linear', x_axis='time')
    plit.title('Dual Tone')
    plt.show()

# def stft(data, fft_size=512, step_size=160, win='hann'):
#     F = np.fft.rfft(librosa.stft(data, n_fft=fft_size, 
def stft(data, fft_size=512, step_size=160, win='hann', padding=True):
    # return librosa.stft(data, hop_length=hop_len, win_length=win_len, window=win)
    # short time fourier transform
    # print('---------------------------------\n')
    # print(data)
    if padding == True:
        # for 16K sample rate data, 48192-192 = 48000
        pad = np.zeros(192,)
        data = np.concatenate((data,pad),axis=0)
    # padding hanning window 512-400 = 112
    window = np.concatenate((np.zeros((56,)),np.hanning(fft_size-112),np.zeros((56,))),axis=0)
    win_num = (len(data) - fft_size) // step_size
    # print(win_num)
    # print('=======================')
    out = np.ndarray((win_num, fft_size), dtype=data.dtype)
    for i in range(win_num):
        left = int(i * step_size)
        right = int(left + fft_size)
        out[i] = data[left: right] * window
    F = np.fft.rfft(out, axis=1)
    return F

def istft(data, hop_len, win_len, n_fft=512, win='hann'):
    return librosa.istft(data, hop_length=hop_len, win_length=win_len, window=win)
    
def gen_crm(Y, S):
    M = np.zeros(Y.shape) # 0: Speech, 1: noise
    M_r_num = np.multiply(Y[:,:,0],S[:,:,0]) + np.multiply(Y[:,:,1],S[:,:,1])
    M_r_den = np.square(Y[:,:,0]) + np.square(Y[:,:,1])
    M_r = np.divide(M_r_num, M_r_den)

    M_i_num = np.multiply(Y[:,:,0],S[:,:,0]) - np.multiply(Y[:,:,1],S[:,:,1])
    M_i_den = np.square(Y[:,:,0]) + np.square(Y[:,:,1])
    M_i = M_i = np.divide(M_i_num, M_i_den)

    # return M_r + jM_i
    return M


# def extending_complex_domain():
#     S_r = M_r + jM_i
#     S_i = 
    
# Compress the cIRM, hyperbolic tangent
# C: steepness M: mask     
def crm_compress(M, K=10, C=0.1):
    num = 1 - np.exp(-C * M)
    num[num == np.inf] = 1
    num[num == -np.inf] = -1

    den = 1 + np.exp(-C * M)
    den[den == np.inf] = 1
    den[den == -np.inf] = -1
    crm = K * np.divide(num, den)

    return crm

# Complex Ratio Mask
def get_crm(c_data, m_data, K=10, C=0.1):
    M = gen_crm(c_data, m_data)  # c: clean m: inference
    crm = crm_compress(M, K, C) 
    return crm

def get_stft(data):
    return expand_real_imag(stft(data))

def get_istft(data):
    return expand_real_imag(istft(data)) 

def expand_real_imag(data, d_type='n'):

    if d_type == 'n':
        d_data = np.zeros((data.shape[0], data.shape[1], 2))
        d_data[:, :, 0] = np.real(data)
        d_data[:, :, 1] = np.imag(data) 
        
    elif d_type == 's':
        d_data = np.zeros((data.shape[0], data.shape[1] * 2))
        d_data[:, : : 2] = np.real(data)
        d_data[:, 1::2] = np.imag(data)

    return d_data

def SNR(true_file,pred_file):
    T_true,_ = librosa.load(true_file,sr=16000)
    F_true = fast_stft(T_true)
    T_pred, _ = librosa.load(pred_file,sr=16000)
    F_pred = fast_stft(T_pred)
    F_inter = F_true - F_pred
    P_true = np.sum(np.square(F_true[:,:,0])+np.square(F_true[:,:,1]))
    P_inter = np.sum(np.square(F_inter[:,:,0])+np.square(F_inter[:,:,1]))
    return 10*np.log10(np.divide(P_true,P_inter))

if __name__ == '__main__':
    PATH = './data/'
    # display_mel_spectrogram()
    # print(SNR())
    display_mel_spectrogram()
