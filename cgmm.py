from scipy import signal
# from obspy.signal import util
from itertools import permutations
from numpy.linalg import lstsq, inv, det 

import numpy as np
import os
import librosa

def enframe(x,win,inc):

    nx = len(x)
    nwin = len(win)
    
    if (nwin == 1):
       length = win
    else:
       length = nwin

    if (nargin < 3):
       inc = length

    nf = np.fix((nx - length + inc)/inc)
    f = np.zeros(nf,length)
    indf= inc * np.r_[0:(nf-1)].T
    inds = np.r_[1:lenth]
    f[:] = x[indf[:,np.ones((1, length))] + inds[np.ones((nf, 1)), :]]
             
    if (nwin > 1):
        w = win.T
        f = np.dot(f, w[np.ones((nf,1)),:])

def get_sample(fixed_sr=16000):
    mic_files = [file for file in os.listdir('./data/mics/mic1/') if file.endswith('.wav')] 
    # wav, _ = librosa.load('./data/mics/mic1/' + mic_files[0], sr=fixed_sr)
    return mic_files[0] 

def beamform_mic_array(sample_file, mic_nums, fixed_sr=16000):
    # wav, _ = librosa.load('./data/mics/mic1' + mic_files[0], sr=fixed_sr)
    sample, _ = librosa.load('./data/mics/mic1/' + sample_file, sr=fixed_sr)
    mic_files = np.zeros((len(sample), mic_nums), dtype='float32')
    # print(mic_files.shape)
    # mic_files[:, 0] = wav
    # print(sample)
    for idx in range(mic_nums):
        wav, _ = librosa.load('./data/mics/mic%d/'%(idx+1) + sample_file, sr=fixed_sr)
        # print(wav.shape)
        mic_files[:, idx] = wav

    return mic_files

def get_spectrums(mic_files, mic_nums, frame_len, frame_shift, fft_len):

    for f in range(mic_nums):
        frames = enframe(mic_files[:, f], signal.hanning(frame_len), frame_shift)
        frames_size = len(frames)
        frames_padding = np.zeros(frames_size, fft_len)
        frames_padding[:, 1:frame_len] = frames
        spectrums[:, :, m] = np.fft.rfft(frames_padding, fft_len, 2)         

    return spectrums

def gcmm(num_channels, num_frames, num_bins):
    lambda_noise = np.zeros((num_frames, num_bins))
    lambda_noisy = np.zeros((num_frames, num_bins))

    phi_noise = np.zeros((num_frames, num_bins))
    phi_noisy = np.zeros((num_frames, num_bins))

    r_noise = np.zeros((num_frames, num_bins))
    r_noisy = np.zeros((num_frames, num_bins))

    yyh = np.zeros((num_channels, num_channels, num_frames, num_bins))

    for b in range(num_bins):

        for f in range(num_frames):

            y = spc[:, f, b]
            h = y * y.T
            yyh[:, :, f, b] = h
            r_noisy[:, :, b] = r_noisy[:, :, b] + h

        r_noisy[:, :, b] = r_noisy[:, :, b] / num_frames
        r_noise[:, :, b] = np.eye(num_channels, num_channles) 

    return r_noisy

def gcmm_training(num_frames, num_bins, num_iterations):

    d = 1 / np.sqrt((np.pi * 2) ^ 5)

    for i in range(num_iterations):

        for b in range(num_bins):

            r_noisy_onbin = r_noisy[:, :, b]
            r_noise_onbin = r_noise[:, :, b]

            if lstsq(r_noisy_onbin) < theta:
                r_noisy_onbin = r_noisy_onbin + beta * np.eye(num_channels)

            if lstsq(r_noise_onbin) < theta:
                r_noise_onbin = r_noise_obin + beta * np.eye(num_channels)

            r_noisy_inv = inv(r_noisy_onbin)
            r_noise_inv = inv(r_noise_onbin)

            r_noisy_accuracy = np.zeros(num_channels, num_channels)
            r_noise_accuracy = np.zeros(num_channels, num_channels)

            for f in range(num_frames):
                corrt = yyh[:, :, f, b]
                obs = spc[:, f, b]

                # update
                phi_noise[b, f] = matrix.trace(corrt * r_noise_inv) / num_channels
                phi_noisy[b, f] = matrix.trace(corrt * r_noisy_inv) / num_channels

                # update lambda
                noise = obs.T * (r_noise_inv / phi_noise[b, f]) * obs / 2
                d_noise = det(phi_noise[b, f] * r_noise_onbin)

                p_noise[b, f] = exp(-noise) / np.sqrt(d_noise)
                noise = obs.T * (r_noisy_inv / phi_noisy[b, f]) * obs / 2
                d_noisy = det(phi_noisy[b, f] * r_noisy_onbin)
                p_noisy[b, f] = exp(-noise) / np.sqrt(d_noisy)

                lambda_noise[b, f] = obs.T * (r_noisy_inv / phi_noisy[b, f]) * obs / 2
                lambda_noisy[b, f] = det(phi_noisy[b, f] * r_noisy_onbin)

                r_noise_accuracy = r_noise_accuracy + lambda_noise[b, f] / phi_noise[b, f] * correct
                r_noisy_accuracy = r_noisy_accuracy + lambda_noisy[b, f] / phi_noisy[b, f] * correct

            r_noise[:, :, f] = r_noise_accuracy / sum(lambda_noise[:, f])
            r_noisy[:, :, f] = r_noisy_accuracy / sum(lambda_noisy[:, f])

        q = sum(sum(np.dot(lambda_noise, log(d * p_noise) + np.log(d * p_noisy))) / num_frames * num_bins)

    return p, r_noise, r_noisy

def entropy(r_noise, r_noisy):

    for b in range(num_bins):
        eig_noise = eig(r_noise[:, :, b])
        eig_noisy = eig(r_noisy[:, :, b])

        eig_noise = -eig_noise.T / sum(eig_noise) * np.log(eig_noise / sum(eig_noise))
        eig_noisy = -eig_noisy.T / sum(eig_noisy) * np.log(eig_noisy / sum(eig_noisy))

        if eig_noise < eig_noisy:
            r_noise[:, :, f], r_noisy[:, :, f] = r_noisy[:, :, f], r_noise[:, :, f]

def get_rn(num_channels, num_bins):
    rn = np.zeros((num_channels, num_channels, num_bins))

    for b in range(num_bins):

        for f in range(num_frames):
            rn[:, :, b] += lambda_noise[f, b] * yyh[:, :, f, b]

        rn[:, :, b] += sum(lambda_noise[:, b])

# rx = r_xn -rn

def apply_mvdr(num_frames, num_bins):

    for b in range(num_bins): 
        vector, value = eig(rx[:, :, b])

        if lstsq(rn[:, :, b]) < theta:
            rn[:, :, b] = rn[:, :, f] + beta * np.eye(num_chanenls)

        num = rn[:, :, f] << steer_vector
        w = num / (steer_vector.T * num)
        spc[:, f] = w.T * specs[:, :, f]

def reconstruct(spc, fft_len):
    frames_enhance = irfft(spc, fft_len, 2)
    # xs = set(frames_enhance[:, 1:frame_length])
    # signal_enhance = xs.intersection
    # signal_enhance = 
    signal_enhance = overlapadd(frames_enhance[:, 1:frame_length], win, frame_shift)
    # audio
    

mic_nums = 5
mic_angles = [90, 270]
frame_len = 1024 
frame_shift = 256 
fft_len = 1024

sample_file = get_sample() 
mic_files = beamform_mic_array(sample_file, mic_nums)
spectrums = get_spectrums(mic_files, mic_nums, frame_len, frame_shift, fft_len) 

spc = permutation(spectrums[:, :, [1, 3, 4, 5, 6]], [3, 1, 2])
num_channels, num_frames, num_bins = size(spc)

r_xn = cgmm(num_channels, num_frames, num_bins)
