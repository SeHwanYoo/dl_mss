# Deep Learning for Multi-Channel Speech Spearation
# Author: Sehwan Yoo(krstyle03v@gmail.com)
# National: KOREA (No China) 
# University of Surrey (United Kingdom) 
# Description:
# This file carries out making beamformed environment via pyroomacoustics
# And it makes a mat file using STFT
# All of data implementation is going to be here
import sys
import os
import pyroomacoustics as pra
from tqdm import tqdm
import scipy.io as sio
import librosa
import libs
import numpy as np 
from scipy.io import wavfile
import matplotlib.pyplot as plt
import re # regular expressiokn
from datetime import datetime
import itertools
import time
import numpy as np
from numpy.linalg import eig, det
import math

DATA_PATH = './data/'

TRAIN_PATH = DATA_PATH + 'train/'
# NOISE_PATH = DATA_PATH + 'noise/'
# TRAIN1_PATH = DATA_PATH + 'train1/'
# TRAIN2_PATH = DATA_PATH + 'train2/'

MIX_PATH = DATA_PATH + 'mix/'
CRM_PATH = DATA_PATH + 'crm/'
TXT_PATH = DATA_PATH + 'txt/'
MICS_PATH = DATA_PATH + 'mics/'
BEAMFORMED_PATH = DATA_PATH + 'beamformed/'

def create_dir():

    # if not os.path.isdir(MIX_PATH):
    #     os.mkdir(MIX_PATH)

    if not os.path.isdir(CRM_PATH):
        os.mkdir(CRM_PATH)

    if not os.path.isdir(TXT_PATH):
        os.mkdir(TXT_PATH)

    if not os.path.isdir(MICS_PATH + 'mic1/'):
        os.mkdir(MICS_PATH + 'mic1/')

    if not os.path.isdir(MICS_PATH + 'mic2/'):
        os.mkdir(MICS_PATH + 'mic2/')

    if not os.path.isdir(MICS_PATH + 'mic3/'):
        os.mkdir(MICS_PATH + 'mic3/')

    if not os.path.isdir(MICS_PATH + 'mic4/'):
        os.mkdir(MICS_PATH + 'mic4/')

    if not os.path.isdir(MICS_PATH + 'mic5/'):
        os.mkdir(MICS_PATH + 'mic5/')

    if not os.path.isdir(BEAMFORMED_PATH + 'beam1/'):
        os.mkdir(BEAMFORMED_PATH + 'beam1/')
        
    if not os.path.isdir(BEAMFORMED_PATH + 'beam2/'):
        os.mkdir(BEAMFORMED_PATH + 'beam2/')
        
    if not os.path.isdir(BEAMFORMED_PATH + 'beam3/'):
        os.mkdir(BEAMFORMED_PATH + 'beam3/')
        
    if not os.path.isdir(BEAMFORMED_PATH + 'beam4/'):
        os.mkdir(BEAMFORMED_PATH + 'beam4/')
        
    if not os.path.isdir(BEAMFORMED_PATH + 'beam5/'):
        os.mkdir(BEAMFORMED_PATH + 'beam5/')

# Complex Ratio Mask
# def crm_files(f1, f2):
#     # data_list = []
#     crm_list = []
#     f1_fs, f1_signal = wavfile.read(TRAIN1_PATH + f1)
#     f2_fs, f2_signal = wavfile.read(TRAIN1_PATH + f2)

#     for nn in range(num_speakers):
#         crm_list.append(utils.get_crm(f_list[nn], f_mix))

#     mix_name += post_name
#     crm_name += post_name

#     # making txt file
#     with open('./data/txt/dataset.txt', 'a') as f:
#         f.write(mix_name + '.npy')

#         for t in range(len(crm_list)):
#             line = '/' + crm_name + ('') + '.npy'
#             f.write(line)
#         f.write('\n')

#     np.save(('./data/mix/%s.npy' %mix_name), f_mix)

#     for i in range(len(crm_list)):
#         name = crm_name + ('-%05d' %train_path[i])
#         np.save(('./data/crm/%s.npy' %name), crm_list[i]) 
                        

# mixing a speech and background
# def beamformed_das_files():
#     train_files = [file for file in os.listdir(TRAIN_PATH) if file.endswith('.wav')]
#     noise_files = [file for file in os.listdir(NOISE_PATH) if file.endswith('.wav')]

#     for train_file in tqdm(train_files):
#         t_fs, t_signal = wavfile.read(TRAIN_PATH + train_file)

#         i = 0
#         for noise_file in noise_files:
#             n_fs, n_signal = wavfile.read(NOISE_PATH + nois2e_file)

#             filename = 'mixed' +  re.findall('\d+', train_file)[0] + '_' + str(i)
#             i += 1

#             room = pra.ShoeBox([4, 6], fs=t_fs)
#             echo = pra.linear_2D_array(center=[2, 1.5], M=4, phi=0, d=0.1)
#             mics = pra.Beamformer(echo, room.fs)
        
#             room.add_source(np.array([1.5, 4.5]), delay=0., signal=t_signal)
#             room.add_source(np.array([2.5, 4.5]), delay=0., signal=n_signal[:len(t_signal)])

#             room.add_microphone_array(mics)
#             room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])

#             room.compute_rir()
#             room.simulate()

#             wavfile.write(OUTPUT_PATH + filename + '_1.wav', t_fs, room.mic_array.signals[0,:])
#             wavfile.write(OUTPUT_PATH + filename + '_2.wav', t_fs, room.mic_array.signals[1,:])
#             wavfile.write(OUTPUT_PATH + filename + '_3.wav', t_fs, room.mic_array.signals[2,:])
#             wavfile.write(OUTPUT_PATH + filename + '_4.wav', t_fs, room.mic_array.signals[3,:])

def beamformed_doa_plot(comb):
    f1_data = f1['data']
    f2_data = f2['data']

    # azimuth = np.array([math.atan2(1.5, 0.5), math.atan2(1.5, -0.5)])
    azimuth = np.array([90., 270.,]) * np.pi / 180
    distance = 1.5 

    c = 343.    # speed of sound
    fs = 16000  # sampling frequency
    nfft = 256  # FFT size
    freq_range = [300, 400] 
    sr = 16000
    snr_db = 5.    # signal-to-noise ratio
    # sigma2 = 10**(-snr_db / 10) / (4. * np.pi * distance)**2

    # Add sources of 1 second duration
    rng = np.random.RandomState(23)
    duration_samples = int(sr)

    room_dim = np.r_[4., 6.]
    room = pra.ShoeBox(room_dim, fs=sr)

    echo = pra.linear_2D_array(center=(room_dim/2), M=5, phi=0, d=0.5)
    room.add_microphone_array(pra.MicrophoneArray(echo, room.fs))
    # R = pra.linear_2D_array([2, 1.5], 4, 0, 0.04) 


    # source_location = room_dim / 2 + distance * np.r_[np.cos(ang), np.sin(ang)]
    # source_signal = rng.randn(duration_samples)
    # room.add_source(source_location, signal=source_signal)  
    
    # room.add_source(np.array([1.5, 4.5]), delay=0., signal=f1_data) 
    # room.add_source(np.array([2.5, 4.5]), delay=0., signal=f2_data[:len(f1_data)])

    for ang in azimuth:
        source_location = room_dim / 2 + distance * np.r_[np.cos(ang), np.sin(ang)] 
        source_signal = rng.randn(duration_samples)
        room.add_source(source_location, signal=source_signal)

    room.simulate()

    X = np.array([pra.stft(signal, nfft, nfft // 2, transform=np.fft.rfft).T for signal in room.mic_array.signals])

    # DOA_algorithm = 'MUSIC'
    # spatial_resp = dict()
    
    doa = pra.doa.algorithms['MUSIC'](echo, fs, nfft, c=c, num_src=2, max_four=4)

    # this call here perform localization on the frames in X
    doa.locate_sources(X, freq_range=freq_range)

    spatial_resp = doa.grid.values

    # normalize   
    min_val = spatial_resp.min()
    max_val = spatial_resp.max()
    spatial_resp = (spatial_resp - min_val) / (max_val - min_val)

    # plotting param
    base = 1.
    height = 10.
    true_col = [0, 0, 0]

    # loop through algos
    phi_plt = doa.grid.azimuth

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    c_phi_plt = np.r_[phi_plt, phi_plt[0]]
    c_dirty_img = np.r_[spatial_resp, spatial_resp[0]]
    ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=3,
            alpha=0.55, linestyle='-',
            # label="spatial spectrum"
            )
    # plt.title('MUSIC')

    # plot true loc
    # for angle in azimuth:
    #     ax.plot([angle, angle], [base, base + height], linewidth=3, linestyle='--',
    #         color=true_col, alpha=0.6)
    # K = len(azimuth)
    # ax.scatter(azimuth, base + height*np.ones(K), c=np.tile(true_col,
    #            (K, 1)), s=500, alpha=0.75, marker='*',
    #            linewidths=0,
    #            # label='true locations'
    #            )

    plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, framealpha=0.5,
              scatterpoints=1, loc='center right', fontsize=16,
              ncol=1, bbox_to_anchor=(1.6, 0.5),
              handletextpad=.2, columnspacing=1.7, labelspacing=0.1)

    ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
    ax.xaxis.set_label_coords(0.5, -0.11)
    ax.set_yticks(np.linspace(0, 1, 2))
    ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
    ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
    ax.set_ylim([0, 1.05 * (base + height)]);

    plt.show()


# Linear            
def beamformed_das(comb, people_num, sr=16000):
    f1 = comb[0]
    f2 = comb[1] 
# def beamformed_das(f1, f2, people_num, sr=16000):
    f1_data = f1['data']
    f2_data = f2['data']
    signal_len = len(f1['data']) 
    distance = 1.5

    # azimuth = np.array([math.atan2(1.5, 0.5), math.atan2(1.5, -0.5)])
    azimuth = np.array([90., 270.,]) * np.pi / 180
    
    # centre = [2, 1.5]
    room_dim = np.r_[4, 6] 
    room = pra.ShoeBox(room_dim, fs=sr)
    echo = pra.linear_2D_array(center=(room_dim / 2), M=5, phi=0, d=0.5)
    echo = np.concatenate((echo, np.array((room_dim / 2), ndmin=2).T), axis=1)
    mics = pra.Beamformer(echo, room.fs)
    room.add_microphone_array(mics)    
        
    # room.add_source(np.array([1.5, 4.5]), delay=0., signal=f1_data) 
    # room.add_source(np.array([2.5, 4.5]), delay=0., signal=f2_data[:len(f1_data)])
    signals = [f1_data, f2_data]
    for i, ang in enumerate(azimuth):
        source_location = room_dim / 2 + distance * np.r_[np.cos(ang), np.sin(ang)]
        source_signal = signals[i] 
        room.add_source(source_location, signal=source_signal[:signal_len], delay=0)

    mics.rake_delay_and_sum_weights(room.sources[0][:1])

    # room.plot(freq=[300, 400, 500, 1000, 2000, 4000], img_order=0)
    # plt.show()
    # ax.legend(['300', '400', '500', '1000', '2000', '4000'])
    # fig.set_size_inches(20, 8)

    room.compute_rir()
    room.simulate()

    filename = 'beamformeded_%05d-%05d'%(f1['filename'], f2['filename']) + '.wav'
    
    with open(TXT_PATH + 'build_beamformeded.txt', 'a') as f:
        f.write(filename)
        f.write('\n')

    for i in range(5):
        wavfile.write(MICS_PATH + 'mic%d/'%(i+1) + filename, sr, room.mic_array.signals[i,:])

    # return room

def generate_crm_list():

    beam_list = []
    crm_list = []
    fixed_sr = 16000

    for idx in range(5):
        path = MICS_PATH + 'mic%d/'%(idx+1)
        beam_files = [file for file in os.listdir(path) if file.endswith('.wav')]

        for f in beam_files:
            filename = f[13:(f.find('.'))]

            # for ff in filename.split('-'):
            speech = filename.split('-')
            s1_name = int(speech[0])
            s2_name = int(speech[1])
            name = '%05d'%s1_name + '-%05d'%s2_name
            
            beam_y, _ = librosa.load(path + f, sr=fixed_sr)
            s1_y, _ = librosa.load(TRAIN_PATH + 'trim_audio_train%d'%s1_name + '.wav', sr=fixed_sr)
            s2_y, _ = librosa.load(TRAIN_PATH + 'trim_audio_train%d'%s2_name + '.wav', sr=fixed_sr)

            beamed = libs.get_stft(beam_y[:len(s1_y), ])
            s1 = libs.get_stft(s1_y)
            s2 = libs.get_stft(s2_y)

            s1_crm = libs.get_crm(s1, beamed[0:len(s1)])
            s2_crm = libs.get_crm(s2, beamed[0:len(s1)])

            # Only 1 microphone write on database
            if idx == 1:

                with open(TXT_PATH + 'dataset.txt', 'a') as t:
                    t.write('beam' + name + '.npy')
                    t.write('/crm' + name + ("-%05d"%int(s1_name)) + '.npy')
                    t.write('/crm' + name + ("-%05d"%int(s2_name)) + '.npy')
                    t.write('\n')

                    np.save('./data/crm/crm%s-%05d.npy'%(name, int(s1_name)), s1_crm)
                    np.save('./data/crm/crm%s-%05d.npy'%(name, int(s2_name)), s2_crm)

            np.save(('./data/beamformed/beam%d/beam%s.npy'%((idx+1), name)), beamed)

# def generate_mix2_list():
#     num_speakers = 2 
#     train1_files = [file for file in os.listdir(TRAIN1_PATH) if file.endswith('.wav')]
#     train2_files = [file for file in os.listdir(TRAIN2_PATH) if file.endswith('.wav')]

#     F_list = [] # STFT list for each sample
#     beamformed_data = []
#     crm_list = []
#     mix_rate = 1.0 / float(num_speakers)
#     mix_name = 'mix'
#     crm_name = 'crm'
#     post_name = ''

#     fix_sr = 16000

#     for t1 in train1_files:
#         t1_y, _ = librosa.load(TRAIN1_PATH + t1, sr=fix_sr)

#         for t2 in train2_files:
#             t2_y, _ = librosa.load(TRAIN2_PATH + t2, sr=fix_sr)

#             t1_name = int(re.findall('\d+', t1)[0])
#             t2_name = int(re.findall('\d+', t2)[0])

#             print(re.findall('\d+', t1)[0])

#             # re.findall('\d+', f1)[0] + '&' + re.findall('\d+', f2)[0]
#             post_name = '%05d'%t1_name + '-%05d'%t2_name

#             mix = np.zeros(shape=t1_y.shape)
#             mix = (t1_y * mix_rate) + (t2_y * mix_rate)
#             F_mix = libs.get_stft(mix)

#             F_list = [libs.get_stft(t1_y), libs.get_stft(t2_y)]
#             F_mix = libs.get_stft(mix)

#             print(F_list[0].shape)
#             print(F_mix.shape)
            
#             crm_list = [libs.get_crm(F_list[0], F_mix), libs.get_crm(F_list[1], F_mix)]

#             # mix_name += post_name
#             # crm_name += post_name
#             mix_name = 'mix' + post_name
#             crm_name = 'crm' + post_name

#             with open(TXT_PATH + 'dataset.txt', 'a') as f:
#                 f.write(mix_name + '.npy')
#                 f.write('/' + crm_name + ("-%05d"%t1_name) + '.npy')
#                 f.write('/' + crm_name + ("-%05d"%t2_name) + '.npy')
#                 f.write('\n')

#             np.save(('./data/mix/%s.npy'%mix_name), F_mix)
#             np.save(('./data/crm/%s'%crm_name + ('-%05d'%t1_name)), crm_list[0])
#             np.save(('./data/crm/%s'%crm_name + ('-%05d'%t2_name)), crm_list[1])

def split_dataset(train_rate=0.5, test_rate=0.5):
    dataset_path = TXT_PATH + 'dataset.txt'
    # print(dataset_path)
    train_path = TXT_PATH + 'train.txt'
    test_path = TXT_PATH + 'test.txt'

    dataset = open(dataset_path, 'r')
    dataset_list = [] 
    while True:
        f = dataset.readline()
        if not f: break
        dataset_list.append(f)

    dataset.close()
    # print(dataset.readline())
    tot_dataset = len(dataset_list)
    
    tot_train = train_rate * tot_dataset
    tot_test = test_rate * tot_dataset

    train_txt = open(train_path, 'a')
    test_txt = open(test_path, 'a') 

    for i in range(tot_dataset):
        line = dataset_list[i]

        if tot_train > 0:
            tot_train -= 1
            train_txt.write(line)

        elif tot_test > 0:
            tot_test -= 1
            test_txt.write(line)

    train_txt.close()
    test_txt.close()
    
# class beamformeder_circle():
    
#     def __init__(self, people_num, sr=16000):
#         self.people_num = people_num
#         self.sr = sr
        
    
#     def mvdr(self, comb_list):
#         f1 = comb_list[0]['data']
#         f2 = comb_list[1]['data']
#         filename = '-%05d'%comb_list[0]['filename'] + '-%05d'%comb_list[1]['filename']

#         sig1 = np.array(f1, dtype=float)
#         sig1 = pra.normalize(sig1)
#         sig1 = pra.highpass(sig1, self.sr)
#         delay1 = 0

#         sig2 = np.array(f2, dtype=float)
#         sig2 = pra.normalize(sig2)
#         sig2 = pra.highpass(sig2, self.sr)
#         delay2 = 1 

#         center = [2, 1.5]
#         fir_f = 0.100
#         Lg_t = 0.100                # filter size in seconds
#         Lg = np.ceil(Lg_t * self.sr)       # in sample

#         fft_len = 1024
#         mics_num = 6
#         sigma2_n = 5e-7
#         t0 = 1./(self.sr * np.pi * 1e-2)  # starting time function of sinc decay in RIR response
#         delay = 0.050 # Beamformer delay in seconds
#         # absorption = 0.1
#         # max_order_sim = 2
#         room = pra.ShoeBox([4, 6], fs=self.sr, t0=t0, sigma2_awgn=sigma2_n) 

#         room.add_source(np.array([1, 4.5]), delay=delay1, signal=sig1)
#         room.add_source(np.array([2.8, 4.3]), delay=delay2, signal=sig2)

#         R = pra.circular_2D_array(center=center, M=6, phi0=0, radius=37.5e-3)

#         mics = pra.Beamformer(R, room.fs, N=1024, Lg=Lg)
#         room.add_microphone_array(mics)
#         room.compute_rir()
#         room.simulate() 

#         room.compute_rir()
#         room.simulate()
#         # print(np.shape(room.sources[0][0:2]))
#         # print('==================================================\n')
#         # print(np.shape(room.sources[1][0:2]))
#         good_source = room.sources[0][0:2]
#         bad_source = room.sources[1][0:2]

#         mics.rake_mvdr_filters(good_source, bad_source, sigma2_n*np.eye(mics.Lg*mics_num), delay=delay) 
#         # mics.rake_mvdr_filters(room1.sources[0][0:1],
#         #                 room1.sources[1][0:1],
#         #                 sigma2_n*np.eye(mics.Lg*mics.M), delay=delay)

#         # print(len(room.sources))
#         # input_mic1 = pra
#         # output = mics.procss()
#         # print('--------------------------------------%s'%len(mics.signals))
#         # mic=1
#         for s in range(len(mics.signals)):
#             input_mic = pra.normalize(pra.highpass(mics.signals[s], self.sr))
#             wavfile.write(MICS_PATH + 'mic%d/mic%s.wav'%((s+1), filename), self.sr, input_mic)

#         # input_mic = pra.normalize(pra.highpass(mics.signals[0], sr))
#         # wavfile.write(MICS_PATH + 'mic2/input.wav', sr, input_mic)

#         # out_DirectMVDR = pra.normalize(pra.highpass(output, sr))
#         # wavfile.write(DATA_PATH + 'output_DirectMVDR.wav', sr, out_DirectMVDR)


#         # output = mics.process() 

#         # wavfile.write(OUTPUT_PATH + filename + '_1.wav', fs, room.mic_array.signals[0,:])
#         # wavfile.write(OUTPUT_PATH + filename + '_2.wav', fs, room.mic_array.signals[1,:])


#         # print(f1)
#         # print(f2)
#         # return 0

def generate_data_list(people_num):
    # files = [file for file in os.listdir(TRAIN_PATH) if file.endswith('.wav')]
    files = [file for file in os.listdir(DATA_PATH + '_train/') if file.endswith('.wav')]

    file_list = []
    for f in tqdm(files):
        f_data, f_sr = librosa.load(TRAIN_PATH + f)
        data = {'data': f_data,
                'sample_rate': f_sr,
                'filename': int(re.findall('\d+', f)[0])}
        file_list.append(data)

    # for i in tqdm(range(1, len(file_list), 2)):
    # for i in tqdm(range(3000, 4500, 2)):
    combination = itertools.combinations(file_list, people_num)
    for comb in tqdm(combination):
        beamformed_das(comb, people_num)
        # beamformed_das(file_list[i], file_list[i+1], people_num)


create_dir()

people_num = 2
build = True 
crm = True 

if build:
    # Generating mix and crm data from dataset
    generate_data_list(people_num)  
    # beamformed_doa_plot()

if crm:
    generate_crm_list()


# Split dataset Train:50% and Validation 50%
split_dataset()
