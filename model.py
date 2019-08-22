from tqdm import tqdm
from sklearn.model_selection import train_test_split
# from speech_model import *
import model_speech as sp 
from model_loss import discriminate_loss as speech_loss 

from datetime import datetime 

import os
import scipy.io as sio
# from speech_model import making_speech_model as speech_model

# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
# from keras.models import Model, load_model
# from MyGenerator import AudioGenerator
# from keras.callbacks import TensorBoard
# from keras import optimizers

from tensorflow.keras import Model, optimizers
from generator import multiple_mics_generator

from tensorflow.keras.losses import MeanSquaredError

os.environ['CUDA_VISIBEL_DEVICES'] = '0'

DATA_PATH = './data/'
BEAMFORMED_PATH = DATA_PATH + 'beamformed/'

TRAIN_PATH = DATA_PATH + 'train/'
TEST_PATH = DATA_PATH + 'test/'
VALIDATION_PATH = DATA_PATH + 'validation/'


if __name__ == '__main__':

    model_vars = {
        'people_num': 2,
        'epochs': 100,
        'init_epochs': 0,
        'batch_size': 8,
        'gamma_loss': 0.1,
        'beta_loss': 0.5,
    }


    speech_model = sp.speech_model(model_vars['people_num'])
    speech_model.compile(optimizer=optimizers.Adam(), loss=MeanSquaredError())

    trainfiles = []
    testfiles = [] 
    with open((DATA_PATH + 'txt/train.txt'), 'r') as f:
        trainfiles = f.readlines() 

    with open((DATA_PATH + 'txt/test.txt'), 'r') as f:
        testfiles = f.readlines()

    train_vars = {
        'path': DATA_PATH,
        'files': trainfiles, 
        'batch_size': 1, 
        'shuffle': True
    }

    validation_vars = {
        'path': DATA_PATH,
        'files': testfiles, 
        'batch_size': 1, 
        'shuffle': True
    }

    train_generator = multiple_mics_generator(model_vars, train_vars) 
    validation_generator = multiple_mics_generator(model_vars, validation_vars)

    print('----------Training---------------')

    speech_model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator)

    # speech_model.train_on_batch(
    # for batch in generator:
    #     train_on_batch(X, Y)

    from keras.models import load_model
    model.save('./model/model_speech.h5')


    # print('----------Evaluate---------------')
    # scores = model.evaluate_generator(test_generator, steps=5)
    # print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

    # # 6. 모델 사용하기
    # print("-- Predict --")
    # output = model.predict_generator(test_generator, steps=5)
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # print(test_generator.class_indices)
    # print(output)   
