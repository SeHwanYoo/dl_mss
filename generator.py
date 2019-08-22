import numpy as np
from tensorflow.keras.utils import Sequence
# from tensorflow.keras

class multiple_mics_generator(Sequence):

    def __init__(self, model_vars, vars):
        self.people_num = model_vars['people_num']
        self.epochs = model_vars['epochs']
        self.init_epochs = model_vars['init_epochs'] 
        self.files = vars['files']
        self.batch_size = 4
        self.shuffle = vars['shuffle']
        self.on_epoch_end()

        self.x_dim = (298, 257, 2)
        self.y_dim = (298, 257, 2, 2)

        self.data_path = vars['path'] 

    def __len__(self):
        return int(np.floor(len(self.files)) / self.batch_size)

    def __getitem__(self, index):
        idx = self.idx[index * self.batch_size: (index + 1) * self.batch_size]
        file_names = [self.files[i] for i in idx]
        X, Y = self.__getdata__(file_names)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idx = np.arange(len(self.files))   
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __getdata__(self, file_name):
        X1_batch = np.empty((self.batch_size, *self.x_dim))
        X2_batch = np.empty((self.batch_size, *self.x_dim))
        X3_batch = np.empty((self.batch_size, *self.x_dim))
        X4_batch = np.empty((self.batch_size, *self.x_dim))
        X5_batch = np.empty((self.batch_size, *self.x_dim))
        Y = np.empty((self.batch_size, *self.y_dim))


        for n, f_id in enumerate(file_name):
            f_info = f_id.strip().split('/')

            # Same file name, but different folder because of those are coming from different microphones
            X1_batch[n, ] = np.load(self.data_path + 'beamformed/beam1/' + f_info[0])
            X2_batch[n, ] = np.load(self.data_path + 'beamformed/beam2/' + f_info[0])
            X3_batch[n, ] = np.load(self.data_path + 'beamformed/beam3/' + f_info[0])
            X4_batch[n, ] = np.load(self.data_path + 'beamformed/beam4/' + f_info[0])
            X5_batch[n, ] = np.load(self.data_path + 'beamformed/beam5/' + f_info[0])
            
            # Complex Ratio Mask
            for k in range(2):
                Y[n, :, :, :, k] = np.load(self.data_path + 'crm/' +  f_info[k + 1])

        X_batch = [X1_batch, X2_batch, X3_batch, X4_batch, X5_batch] 

        return X_batch, Y
