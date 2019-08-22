import keras.backend as K

# def model_loss(gamma=0.1,num_speaker=2):
def discriminate_loss(model_vars):

    num_speaker = model_vars['people_num']
    gamma = model_vars['gamma_loss']
    

    # true:True target, Pred: Prediction 
    def loss(S_true,S_pred,gamma=gamma,num_speaker=num_speaker):
        sum = 0
        for i in range(num_speaker):
            sum += K.sum(K.flatten((K.square(S_true[:,:,:,i]-S_pred[:,:,:,i]))))
            for j in range(num_speaker):
                if i != j:
                    sum -= gamma*K.sum(K.flatten((K.square(S_true[:,:,:,i]-S_pred[:,:,:,j]))))

        loss = sum / (num_speaker*298*257*2)
        return loss
    
    # return loss, oprimizer
