# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback
import pickle
import time, datetime
    
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def plot_hist(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
  
def save_model_results(model,
                       history, 
                       root_path, 
                       model_name, 
                       saveModelFull=True, 
                       saveModelJSON=True,
                       saveModelPlot=True, 
                       saveWeights=True,
                       saveHistory=True,
                       verbose=True):
    models_path = root_path+'../models/'
    if saveModelFull:
        if verbose: print("[INFO] Saving full model...")
        model.save(models_path + model_name + '_model.h5')    
        if verbose: print("[INFO] Model saved successfully.\n")
        
    #if saveModelJSON: #JSON can be called only when the model is not compiled to TPU
    #    with open(models_path + model_name + "_model.json", "w") as json_file:
    #        json_file.write(model.to_json())
            
    if saveModelPlot:
        if verbose: print("[INFO] Saving model plot...")
        plot_model(model, to_file=models_path + model_name + '_modelplot.png')
        if verbose: print("[INFO] Model plot saved.\n")
        
    if saveWeights:
        if verbose: print("[INFO] Saving model weights...")
        model.save_weights(models_path + model_name + '_weigths.h5')
        if verbose: print("[INFO] Model weights saved successfully.\n")
        
    if saveHistory:
        if verbose: print("[INFO] Saving history of the training...")
        with open(models_path + model_name+ '_history.plk', 'wb') as file:
            pickle.dump(history.history, file)
        if verbose: print("[INFO] History of the training successfully.\n")

def get_save_name(model_name):
    save_file_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_')+model_name
    return save_file_name

#class train_config():
#  def __init__(self,tag,dataset,lenvocab,opt='adam', lr=0.001, drop_lstm=0.0, drop_rec=0.0, emb_size1=64, emb_size2=64, batch_size=256, lstm_units=256,maxlength=50):
#    self.tag=tag
#    self.dataset=dataset
#    self.lenvocab=lenvocab
#    self.opt=opt
#    self.lr=lr
#    self.drop_lstm=drop_lstm
#    self.drop_rec=drop_rec
#    self.emb_size1=emb_size1
#    self.emb_size2=emb_size2
#    self.batch_size=batch_size
#    self.lstm_units=lstm_units
#    self.maxlength=maxlength
#
#def get_model_custom(TC): ## TC = Training configuration
#    
#    input1 =  Input(shape=(TC.maxlength,))
#    uni_layer = Embedding(TC.lenvocab,TC.emb_size1, input_length=TC.maxlength, mask_zero=True)(input1)
#    input2   = Input(shape=(TC.maxlength,))
#    bi_layer = Embedding(TC.lenvocab,TC.emb_size1, input_length=TC.maxlength, mask_zero=True)(input1)
#    ngram_layer = concatenate([uni_layer, bi_layer])
#    lstm_layer = Bidirectional(LSTM(TC.lstm_units, return_sequences=True, dropout=TC.drop_lstm, recurrent_dropout=TC.drop_rec))(ngram_layer)
#    time_dist_layer = TimeDistributed(Dense(4, activation='softmax'))(lstm_layer)
#    model = K.models.Model([input1,input2],time_dist_layer)
#    
#    if tconfig.opt=='adam':
#      optim = K.optimizers.Adam(lr=tconfig.lr)
#    else:
#      optim = K.optimizers.Adam(lr=tconfig.lr)
#    
#    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics = ['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
#    return model
#  
#def get_model_name(tc):
#  model_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_')+"%s_%s_%s_lr%s_drLS%s_drRec%s_emb1%s_emb2%s_batch%s_units%s_maxl%s" % (str(tc.tag),str(tc.dataset), tc.opt, str(tc.lr), str(tc.drop_lstm), str(tc.drop_rec), str(tc.emb_size1), str(tc.emb_size2), str(tc.batch_size), str(tc.lstm_units), str(tc.maxlength) )
#  return model_name

# Model to TPU
#TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
#tf.logging.set_verbosity(tf.logging.INFO)

#def toTPU(mdl):
    #tpu_model = tf.contrib.tpu.keras_to_tpu_model(mdl,strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
    #return tpu_model