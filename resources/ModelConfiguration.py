from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense,TimeDistributed, concatenate
import tensorflow.keras as K

def get_model_custom(lenvocab, LR=0.0005, maxlength=50, drop_lstm=0.4, drop_rec=0.4, emb_size1=64, emb_size2=64, lstm_units=256): ## TC = Training configuration
    
    input1 =  Input(shape=(maxlength,))
    uni_layer = Embedding(lenvocab,emb_size1, input_length=maxlength, mask_zero=True)(input1)
    input2   = Input(shape=(maxlength,))
    bi_layer = Embedding(lenvocab,emb_size2, input_length=maxlength, mask_zero=True)(input2)
    ngram_layer = concatenate([uni_layer, bi_layer])
    lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=drop_lstm, recurrent_dropout=drop_rec))(ngram_layer)
    time_dist_layer = TimeDistributed(Dense(4, activation='softmax'))(lstm_layer)
    model = K.models.Model([input1,input2],time_dist_layer)    
    optim = K.optimizers.Adam(lr=LR)
    
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics = ['acc', K.metrics.Recall(), K.metrics.Precision()])

    return model