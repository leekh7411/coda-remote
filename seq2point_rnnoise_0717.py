
# coding: utf-8

# In[1]:


#!pip install keras==2.1.6
#!ls


# In[2]:


#!wget http://210.89.182.185:7070/coda/GM12878.npz
#!wget http://210.89.182.185:7070/coda/GM18526.npz


# In[3]:


INPUT_MARKS  = ["H3K27AC", "H3K27ME3", "H3K36ME3", "H3K4ME1", "H3K4ME3", "INPUT"]
OUTPUT_MARKS = ["H3K27AC"]
MARK_INDEX   = [0, 1, 2, 3, 4, 5]
PEAK_MARK_INDEX = [0, 1, 2, 3, 4]
SEQ_LENGTH = 1001
#DATA_PATH = './data/GM12878_5+1marks-K4me3_all_subsample-0.5e6-None_rS-0_numEx-1000_seqLen-1001_peakFrac-0.5_peaksFac-H3K27AC_norm-arcsinh.npz'
DATA_PATH = 'GM12878.npz'
EVAL_PATH = 'GM18526.npz'
zero_out_non_bins = True


# ### Preprocessed dataset
# * Load GM12878 dataset 
# * example size 10000

# In[4]:


import os 
import numpy as np

def input_not_before_end(list_of_marks):
    return ('INPUT' not in list_of_marks[:-1])

def load_seq_dataset():
    seq_length = SEQ_LENGTH
    input_marks = INPUT_MARKS
    output_marks = OUTPUT_MARKS
  
    assert(input_not_before_end(output_marks))
    assert(input_not_before_end(input_marks))
    
    dataset_path = os.path.join(DATA_PATH)

    #try:      
    with np.load(dataset_path) as data:
        X = data['X'].astype('float32')
        Y = data['Y'].astype('float32')
        peakPValueX = data['peakPValueX'].astype('float32')
        peakPValueY = data['peakPValueY'].astype('float32')
        peakBinaryX = data['peakBinaryX'].astype('int8')
        peakBinaryY = data['peakBinaryY'].astype('int8')
    #except:
        #raise Exception, "Dataset doesn't exist or is missing a required matrix."

    
    marks_idx =  MARK_INDEX
    peak_marks_idx = PEAK_MARK_INDEX
    
    X = X[..., marks_idx]
    peakPValueX = peakPValueX[..., peak_marks_idx]
    peakBinaryX = peakBinaryX[..., peak_marks_idx]

    assert(np.all(peakPValueX >= 0) & np.all(peakPValueY >= 0))

    if (X.shape[0], X.shape[1]) != (Y.shape[0], Y.shape[1]):
        raise Exception, "First two dimensions of X and Y shapes (num_examples, seq_length)                           need to agree."
    if (peakPValueX.shape[0], peakPValueX.shape[1]) != (peakPValueY.shape[0], peakPValueY.shape[1]):
        raise Exception, "First two dimensions of peakPValueX and peakPValueY shapes                           (num_examples, seq_length) need to agree."
    if len(peakPValueX) != len(X):
        raise Exception, "peakPValueX and X must have same length."

    if ((seq_length != X.shape[1]) or (seq_length != peakPValueX.shape[1])):
        raise Exception, "seq_length between model and data needs to agree"

    return X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY


# #### Seq2point model's data
# * X: (num_examples) x (seq_length) x (num_input_marks)
# * Y: (num_examples) x ( 1 ) x (num_output_marks)
# 
# #### Seq2seq model's data
# * X: (num_examples) x (seq_length) x (num_input_marks)
# * Y: (num_examples) x (seq_length) x (num_output_marks)

# In[5]:


X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY = load_seq_dataset()

if zero_out_non_bins:
    peakPValueX = peakPValueX * peakBinaryX
    peakPValueY = peakPValueY * peakBinaryY 

def sq2p_process_X(X):
    return X

def sq2p_process_Y(Y):
    '''mid = (SEQ_LENGTH - 1) / 2
    mid = int(mid)
    return Y[:, mid:mid+1, :]'''
    return Y
    
    
X = sq2p_process_X(X)
Y = sq2p_process_Y(Y)
peakPValueX = sq2p_process_X(peakPValueX)
peakPValueY = sq2p_process_Y(peakPValueY)
peakBinaryX = sq2p_process_X(peakBinaryX)
peakBinaryY = sq2p_process_Y(peakBinaryY)    


# In[6]:


class DataNormalizer(object):
    def __init__(self, mode):
        self.b = None
        self.W = None
        self.mode = mode
        if mode not in ['ZCA', 'Z', '01', 'identity']:
            raise ValueError, "mode=%s must be 'ZCA', 'Z', '01', or 'identity'" % mode


    def fit(self, X_orig):
        """
        Learns scaling parameters on the X_orig dataset. Does not modify X_orig.
        """        
        if len(X_orig.shape) != 2 and len(X_orig.shape) != 3:
            raise ValueError, "X must be either a 3-tensor of shape num_examples x seq_length x                                num_input_marks, or a 2-tensor of shape num_examples x num_input_marks"
        if self.mode == 'identity':
            return None        

        X = np.copy(X_orig)
        num_input_marks = X.shape[-1]

        # If X is a 3-tensor, reshape X such that it is a 2-tensor of shape 
        # (num_examples * seq_length) x num_input_marks. 
        if len(X.shape) == 3:    
            X = np.reshape(X, (-1, num_input_marks))
        
        self.b = np.mean(X, axis=0) 

        X -= self.b

        if self.mode == 'ZCA':
            sigma = np.dot(X.T, X) / X.shape[0]
            U, S, V = np.linalg.svd(sigma)
            self.W = np.dot(
                np.dot(U, np.diag(1 / np.sqrt(S + 1e-5))),
                U.T)
        elif self.mode == 'Z':
            self.W = np.empty(num_input_marks)
            for idx in range(num_input_marks):
                self.W[idx] = np.std(X[:, idx])
        elif self.mode == '01':
            self.W = np.empty(num_input_marks)
            for idx in range(num_input_marks):
                self.W[idx] = np.max(np.abs(X[:, idx]))

        return None            


    def transform(self, X):
        if len(X.shape) != 2 and len(X.shape) != 3:
            raise ValueError, "X must be either a 3-tensor of shape num_examples x seq_length x                                num_input_marks, or a 2-tensor of shape num_examples x num_input_marks"

        if self.mode == 'identity':
            return X
            
        assert self.b is not None
        assert self.W is not None

        num_input_marks = X.shape[-1]
        orig_shape = X.shape

        if self.mode == 'ZCA':            
            X = np.reshape(X, (-1, num_input_marks))
            if self.W.shape[1] != X.shape[1]:
                raise ValueError, "When doing a ZCA transform, X and W must have the same number of columns."
            X = np.dot(
                X - self.b,
                self.W.T)
            X = np.reshape(X, orig_shape)
        elif self.mode in ['Z', '01']:
            if (len(self.b) != num_input_marks) or (len(self.W) != num_input_marks):
                print("X.shape: ", X.shape)
                print("b.shape: ", self.b.shape)
                print("W.shape: ", self.W.shape)
                raise ValueError, "The shapes of X, b, and W must all share the same last dimension."                
            for idx in range(num_input_marks):
                X[..., idx] = (X[..., idx] - self.b[idx]) / self.W[idx]

        return X


# In[7]:


scale_input = "01"
normalizer = DataNormalizer(scale_input)
normalizer.fit(X)
X = normalizer.transform(X)


# In[8]:


NB_EPOCH          = 10
VALID_SPLIT       = 0.05
BATCH_SIZE        = 100


# In[9]:


from keras.callbacks import ModelCheckpoint, EarlyStopping

seq2point_weight_bin = 's2q-rnn'
bin_checkpointer = ModelCheckpoint(
    filepath=os.path.join('.', '%s-weights.hdf5'%seq2point_weight_bin), 
    verbose=1, 
    save_best_only=True)

bin_earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=0)


# In[10]:


import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras import losses
from keras import regularizers
from keras.constraints import min_max_norm
from keras.constraints import Constraint
from keras import backend as K
def rnn_model():
    main_input = Input(shape=(SEQ_LENGTH, len(INPUT_MARKS)), name='main_input')
    # Noise Spectral Estimation
    noise_gru = GRU(
      48, 
      activation='relu', 
      recurrent_activation='sigmoid', 
      return_sequences=True, 
      name='noise_gru'
    )(main_input)

    # Spectral Subtraction
    denoise_input = keras.layers.concatenate([noise_gru, main_input])
    denoise_gru = GRU(
      96, 
      activation='relu', 
      recurrent_activation='sigmoid', 
      return_sequences=True, 
      name='denoise_gru'
    )(denoise_input)
        
    denoise_output = Dense(
      1, 
      activation='relu', 
      name='denoise_output'
    )(denoise_gru)
    
    '''
    denoise_output = keras.layers.Conv1D(
        filters=len(OUTPUT_MARKS),
        kernel_size=SEQ_LENGTH,
        strides=1,
        padding='same',
        activation='relu'
    )(denoise_dense)
    '''

    # Peak ? Nope
    peak_input = keras.layers.concatenate([denoise_gru, main_input])  

    peak_gru = GRU(
      24, 
      activation='relu', 
      recurrent_activation='sigmoid', 
      return_sequences=True, 
      name='peak_gru'
    )(peak_input)
    
    peak_output = Dense(
      1, 
      activation='sigmoid', 
      name='peak_output'
    )(peak_gru)
    '''
    peak_output = keras.layers.Conv1D(
        filters=len(OUTPUT_MARKS),
        kernel_size=SEQ_LENGTH,
        strides=1,
        padding='same',
        activation='sigmoid'
    )(peak_dense)
    '''

    model = Model(inputs=main_input, outputs=[denoise_output, peak_output])

    return model


# In[11]:


rnn_model = rnn_model()
rnn_model.summary()


# In[12]:


rnn_model.compile(loss=[losses.mean_squared_error, losses.binary_crossentropy],
              optimizer='adam')


# In[13]:


print('X-shape:',X.shape)
print('Y-shape:',Y.shape)
print('pY-shape:',peakBinaryY.shape)

rnn_model.fit(X, [Y, peakBinaryY], 
             callbacks=[bin_checkpointer, bin_earlystopper],
             epochs=NB_EPOCH,
             validation_split=VALID_SPLIT,
             batch_size=BATCH_SIZE)

