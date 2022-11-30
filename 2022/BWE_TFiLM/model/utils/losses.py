import keras.backend as K
import tensorflow as tf
import functools
from model.utils.OBM import *    
window_fn = functools.partial(tf.signal.hann_window, periodic=True)

def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def si_sdr_loss(y_true, y_pred):
    print("######## SI-SDR LOSS ########")
    print("y_true shape:      ", K.int_shape(y_true))   
    print("y_pred shape:      ", K.int_shape(y_pred)) 

    x = K.squeeze(y_true,axis=-1)
    y = K.squeeze(y_pred,axis=-1)
    y_pred_shape = K.shape(y_pred)
    
    smallVal = 0.0000000001 # To avoid divide by zero
    
    a = K.sum(y*x,axis=-1,keepdims=True) / (K.sum(x*x,axis=-1,keepdims=True) + smallVal)
    #print("a shape:      ", K.int_shape(a)) 
    #print("a:            ", K.eval(a))
    
    xa = a * x;
    #print("xa shape:      ", K.int_shape(xa)) 
    #print("xa:            ", K.eval(xa))
    
    xay = xa - y;
    #print("xay shape:      ", K.int_shape(xay)) 
    #print("xay:            ", K.eval(xay))
    
    d = K.sum(xa*xa,axis=-1,keepdims=True)/(K.sum(xay*xay,axis=-1,keepdims=True) + smallVal)
    #print("d shape:      ", K.int_shape(d)) 
    #print("d:            ", K.eval(d))
    
    d = -K.mean(10*log10(d))
    
    #print("d shape:      ", K.int_shape(d)) 
    #print("d:            ", K.eval(d))
    
    print("Compiling SI-SDR LOSS Done!")
    return d


def estoi_loss(I=8,nbf=200):
    def estoi_loss_inner(y_true, y_pred):
        # print("######## ESTOI LOSS ########")
        # print("y_true shape:      ", K.int_shape(y_true))   
        # print("y_pred shape:      ", K.int_shape(y_pred)) 

        y_true = K.squeeze(y_true,axis=-1)
        y_pred = K.squeeze(y_pred,axis=-1)
        y_pred_shape = K.shape(y_pred)
        
        stft_true = tf.signal.stft(y_true,256,128,512,window_fn,pad_end=False)  
        stft_pred = tf.signal.stft(y_pred,256,128,512,window_fn,pad_end=False)  
        # print("stft_true shape:   ", K.int_shape(stft_true))
        # print("stft_pred shape:   ", K.int_shape(stft_pred))
        
        OBM1 = tf.convert_to_tensor(OBM)
        OBM1 = K.tile(OBM1,[y_pred_shape[0],1,])
        OBM1 = K.reshape(OBM1,[y_pred_shape[0],15,257,])
        # print("OBM1 shape:        ", K.int_shape(OBM1))
        
        OCT_pred = K.sqrt(tf.matmul(OBM1,K.square(K.abs(tf.transpose(stft_pred,perm=[0,2,1])))))    
        OCT_true = K.sqrt(tf.matmul(OBM1,K.square(K.abs(tf.transpose(stft_true,perm=[0,2,1])))))  
        OCT_pred_shape = K.shape(OCT_pred)
      
        
        #print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[0]))
        #print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[1]))  
        #print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[2]))
        N = 30          # length of temporal envelope vectors
        J = 15          # Number of one-third octave bands (cannot be varied) 
        M = int(nbf-(N-1)) # number of temporal envelope vectors
        smallVal = 0.0000000001 # To avoid divide by zero

       

        d = K.variable(0.0,'float')
        for i in range(0, I):
            for m in range(0, M):    
                x = K.squeeze(tf.slice(OCT_true,    [i,0,m], [1,J,N]),axis=0)
                y = K.squeeze(tf.slice(OCT_pred,    [i,0,m], [1,J,N]),axis=0)                   
                #print("x shape:   ", K.int_shape(x))
                #print("y shape:   ", K.int_shape(y))                                
                #print("x shape:   ", K.eval(x))
                #print("y shape:   ", K.eval(y))
                           
                xn = x-K.mean(x,axis=-1,keepdims=True)
                #print("xn shape:   ", K.eval(xn))
                yn = y-K.mean(y,axis=-1,keepdims=True)
                #print("yn shape:   ", K.eval(yn))            
                
                xn = xn / (K.sqrt(K.sum(xn*xn,axis=-1,keepdims=True)) + smallVal )
                #print("xn shape:   ", K.eval(xn))
                
                yn = yn / (K.sqrt(K.sum(yn*yn,axis=-1,keepdims=True)) + smallVal )
                #print("yn shape:   ", K.eval(yn))
                
                xn = xn - K.tile(K.mean(xn,axis=-2,keepdims=True),[J,1,])
                #print("xn shape:   ", K.eval(xn))

                yn = yn - K.tile(K.mean(yn,axis=-2,keepdims=True),[J,1,])
                #print("yn shape:   ", K.eval(yn))
                
                xn = xn / (K.sqrt(K.sum(xn*xn,axis=-2,keepdims=True)) + smallVal )
                #print("xn shape:   ", K.eval(xn))
                
                yn = yn / (K.sqrt(K.sum(yn*yn,axis=-2,keepdims=True)) + smallVal )
                #print("yn shape:   ", K.eval(yn))
                            
                di = K.sum( xn*yn ,axis=-1,keepdims=True)
                #print("di shape:   ", K.eval(di))
                di = 1/N*K.sum( di ,axis=0,keepdims=False)
                #print("di shape:   ", K.eval(di))
                d  = d + di
                #print("d shape:   ", K.eval(d))

        # print("Compiling ESTOI LOSS Done!")
        return 1-(d/K.cast(I*M,dtype='float'))
    return estoi_loss_inner




def stoi_loss(I=8,nbf=200):
    def stoi_loss_inner(y_true, y_pred):
        # print("######## STOI LOSS ########")
        # print("y_true shape:      ", K.int_shape(y_true))   
        # print("y_pred shape:      ", K.int_shape(y_pred)) 
        # exit()

        y_true = K.squeeze(y_true,axis=-1)
        y_pred = K.squeeze(y_pred,axis=-1)
        y_pred_shape = K.shape(y_pred)
        
        stft_true = tf.signal.stft(y_true,256,128,512,window_fn,pad_end=False)  
        stft_pred = tf.signal.stft(y_pred,256,128,512,window_fn,pad_end=False)  
        # print("stft_true shape:   ", K.int_shape(stft_true))
        # print("stft_pred shape:   ", K.int_shape(stft_pred))
        
        OBM1 = tf.convert_to_tensor(OBM)
        OBM1 = K.tile(OBM1,[y_pred_shape[0],1,])
        OBM1 = K.reshape(OBM1,[y_pred_shape[0],15,257,])
        # print("OBM1 shape:        ", K.int_shape(OBM1))
        
        OCT_pred = K.sqrt(tf.matmul(OBM1,K.square(K.abs(tf.transpose(stft_pred,perm=[0,2,1])))))    
        OCT_true = K.sqrt(tf.matmul(OBM1,K.square(K.abs(tf.transpose(stft_true,perm=[0,2,1])))))  
        OCT_pred_shape = K.shape(OCT_pred)
      
        
        #print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[0]))
        #print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[1]))  
        #print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[2]))
         
        N = 30          # length of temporal envelope vectors
        J = 15          # Number of one-third octave bands (cannot be varied) 
        M = int(nbf-(N-1)) # number of temporal envelope vectors
        smallVal = 0.0000000001 # To avoid divide by zero
        doNorm   = True
        
        # if doNorm:
        #     print("Apply Normalization and Clipping")
        
        c = K.constant(5.62341325,'float')  # 10^(-Beta/20) with Beta = -15
        d = K.variable(0.0,'float')
        for i in range(0, I): # Run over mini-batches
            for m in range(0, M): # Run over temporal envelope vectors    
                x = K.squeeze(tf.slice(OCT_true,    [i,0,m], [1,J,N]),axis=0)
                y = K.squeeze(tf.slice(OCT_pred,    [i,0,m], [1,J,N]),axis=0)                   
                #print("x shape:   ", K.int_shape(x))
                #print("y shape:   ", K.int_shape(y))                                
                #print("x shape:   ", K.eval(x))
                #print("y shape:   ", K.eval(y))
                
                if doNorm:
                    alpha = K.sqrt(K.sum(K.square(x),axis=-1,keepdims=True) / (K.sum(K.square(y),axis=-1,keepdims=True)) + smallVal)
                    #print("alpha shape:   ", K.int_shape(alpha))                                
                    #print("alpha shape:   ", K.eval(alpha))
                    
                    alpha = K.tile(alpha,[1,N,])
                    #print("alpha shape:   ", K.int_shape(alpha))                                
                    #print("alpha shape:   ", K.eval(alpha))

                    ay	= y*alpha   
                    #print("aY shape:   ", K.int_shape(ay))                                
                    #print("aY shape:   ", K.eval(ay))

                    y = K.minimum(ay,x+x*c)
                    #print("aY shape:   ", K.int_shape(ay))                                
                    #print("aY shape:   ", K.eval(ay)) 
                
                xn = x-K.mean(x,axis=-1,keepdims=True)
                #print("xn shape:   ", K.eval(xn))
                xn = xn / (K.sqrt(K.sum(xn*xn,axis=-1,keepdims=True)) + smallVal )
                #print("xn shape:   ", K.eval(xn))
                yn = y-K.mean(y,axis=-1,keepdims=True)
                #print("yn shape:   ", K.eval(yn))
                yn = yn / (K.sqrt(K.sum(yn*yn,axis=-1,keepdims=True)) + smallVal )
                #print("yn shape:   ", K.eval(yn))
                di = K.sum( xn*yn ,axis=-1,keepdims=True)
                #print("di shape:   ", K.eval(di))
                d  = d + K.sum( di ,axis=0,keepdims=False)
                #print("d shape:   ", K.eval(K.sum( di ,axis=0,keepdims=False)))

        # print("Compiling STOI LOSS Done!")
        return 1-(d/K.cast(I*J*M,dtype='float'))
    return stoi_loss_inner


    
    
def stsa_mse(y_true, y_pred):
    print("######## STSA-MSE LOSS ########")
    print("y_true shape:      ", K.int_shape(y_true))   
    print("y_pred shape:      ", K.int_shape(y_pred)) 
    
    y_true = K.squeeze(y_true,axis=-1)
    y_pred = K.squeeze(y_pred,axis=-1)
    
    stft_true = K.abs(tf.contrib.signal.stft(y_true,256,128,256,window_fn,pad_end=False))  
    stft_pred = K.abs(tf.contrib.signal.stft(y_pred,256,128,256,window_fn,pad_end=False))  
    print("stft_true shape:   ", K.int_shape(stft_true))
    print("stft_pred shape:   ", K.int_shape(stft_pred))
    
    d = K.mean(K.square(stft_true-stft_pred))

    print("Compiling STSA-MSE LOSS Done!")
    return d