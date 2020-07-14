#A De La iglesia Lab projectect
#Sleep Identification Enabled by Supervised Training Algorithms
#Codename: SIESTA

#Load Kivy for gui
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.widget import Widget
from kivy.clock import Clock

#Load Panda for data manage
import pandas as pd
import pyedflib 
import datetime
import time
import numpy as np
import itertools
import scipy 
import scipy.signal
from scipy.signal import butter, lfilter

from sklearn import preprocessing
import os
#CODES

def calculate_psd_and_f(signal,fs,epoch):
    epoch = epoch*fs
    corr_signal = signal[:len(signal)-(len(signal)%epoch)]
    new_signal = np.reshape(corr_signal,(len(corr_signal)//epoch,epoch))
    fr,p = scipy.signal.welch(new_signal,fs=fs,nperseg=fs*10,scaling='spectrum')
    return fr,p

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a    

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def CreateFeaturesDataFrame(eeg,emg,epoch=10,fs=400):
    eeg_features,eeg_feature_labels = extract_features(eeg,'EEG',epoch=epoch,fs=fs,is_emg=False)
    emg_features,emg_feature_labels = extract_features(emg,'EMG',epoch=10,fs=400,is_emg=True)
    feature_matrix = np.column_stack((eeg_features,emg_features[:,1:]))
    feature_labels = ['Epoch'] + eeg_feature_labels[1:] + emg_feature_labels[1:]
    data = pd.DataFrame(feature_matrix,columns=feature_labels)
    final_data = pd.DataFrame(data.iloc[:,1:])
    return final_data

def readEDFfile(file_name):

    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    eeg1 = f.readSignal(0)
    eeg2 = f.readSignal(1)
    emg = f.readSignal(2)
    
    start_date = datetime.datetime(f.getStartdatetime().year,f.getStartdatetime().month,f.getStartdatetime().day,
                                   f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second)
    fd = f.getFileDuration()
    file_end = start_date + datetime.timedelta(seconds = fd)

    step = datetime.timedelta(seconds=10)  

    time_stamps = []

    while start_date < file_end:
        time_stamps.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
        start_date += step
    
    return eeg1,eeg2,emg,time_stamps

def extract_features(signal,signal_label,epoch,fs,is_emg=False):
    
    
    fr,p = calculate_psd_and_f(signal,fs,epoch)
    epoch = epoch*fs
    
    #Data from EEG
    
    if (is_emg == False):
        max_fr = 50
        
        ## Calculate the total power, total power per epoch, and extract the relevant frequencies.
        ## IMPORTANT NOTE: These are not the ACTUAL power values, they are standardized to account
        ## for individual variability, and are thus relative.
        freq = fr[(fr>=0.5) & (fr <=max_fr)]
        sum_power = p[:,(fr>=0.5) & (fr <=max_fr)]
        max_power = np.max(sum_power,axis=1)
        min_power = np.min(sum_power,axis=1)
        range_power = max_power - min_power
        std_power = ((sum_power.T-min_power)/range_power).T
           
        ## Calculate the relative power at the different brain waves:
        delta = np.sum(std_power[:,(freq>=0.5) & (freq <=4)],axis=1)
        
         
        thetacon = np.sum(std_power[:,(freq>=4) & (freq <=12)],axis=1)
        theta1 = np.sum(std_power[:,(freq>=6) & (freq <=9)],axis=1)
        theta2 = np.sum(std_power[:,(freq>=5.5) & (freq <=8.5)],axis=1)
        theta3 = np.sum(std_power[:,(freq>=7) & (freq <=10)],axis=1)
        
        beta = np.sum(std_power[:,(freq>=20) & (freq <=40)],axis=1)
                
        alpha = np.sum(std_power[:,(freq>=8) & (freq <=13)],axis=1)
        sigma = np.sum(std_power[:,(freq>=11) & (freq <=15)],axis=1)
        spindle = np.sum(std_power[:,(freq>=12) & (freq <=14)],axis=1)
        gamma= np.sum(std_power[:,(freq>=35) & (freq <=45)],axis=1)
        
        temp1= np.sum(std_power[:,(freq>=0.5) & (freq <=20)],axis=1)
        temp2= np.sum(std_power[:,(freq>=0.5) & (freq <=50)],axis=1)
        
        temp3= np.sum(std_power[:,(freq>=0.5) & (freq <=40)],axis=1)
        temp4= np.sum(std_power[:,(freq>=11) & (freq <=16)],axis=1)
        
        
        EEGrel1 = thetacon/delta;
        EEGrel2 = temp1/temp2;
        EEGrel3 = temp4/temp3;
        
        hann = np.hanning(12);
        
        spindelhan1=np.convolve(hann,EEGrel3,'same');
        
        spindelhan=np.transpose(spindelhan1);
        
        ## Calculate the 90% spectral edge:
        spectral90 = 0.9*(np.sum(sum_power,axis=1))
        s_edge = np.cumsum(sum_power,axis=1)
        l = [[n for n,j in enumerate(s_edge[row_ind,:]) if j>=spectral90[row_ind]][0] for row_ind in range(s_edge.shape[0])]
        spectral_edge = np.take(fr,l) # spectral edge 90%, the frequency below which power sums to 90% of the total power
        
         ## Calculate the 50% spectral mean:
        spectral50 = 0.5*(np.sum(sum_power,axis=1))
        s_mean = np.cumsum(sum_power,axis=1)
        l = [[n for n,j in enumerate(s_mean[row_ind,:]) if j>=spectral50[row_ind]][0] for row_ind in range(s_mean.shape[0])]
        spectral_mean50 = np.take(fr,l) 
                
    else:
        #for EMG
        max_fr = 100
        
        ## Calculate the total power, total power per epoch, and extract the relevant frequencies: 
        freq = fr[(fr>=0.5) & (fr <=max_fr)]
        sum_power = p[:,(fr>=0.5) & (fr <=max_fr)]
        max_power = np.max(sum_power,axis=1)
        min_power = np.min(sum_power,axis=1)
        range_power = max_power - min_power
        std_power = ((sum_power.T-min_power)/range_power).T
    
    
    ## Calculate the Root Mean Square of the signal
    signal = signal[0:p.shape[0]*epoch]
    s = np.reshape(signal,(p.shape[0],epoch))
    rms = np.sqrt(np.mean((s)**2,axis=1)) #root mean square
    ## Calculate amplitude and spectral variation:
    amplitude = np.mean(np.abs(s),axis=1)
    amplitude_m=np.median(np.abs(s),axis=1)
    signal_var = (np.sum((np.abs(s).T - np.mean(np.abs(s),axis=1)).T**2,axis=1)/(len(s[0,:])-1)) # The variation
    ## Calculate skewness and kurtosis
    m3 = np.mean((s-np.mean(s))**3,axis=1) #3rd moment
    m2 = np.mean((s-np.mean(s))**2,axis=1) #2nd moment
    m4 = np.mean((s-np.mean(s))**4,axis=1) #4th moment
    skew = m3/(m2**(3/2)) # skewness of the signal, which is a measure of symmetry
    kurt = m4/(m2**2) #kurtosis of the signal, which is a measure of tail magnitude
    
    ## Calculate more time features
    
    signalzero=preprocessing.maxabs_scale(s,axis=1)
    zerocross = (np.diff(np.sign(signalzero)) != 0).sum(axis=1)
        
    maxs = np.amax(s,axis=1)
    mins = np.amin(s,axis=1)
    
    peaktopeak= maxs - mins
    
    arv1 = ((np.abs(s)))

    arv = np.sum(arv1,axis=1)

    arv = arv / len(s)
                 
    #Energy and amplitud           
    
            
    deltacomp = butter_bandpass_filter(s, 0.5, 4, fs, 5)
    #calculate energy like this
    deltaenergy = sum([x*2 for x in np.matrix.transpose(deltacomp)])
    deltaamp = np.mean(np.abs(deltacomp),axis=1)
         
        
    thetacomp = butter_bandpass_filter(s, 4, 12, fs, 5)
    #calculate energy like this
    thetaenergy = sum([x*2 for x in np.matrix.transpose(thetacomp)])
    thetaamp = np.mean(np.abs(thetacomp),axis=1)
                 
       
    theta1comp = butter_bandpass_filter(s, 6, 9, fs, 5)
    #calculate energy like this
    theta1energy = sum([x*2 for x in np.matrix.transpose(theta1comp)])
    theta1amp = np.mean(np.abs(theta1comp),axis=1)  
    
    theta2comp = butter_bandpass_filter(s, 5.5, 8.5, fs, 5)
    #calculate energy like this
    theta2energy = sum([x*2 for x in np.matrix.transpose(theta2comp)])
    theta2amp = np.mean(np.abs(theta2comp),axis=1)
                 
    theta3comp = butter_bandpass_filter(s, 7, 10, fs, 5)
    #calculate energy like this
    theta3energy = sum([x*2 for x in np.matrix.transpose(theta3comp)])
    theta3amp = np.mean(np.abs(theta3comp),axis=1)
                 
    betacomp = butter_bandpass_filter(s, 20, 40, fs, 5)
    #calculate energy like this
    betaenergy = sum([x*2 for x in np.matrix.transpose(betacomp)])
    betaamp = np.mean(np.abs(betacomp),axis=1)
    
    alfacomp = butter_bandpass_filter(s, 8, 13, fs, 5)
    #calculate energy like this
    
    alfaenergy = sum([x*2 for x in np.matrix.transpose(alfacomp)])
    
    alfaamp = np.mean(np.abs(alfacomp),axis=1)
                 
    sigmacomp = butter_bandpass_filter(s, 11, 15, fs, 5)
    #calculate energy like this
    sigmaenergy = sum([x*2 for x in np.matrix.transpose(sigmacomp)])
    sigmaamp = np.mean(np.abs(sigmacomp),axis=1)
                 
    spindlecomp = butter_bandpass_filter(s, 12, 14, fs, 5)
    #calculate energy like this
    spindleenergy = sum([x*2 for x in np.matrix.transpose(spindlecomp)])
    spindleamp = np.mean(np.abs(spindlecomp),axis=1)
    
    gammacomp = butter_bandpass_filter(s, 35, 45, fs, 5)
    #calculate energy like this
    gammaenergy = sum([x*2 for x in np.matrix.transpose(gammacomp)])
    gammaamp = np.mean(np.abs(gammacomp),axis=1)
       
    ## Calculate the spectral mean and the spectral entropy (essentially the spectral power distribution):
    spectral_mean = np.mean(std_power,axis=1)
    spectral_entropy = -(np.sum((std_power+0.01)*np.log(std_power+0.01),axis=1))/(np.log(len(std_power[0,:])))
    
     
    ## Create a matrix of all of the features per each epoch of the signal
    corr_signal = signal[:len(signal)-(len(signal)%epoch)]
    epochs = np.arange(len(corr_signal)/epoch)+1
    
    if (is_emg == False):
        feature_matrix = np.column_stack((epochs,delta,deltaenergy,deltaamp, thetacon, thetaenergy, thetaamp, theta1, theta1energy,
                                          theta1amp, theta2, theta2energy, theta2amp, theta3, theta3energy, theta3amp, beta, 
                                          betaenergy, betaamp, alpha, alfaenergy, alfaamp, sigma, sigmaenergy, sigmaamp,
                                          spindle, spindleenergy, spindleamp, gamma, gammaenergy, gammaamp, EEGrel1, EEGrel2, 
                                          spindelhan, spectral_edge, spectral_mean50, zerocross, maxs, peaktopeak, arv,
                                          rms, amplitude, amplitude_m, signal_var, skew, kurt, spectral_mean, spectral_entropy))
                 
        features = (['epochs','delta','deltaenergy','deltaamp','thetacon','thetaenergy','thetaamp', 'theta1','theta1energy',
                     'theta1amp','theta2', 'theta2energy','theta2amp', 'theta3', 'theta3energy','theta3amp', 'beta', 
                     'betaenergy','betaamp','alpha', 'alfaenergy', 'alfaamp', 'sigma', 'sigmaenergy', 'sigmaamp', 
                     'spindle', 'spindlenergy', 'spindleamp', 'gamma', 'gammaenergy', 'gammaamp', 'EEGrel1', 'EEGrel2', 
                     'spindelhan', 'spectral_edge', 'spectral_mean50', 'zerocross', 'maxs' , 'peaktopeak', 'arv',
                     'rms', 'amplitude', 'amplitude_m', 'signal_var', 'skew', 'kurt', 'spectral_mean', 'spectral_entropy'])
    else:
        feature_matrix = np.column_stack((epochs,amplitude,signal_var,skew,kurt,rms,
                                     spectral_mean,spectral_entropy,amplitude_m))
        
        features = (['epochs','amplitude','signal_var','skew',
                          'kurt','rms','spectral_mean','spectral_entropy','amplitude_m'])
    feature_labels = []
    
    for i in range(len(features)):
        feature_labels.append('%s_%s' % (signal_label,features[i]))
    return feature_matrix,feature_labels




#Dialogs 
class LoadDialogFeat(FloatLayout):
    load_Feat = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialogFeat(FloatLayout):
    save_Feat = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

#Main code
class Root(FloatLayout):
    loadfileFeat = ObjectProperty(None)
    savefileFeat = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load_Feat(self):
        content = LoadDialogFeat(load_Feat=self.load_Feat, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save_Feat(self):
        content = SaveDialog(save_Feat=self.save_Feat, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
		
#
    def load_Feat(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            global data_frame
            print (filename[0])
            print (stream)
            eeg1,eeg2,emg,time_stamps = readEDFfile(filename[0])
            print("EDF read")
            print (eeg1)
            data_frame = CreateFeaturesDataFrame(eeg1,emg)
            data_frame['time_stamps'] = time_stamps         
			
#Show in screen
            
            self.text_input.text =  ('Features computed succesfully')

             
            
            
        self.dismiss_popup()

    def save_Feat(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:

            data_frame.to_csv(stream)
            
            
            #Saves training
            
        self.dismiss_popup()


class SiestaFeat(App):
    pass


Factory.register('Root', cls=Root)
Factory.register('LoadDialogFeat', cls=LoadDialogFeat)
Factory.register('SaveDialogFeat', cls=SaveDialogFeat)


if __name__ == '__main__':
    SiestaFeat().run()
