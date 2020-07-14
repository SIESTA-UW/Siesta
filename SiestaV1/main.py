#A De La iglesia Lab project
#Sleep Identification Enabled by Supervised Training Algorithms
#Codename: SIESTA

from time import time
from kivy.app import App
from os.path import dirname, join
from kivy.lang import Builder
from kivy.properties import NumericProperty, StringProperty, BooleanProperty,\
    ListProperty
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.uix.screenmanager import Screen

from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout

#Load Panda for data manage
import pandas as pd

#Load pickle to save fit/training
import pickle

import numpy as np

#Load SK learn for models and scores
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

#Load the model to use
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier


from sklearn import preprocessing
import os
import glob
import pyedflib 
import datetime
import itertools
import scipy 
import scipy.signal
from scipy.signal import butter, lfilter

from math import sin
from kivy.garden.graph import Graph, MeshLinePlot


from kivy.config import Config
Config.set('kivy','window_icon','sleep.png')
#from kivy.core.window import Window
#Window.size = (800, 600)
# to change the kivy default settings 
# we use this module config 

# 0 being off 1 being on as in true / false 
# you can use 0 or 1 && True or False 
Config.set('graphics', 'resizable', '0') 
# CHECK THIS PART FOR THE WINDOW SIZE  
# fix the width of the window  
Config.set('graphics', 'width', '800') 
  
# fix the height of the window  
Config.set('graphics', 'height', '600') 

epoch=10
fs=400
EEGC=2
EMGC=0
##

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

def CreateFeaturesDataFrame(eeg,emg,epoch,fs):
    eeg_features,eeg_feature_labels = extract_features(eeg,'EEG',epoch=epoch,fs=fs,is_emg=False)
    emg_features,emg_feature_labels = extract_features(emg,'EMG',epoch=epoch,fs=fs,is_emg=True)
    feature_matrix = np.column_stack((eeg_features,emg_features[:,1:]))
    feature_labels = ['Epoch'] + eeg_feature_labels[1:] + emg_feature_labels[1:]
    data = pd.DataFrame(feature_matrix,columns=feature_labels)
    final_data = pd.DataFrame(data.iloc[:,1:])
    return final_data

def readEDFfile(file_name,EEGC,EMGC,epochS):

    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    eeg1 = f.readSignal(EEGC-1)
    eeg2 = f.readSignal(1)
    emg = f.readSignal(EMGC-1)
    
    start_date = datetime.datetime(f.getStartdatetime().year,f.getStartdatetime().month,f.getStartdatetime().day,
                                   f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second)
    fd = f.getFileDuration()
    file_end = start_date + datetime.timedelta(seconds = fd)

    step = datetime.timedelta(seconds=epochS)  

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
    
##### 
class LoadDialogFit(FloatLayout):
    load_fit = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SaveDialogFit(FloatLayout):
    save_fit = ObjectProperty(None)
    text_input_1 = ObjectProperty(None)
    cancel = ObjectProperty(None)

#######
class LoadDialogScore(FloatLayout):
    load_Score = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class LoadDialog1Score(FloatLayout):
    load1_Score = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SaveDialogScore(FloatLayout):
    save_Score = ObjectProperty(None)
    text_input_1 = ObjectProperty(None)
    cancel = ObjectProperty(None)
##Setting code
class LoadDialogUser(FloatLayout):
    load_User = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SaveDialogUser(FloatLayout):
    save_User = ObjectProperty(None)
    text_input_1 = ObjectProperty(None)
    cancel = ObjectProperty(None)


		 
#Main Screen Code


class ShowcaseScreen(Screen):
    fullscreen = BooleanProperty(False)

    os.chdir('C:/Users/Charly/AppData/Local/Programs/Python/Python37/APPS/MLC/SiestaV1')
    freq = ObjectProperty(None)
    eegc = ObjectProperty(None)
    emgc = ObjectProperty(None)
    epocha = ObjectProperty(None)
    
    
    def submit(self):
        global fs
        fs =self.freq.text
        fs=int(fs)
        
        global EEGC

        EEGC = self.eegc.text
        EEGC=int(EEGC)
        
        global EMGC
        EMGC = self.emgc.text
        EMGC=int(EMGC)

        global epoch
        epoch = self.epocha.text
        epoch=int(epoch)


    def add_widget(self, *args):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args)
        return super(ShowcaseScreen, self).add_widget(*args)
    
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    
    text_input = ObjectProperty(None)
    text_input_1 = ObjectProperty(None)
    
    loadfilescore = ObjectProperty(None)
    loadfile1score = ObjectProperty(None)
    savefilescore = ObjectProperty(None)

    loadfileFeat = ObjectProperty(None)
    savefileFeat = ObjectProperty(None)

    loadfileUser = ObjectProperty(None)
    savefileUser = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

####
        ####
        #Training CODE

    def show_load_Fit(self):
        content = LoadDialogFit(load_fit=self.load_fit, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save_Fit(self):
        content = SaveDialogFit(save_fit=self.save_fit, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
################
#User database

    def show_load_User(self):
        content = LoadDialogUser(load_User=self.load_User, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load folder", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save_User(self):
        content = SaveDialogUser(save_User=self.save_User, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

        
##Load database User##

    def load_User(self,path, filename):
        os.chdir(path) #Change folder
        global textoDir
        textoDir=os.getcwd()
        print ('%s',textoDir)
        self.text_input.text =  (textoDir)
        self.dismiss_popup()
        
    def save_User(self,path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            feature_files = []
            for root, dirs, files in os.walk(textoDir):
                feature_files += glob.glob(os.path.join(root, '*Features.csv')) 
    
            score_files = []
            for root, dirs, files in os.walk(textoDir):
                score_files += glob.glob(os.path.join(root, '*_scores.csv')) 
            textDir=os.getcwd()
            print ('%s',textDir)
#   
            for i in range(len(feature_files)):
    
                data = pd.read_csv(feature_files[i])
                score = pd.read_csv(score_files[i],sep='\t',header=9)
                manscore = score['Time from Start']
                Y=manscore
                Y2=Y
                index = 0;
                Y2=Y2.replace(1, 'AWAKE')
                Y2=Y2.replace(2, 'NREM')
                Y2=Y2.replace(3, 'REM')

                
                manscore = Y2
                data=data.assign(score=manscore.array)  
            
                file_suffix = '_all.csv'
                data.to_csv(feature_files[i] + file_suffix)
                
                
            
            all_files = []
            for root, dirs, files in os.walk(textoDir):
                all_files += glob.glob(os.path.join(root, '*_all.csv')) 
 
            data1 = pd.read_csv(all_files[0])

            for i in range(1,len(all_files)):
                data=pd.read_csv(all_files[i])
                data1 = data1.append(data)
            
             
            data1 = data1.drop(columns=['Unnamed: 0'])
        
            filename3 = os.path.join(path,filename  + '.csv')
            self.text_input_1.text =  ('''Database created succesfully
File saved in: %s'''%(filename3))
            data1.to_csv(filename3)
            #Saves training
            
        self.dismiss_popup()
	
##############################	
#Train when database is loaded
    def load_fit(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            
            array1 = pd.read_csv(stream);
            arrayT=array1.values;
            X=arrayT[:,2:56]
            Y=arrayT[:,58]

            print('Datos Cargadas C')

            #split5=len(array1)
            #split50=split5/2
            #split50=int(split50)
            
            #X_train = XT[0:split50]
            #X_validation= XT[split50:]
            #Y_train=YT[0:split50]
            #Y_validation=YT[split50:]
            seed = 7
            validation_size = 0.5
            
            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

            
            print('Info Dividida')
            global bc
            #bc = GradientBoostingClassifier()
            bc = BaggingClassifier(RandomForestClassifier())
            #bc=RandomForestClassifier()
            print('Model Loaded GBC')
            bc.fit(X_train, Y_train)
            print('Fiting of training data Exitosa')
            predictionsbc = bc.predict(X_validation)
            print('Resultados Boosting Classifier')

            f1 = (f1_score(Y_validation, predictionsbc, average='macro'))
            print('Fiting of training data Exitosa')
            accu = (accuracy_score(Y_validation, predictionsbc))
            print('Fiting of training data Exitosa')
            kappa = (cohen_kappa_score(Y_validation,predictionsbc))
            confuma=(confusion_matrix(Y_validation,predictionsbc))

            clasrep=(classification_report(Y_validation, predictionsbc))
	    	
#Show in screen
            largo=len(Y)
            dias=largo/8640
            self.text_input.text =  ('''Database Scores:		
F1 = %f 
Accuracy=%f 
Cohen-kappa=%f 
#epoch=%d 
#days=%f 
Confusion Max=

       Awake NREM  REM 
Awake  %d    %d    %d 
NREM   %d    %d    %d 
REM    %d    %d    %d 

Classification Report=

%s

'''%(f1, accu , kappa,largo,dias,confuma[0,0],confuma[0,1],confuma[0,2],confuma[1,0],confuma[1,1],confuma[1,2],confuma[2,0],confuma[2,1],confuma[2,2],clasrep))

             
            bc.fit(X,Y)
            
            
        self.dismiss_popup()

    def save_fit(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            texto =(self.text_input.text)
            stream.write(str(texto))
            #Saves training
            filename3 = os.path.join(path,filename + '.sav')
            pickle.dump(bc, open(filename3, 'wb'))
            #Show in screen
            texto = str(bc)
            texto1 = str(bc[1])

            self.text_input_1.text = ('''Database was saved as: \n
%s. \n Meta Classiffier: %s.
\n Base Clasiffier: %s. '''%(filename3,texto.split('(')[0],texto1.split('(')[0]))
        self.dismiss_popup()

######
        #####
        #SCORING CODE

    def show_load_Score(self):
        content = LoadDialogScore(load_Score=self.load_Score, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load database", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_load1_Score(self):
        content = LoadDialog1Score(load1_Score=self.load1_Score, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Mice", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save_Score(self):
        content = SaveDialogScore(save_Score=self.save_Score, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
		
#LOAD DB
    def load_Score(self, path, filename):
        with open(filename[0],'rb') as stream:

            global fitML
            
            fitML = pickle.load(stream)
                    
            pre, ext = os.path.splitext(filename[0])
            
            file1 = open(pre,"r")
            texto =  file1.read()
            
            self.text_input.text =  (texto)
             
                        
        self.dismiss_popup()
        
    def load1_Score(self, path, filename):
        with open(filename[0],'r') as stream:

            global micefeat
            global X
            
            
            array1 = pd.read_csv(stream);
            array=array1.values;
            X=array[:,2:56]
            plot = MeshLinePlot(color=[1, 0, 0, 1])
            X_scaled = preprocessing.scale(X[:,0])
            plot.points = [(y,X_scaled[y]) for y in range (0,500)]
            self.graph_test.add_plot(plot)   

            global score_feat
          
            score_fe = array1.loc[:,'time_stamps']
            score_feat = pd.DataFrame(score_fe)
            score_feat.insert(1,'EEG_spindelhan',array1.loc[:,'EEG_spindelhan'])
            score_feat.insert(2,'EMG_amplitude',array1.loc[:,'EMG_amplitude'])
            score_feat.insert(3,'EEG_delta',array1.loc[:,'EEG_delta'])
            score_feat.insert(4,'EMG_spectral_entropy',array1.loc[:,'EMG_spectral_entropy'])
            score_feat.insert(5,'EEG_theta1',array1.loc[:,'EEG_theta1'])
            score_feat.insert(6,'EEG_zerocross',array1.loc[:,'EEG_zerocross'])
            
        self.dismiss_popup()

    def save_Score(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:

            predic= fitML.predict(X)  
            #Saves training

            
            score_feat.insert(7,'Score',predic)  
            
            filename3 = os.path.join(path,filename  + '.csv')
            self.text_input_1.text =  ('''Score succesfully finish.
File saved in: %s'''%(filename3))

            score_feat.to_csv(filename3)
            #Saves training

            
        self.dismiss_popup()
#####
            #####
            #Features code
            
    def show_load_Feat(self):
        content = LoadDialogFeat(load_Feat=self.load_Feat, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
		
        self._popup.open()

    def show_save_Feat(self):
        content = SaveDialogFeat(save_Feat=self.save_Feat, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
		
#
    def load_Feat(self, path, filename):
        with open(os.path.join(path, filename[0]),'r') as stream:
		
            global data_frame
            print (filename[0])
            print (stream)
            eeg1,eeg2,emg,time_stamps = readEDFfile(filename[0],EMGC,EEGC,epoch)
            print("EDF read")
            print (eeg1)
            print(emg)
            data_frame = CreateFeaturesDataFrame(eeg1,emg,epoch,fs)
            data_frame['time_stamps'] = time_stamps         
            plot = MeshLinePlot(color=[1, 0, 0, 1])
            plot.points = [(x, eeg1[x]) for x in range (0,1500)]
            self.graph_test.add_plot(plot)
             
            
            
        self.dismiss_popup()

    def save_Feat(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:

            filename3 = os.path.join(path,filename  + '.csv')
            self.text_input.text =  ('''Features computed succesfully
File saved in: %s'''%(filename3))
            data_frame.to_csv(filename3)
            #Saves training
            
        self.dismiss_popup()


###APP CLASEE MAIN CODE
class ShowcaseApp(App):

    def build(self):
        return MyGrid()

    index = NumericProperty(-1)
    current_title = StringProperty()
    time = NumericProperty(0)
    show_sourcecode = BooleanProperty(False)
    sourcecode = StringProperty()
    screen_names = ListProperty([])
    hierarchy = ListProperty([])

    def build(self):
        self.title = "Siesta V 1.0"
        Clock.schedule_interval(self._update_clock, 1 / 60.)
        self.screens = {}
        self.available_screens = sorted([
            'SiestaFit', 'SiestaFeat', 'SiestaScore','SiestaConfig','SiestaUser'])
        self.screen_names = self.available_screens
        curdir = dirname(__file__)
        self.available_screens = [join(curdir, 'data', 'screens',
            '{}.kv'.format(fn).lower()) for fn in self.available_screens]
        self.go_next_screen()

    def on_pause(self):
        return True

    def on_resume(self):
        pass

    def on_current_title(self, instance, value):
        self.root.ids.spnr.text = value

    def go_previous_screen(self):
        self.index = (self.index - 1) % len(self.available_screens)
        screen = self.load_screen(self.index)
        sm = self.root.ids.sm
        sm.switch_to(screen, direction='right')
        self.current_title = screen.name
        self.update_sourcecode()

    def go_next_screen(self):
        self.index = (self.index + 1) % len(self.available_screens)
        screen = self.load_screen(self.index)
        sm = self.root.ids.sm
        sm.switch_to(screen, direction='left')
        self.current_title = screen.name
        self.update_sourcecode()

    def go_screen(self, idx):
        self.index = idx
        self.root.ids.sm.switch_to(self.load_screen(idx), direction='left')
        self.update_sourcecode()

    def go_hierarchy_previous(self):
        ahr = self.hierarchy
        if len(ahr) == 1:
            return
        if ahr:
            ahr.pop()
        if ahr:
            idx = ahr.pop()
            self.go_screen(idx)

    def load_screen(self, index):
        if index in self.screens:
            return self.screens[index]
        screen = Builder.load_file(self.available_screens[index])
        self.screens[index] = screen
        return screen

    def read_sourcecode(self):
        fn = self.available_screens[self.index]
        with open(fn) as fd:
            return fd.read()

    def toggle_source_code(self):
        self.show_sourcecode = not self.show_sourcecode
        if self.show_sourcecode:
            height = self.root.height * .3
        else:
            height = 0

        Animation(height=height, d=.3, t='out_quart').start(
                self.root.ids.sv)

        self.update_sourcecode()

    def update_sourcecode(self):
        if not self.show_sourcecode:
            self.root.ids.sourcecode.focus = False
            return
        self.root.ids.sourcecode.text = self.read_sourcecode()
        self.root.ids.sv.scroll_y = 1

    def showcase_floatlayout(self, layout):

        def add_button(*t):
            if not layout.get_parent_window():
                return
            if len(layout.children) > 5:
                layout.clear_widgets()
            layout.add_widget(Builder.load_string('''
#:import random random.random
Button:
    size_hint: random(), random()
    pos_hint: {'x': random(), 'y': random()}
    text:
        'size_hint x: {} y: {}\\n pos_hint x: {} y: {}'.format(\
            self.size_hint_x, self.size_hint_y, self.pos_hint['x'],\
            self.pos_hint['y'])
'''))
            Clock.schedule_once(add_button, 1)
        Clock.schedule_once(add_button)

    def showcase_boxlayout(self, layout):

        def add_button(*t):
            if not layout.get_parent_window():
                return
            if len(layout.children) > 5:
                layout.orientation = 'vertical'\
                    if layout.orientation == 'horizontal' else 'horizontal'
                layout.clear_widgets()
            layout.add_widget(Builder.load_string('''
Button:
    text: self.parent.orientation if self.parent else ''
'''))
            Clock.schedule_once(add_button, 1)
        Clock.schedule_once(add_button)

    def showcase_gridlayout(self, layout):

        def add_button(*t):
            if not layout.get_parent_window():
                return
            if len(layout.children) > 15:
                layout.rows = 3 if layout.rows is None else None
                layout.cols = None if layout.rows == 3 else 3
                layout.clear_widgets()
            layout.add_widget(Builder.load_string('''
Button:
    text:
        'rows: {}\\ncols: {}'.format(self.parent.rows, self.parent.cols)\
        if self.parent else ''
'''))
            Clock.schedule_once(add_button, 1)
        Clock.schedule_once(add_button)

    def showcase_stacklayout(self, layout):
        orientations = ('lr-tb', 'tb-lr',
                        'rl-tb', 'tb-rl',
                        'lr-bt', 'bt-lr',
                        'rl-bt', 'bt-rl')

        def add_button(*t):
            if not layout.get_parent_window():
                return
            if len(layout.children) > 11:
                layout.clear_widgets()
                cur_orientation = orientations.index(layout.orientation)
                layout.orientation = orientations[cur_orientation - 1]
            layout.add_widget(Builder.load_string('''
Button:
    text: self.parent.orientation if self.parent else ''
    size_hint: .2, .2
'''))
            Clock.schedule_once(add_button, 1)
        Clock.schedule_once(add_button)

    def showcase_anchorlayout(self, layout):

        def change_anchor(self, *l):
            if not layout.get_parent_window():
                return
            anchor_x = ('left', 'center', 'right')
            anchor_y = ('top', 'center', 'bottom')
            if layout.anchor_x == 'left':
                layout.anchor_y = anchor_y[anchor_y.index(layout.anchor_y) - 1]
            layout.anchor_x = anchor_x[anchor_x.index(layout.anchor_x) - 1]

            Clock.schedule_once(change_anchor, 1)
        Clock.schedule_once(change_anchor, 1)

    def _update_clock(self, dt):
        self.time = time()
		
    
		

Factory.register('ShowcaseApp', cls=ShowcaseApp)
Factory.register('LoadDialogFit', cls=LoadDialogFit)
Factory.register('SaveDialogFit', cls=SaveDialogFit)
Factory.register('LoadDialogScore', cls=LoadDialogScore)
Factory.register('LoadDialog1Score', cls=LoadDialog1Score)
Factory.register('SaveDialogScore', cls=SaveDialogScore)
Factory.register('LoadDialogFeat', cls=LoadDialogFeat)
Factory.register('SaveDialogFeat', cls=SaveDialogFeat)
Factory.register('LoadDialogUser', cls=LoadDialogUser)
Factory.register('SaveDialogUSer', cls=SaveDialogUser)



if __name__ == '__main__':
    ShowcaseApp().run()
