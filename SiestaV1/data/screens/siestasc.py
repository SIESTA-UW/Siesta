#A De La iglesia Lab project
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


from sklearn import preprocessing
import os


#Dialogs 
class LoadDialogScore(FloatLayout):
    load_Score = ObjectProperty(None)
    cancel = ObjectProperty(None)

class LoadDialog1Score(FloatLayout):
    load1_Score = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SaveDialogScore(FloatLayout):
    save_Score = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

#Main code
class Root(FloatLayout):

    loadfilescore = ObjectProperty(None)
    loadfile1score = ObjectProperty(None)
    savefilescore = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load_Score(self):
        content = LoadDialogScore(load=self.load_Score, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load database", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_load1_Score(self):
        content = LoadDialog1Score(load1=self.load1_Score, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Mice", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save_Score(self):
        content = SaveDialogScore(save=self.save_Score, cancel=self.dismiss_popup)
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
        with open(filename[0],'rb') as stream:

            global micefeat
            global X
            
            array1 = pd.read_csv(stream);
            array=array1.values;
            X=array[:,2:56]
               
            
          
                        
        self.dismiss_popup()

    def save_Score(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:

            predic= fitML.predict(X)  
            #Saves training
            filename3 = os.path.join(path,filename + '.txt')
            coso = open(filename3,"w")
            np.savetxt(coso,predic,fmt='%s', delimiter=',')
            self.dismiss_popup()


class SiestaScore(App):
    pass


Factory.register('Root', cls=Root)
Factory.register('LoadDialogScore', cls=LoadDialogScore)
Factory.register('LoadDialog1Score', cls=LoadDialog1Score)
Factory.register('SaveDialogScore', cls=SaveDialogScore)


if __name__ == '__main__':
    SiestaScore().run()
