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
class LoadDialogFit(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialogFit(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

#Main code
class Root(FloatLayout):

    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load_Fit(self):
        content = LoadDialogFit(load=self.loadFit, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save_Fit(self):
        content = SaveDialogFit(save=self.saveDit, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
		
#Train when database is loaded
    def load_fit(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            
            array1 = pd.read_csv(stream);
            arrayT=array1.values;
            XT=arrayT[:,2:56]
            YT=arrayT[:,58]

            print('Datos Cargadas C')

            split5=len(array1)
            split50=split5/2
            split50=int(split50)
            
            X_train = XT[0:split50]
            X_validation= XT[split50+1:]
            Y_train=YT[0:split50]
            Y_validation=YT[split50+1:]

            
            print('Info Dividida')
            global bc
            bc = GradientBoostingClassifier()
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
            print(confusion_matrix(Y_validation,predictionsbc))
			
#Show in screen
            
            self.text_input.text =  ('''Database Scores:
			
F1 = %f
Accuracy=%f
Cohen-kappa=%f ''' %(f1, accu , kappa))

             
            
            
        self.dismiss_popup()

    def save_fit(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            stream.write(self.text_input.text)
            #Saves training
            filename3 = os.path.join(path,filename + '.sav')
            pickle.dump(bc, open(filename3, 'wb'))
        self.dismiss_popup()


class SiestaFit(App):
    pass


Factory.register('ShowcaseApp', cls=Root)
Factory.register('LoadDialogFit', cls=LoadDialogFit)
Factory.register('SaveDialogFit', cls=SaveDialogFit)


if __name__ == '__main__':
    SiestaFit().run()
