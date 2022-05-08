import playsound
from gtts import gTTS




class Warninggtts(object):
    def __init__(self,filename):
        self.file=filename

    def saving_speaking(self,texting):
        tts=gTTS(text=texting,lang='ko')
        tts.save(self.file)

    def speak(self):
        playsound.playsound(self.file)
    
    
    