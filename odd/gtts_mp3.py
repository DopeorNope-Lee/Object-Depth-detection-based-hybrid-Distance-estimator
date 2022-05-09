import playsound
from gtts import gTTS


class Warninggtts(object):
    def __init__(self,file):
        self.file=file+".mp3"

    def saving_speaking(self,texting):
        tts=gTTS(text=texting,lang='ko')
        tts.save(self.file)

    def speak(self,filename):
        self.file=filename
        playsound.playsound(self.file)
    
    
    