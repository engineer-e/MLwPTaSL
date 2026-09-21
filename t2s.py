from kokoro import KPipeline
from IPython.display import Audio, display
import soundfile as sf
import numpy as np
import re

class T2S:

    def __init__(self,text,speed=0.9,voice = "af_nicole"):
        self.pipeline = KPipeline(lang_code="a")
        self.text = text
        self.voice = voice 
        self.speed=speed
        self.text = self.align()
        self.generator = self.pipeline(self.text,voice=self.voice,speed=self.speed)
        pass 

    def generate(self,file="test.wav"):
      audio_parts = []
      self.file = file
      for result in self.generator:
         if result.audio is not None:
            audio_parts.append(result.audio.numpy())
      audio = np.concatenate(audio_parts)
      sf.write(file, audio, 24000)

    def display(self):
       display(Audio(self.file, autoplay=True))
    

    def display_(self,file):
           display(Audio(file, autoplay=True))

    def align(self):

      # ---------------------------------------------------------
      # 1. Normalize line endings
      # ---------------------------------------------------------
      
      cleaned_text = self.text.replace("\r\n", "\n").replace("\r", "\n")
      
      
      # ---------------------------------------------------------
      # 2. Fix hyphenated words broken across PDF lines
      #
      # ad-
      # vent
      #
      # becomes:
      #
      # advent
      # ---------------------------------------------------------
      
      cleaned_text = re.sub(
          r"(\w+)-\s*\n\s*(\w+)",
          r"\1\2",
          cleaned_text
      )
      
      
      # ---------------------------------------------------------
      # 3. Split into paragraphs
      # ---------------------------------------------------------
      
      paragraphs = re.split(
          r"\n\s*\n+",
          cleaned_text
      )
      
      
      # ---------------------------------------------------------
      # 4. Clean each paragraph
      # ---------------------------------------------------------
      
      cleaned_paragraphs = []
      
      for paragraph in paragraphs:
      
          # Remove PDF line wrapping inside paragraph
          paragraph = re.sub(
              r"\s*\n\s*",
              " ",
              paragraph
          )
      
          # Normalize spaces
          paragraph = re.sub(
              r"[ \t]+",
              " ",
              paragraph
          )
      
          paragraph = paragraph.strip()
      
          if paragraph:
              cleaned_paragraphs.append(paragraph)
      
      
      # ---------------------------------------------------------
      # 5. Split each paragraph into sentences
      #    and build the final TEXT output
      # ---------------------------------------------------------
      
      output_parts = []
      
      for paragraph in cleaned_paragraphs:
      
          sentences = re.split(
              r"(?<=[.!?])\s+",
              paragraph
          )
      
          sentences = [
              sentence.strip()
              for sentence in sentences
              if sentence.strip()
          ]
      
          # Each sentence gets its own line
          paragraph_text = "\n".join(sentences)
      
          output_parts.append(paragraph_text)
      
      
      # ---------------------------------------------------------
      # 6. Multiple paragraphs separated by a blank line
      # ---------------------------------------------------------
      
      output_text = "\n\n".join(output_parts)
      
      
      # ---------------------------------------------------------
      # 7. Return/use the text
      # ---------------------------------------------------------
      
      print(output_text)
      return output_text
        