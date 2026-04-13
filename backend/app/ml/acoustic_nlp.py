# app/ml/acoustic_nlp.py
import speech_recognition as sr

class AcousticEngine:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Define the threat vocabulary
        self.critical_keywords = ["fire", "bomb", "gun", "stampede", "help", "bhago", "bachao"]
        self.warning_keywords = ["crowded", "pushing", "stuck", "dhakka"]

    def analyze_audio_chunk(self, audio_file_path=None):
        """
        In a real scenario, this streams from the mic. For the hackathon demo,
        you can pass a pre-recorded audio file of people shouting.
        """
        try:
            if audio_file_path:
                with sr.AudioFile(audio_file_path) as source:
                    audio = self.recognizer.record(source)
            else:
                # Use microphone
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=3)

            # Convert speech to text using Google's free API
            text = self.recognizer.recognize_google(audio).lower()
            print(f"🎤 Overheard: '{text}'")

            # NLP Keyword Sniffing
            if any(word in text for word in self.critical_keywords):
                return 3 # CRITICAL: Immediate panic threat
            elif any(word in text for word in self.warning_keywords):
                return 1 # YELLOW: Friction/complaints
            return 0 # GREEN: Normal chatter

        except sr.UnknownValueError:
            return 0 # Could not understand audio
        except Exception as e:
            print(f"Acoustic Error: {e}")
            return 0