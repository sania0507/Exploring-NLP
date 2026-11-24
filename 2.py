import whisper
# Imports OpenAI's Whisper library.
# Whisper is a pre-trained Automatic Speech Recognition (ASR) model.
# ASR means converting spoken language/audio into written text.


from deep_translator import GoogleTranslator
# Imports GoogleTranslator for machine translation.
# It will translate the Hindi text produced by Whisper into English.


model = whisper.load_model("base")
# Loads the pre-trained Whisper "base" model.
# Whisper is a neural-network-based ASR model trained on a large amount
# of multilingual and multitask speech data.
# It takes audio as input and predicts the words spoken in that audio.
# "base" is the model size; larger models are generally more accurate
# but require more computational resources.


result = model.transcribe(
    "hindi_audio2.m4a",
    language="hi"
)
# Sends the Hindi audio file to Whisper for speech recognition.
# Whisper processes the audio and converts the speech into text.
# "hi" specifies Hindi as the input language.
# The result contains the transcribed text along with other information
# such as detected language and audio segments.


hindi_text = result["text"]
# Extracts only the transcription from Whisper's result.
# The transcription is the Hindi text recognized from the audio.


print("Hindi Transcript:")
print(hindi_text)
# Displays the Hindi speech-to-text output.


english_text = GoogleTranslator(
    source="hi",
    target="en"
).translate(hindi_text)
# Uses machine translation to convert the Hindi transcript into English.
# source="hi" means the input text is Hindi.
# target="en" means the desired output language is English.
# The translated text is stored in english_text.


print("\nEnglish Transcript:")
print(english_text)
# Displays the final English translation.