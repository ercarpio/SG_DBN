import os
import io
import datetime


def get_transcripts():
    # Imports the Google Cloud client library
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types

    # Instantiates a client
    client = speech.SpeechClient()

    # The name of the audio files to transcribe
    for root, subFolders, files in os.walk("../audio_input/"):
        for folder in subFolders:
            print folder
        for file in files:
            print(file)
            # Loads the audio into memory
            # with io.open(file, 'rb') as audio_file:
            #     content = audio_file.read()
            #     audio = types.RecognitionAudio(content=content)
            #
            # config = types.RecognitionConfig(
            #     encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
            #     sample_rate_hertz=16000,
            #     language_code='en-US',
            #     enable_word_time_offsets=True)
                # , speech_contexts=[{"phrases": ["Paul", "Estuardo", "Piyush", "Madison",
                #                                 "Mostafa"]}])

            # Detects speech in the audio file
            # response = client.long_running_recognize(config, audio)
            # response = response.result(timeout=90)
            # print(response)


if __name__ == '__main__':
    get_transcripts()
