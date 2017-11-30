import os
import io


def get_transcripts():
    # Imports the Google Cloud client library
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types

    # Instantiates a client
    client = speech.SpeechClient()

    # Transcribe audio files
    responses = dict()
    for root, subFolders, files in os.walk("../audio_input/"):
        for file_name in files:
            file_path = root + "/" + file_name
            print(file_path)
            # Loads the audio into memory
            with io.open(file_path, 'rb') as audio_file:
                content = audio_file.read()
                audio = types.RecognitionAudio(content=content)

            config = types.RecognitionConfig(
                encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
                sample_rate_hertz=16000,
                language_code='en-US',
                enable_word_time_offsets=True,
                speech_contexts=[types.SpeechContext(phrases=["paul", "estuardo", "piyush",
                                                              "madison", "mostafa", "momotaz",
                                                              "katie", "goodbye", "hello", "say",
                                                              "great", "job", "good", "bye",
                                                              "say hello", "great job",
                                                              "great job good bye",
                                                              "great job goodbye"])])

            # Detects speech in the audio file
            responses[file_path] = client.long_running_recognize(config, audio)

    for file_name, future in responses.items():
        out_name = file_name.replace("input", "output")
        response = future.result(timeout=300)
        with open(out_name, "w") as out_file:
            out_file.write(file_name + "\n" + str(response))
            print(out_name)


if __name__ == '__main__':
    get_transcripts()
