import matplotlib.pyplot as plt
import numpy as np
import torch

from sparc import load_model
from sparc.spk_encoder import SpeakerEncodingLayer, SpeakerEncoder
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

def get_average(spk_embeddings):
    """
    Finds the average of the speaker embeddings by taking the mean value for each component of the array.
    spk_embeddings (array) = the array of lists of speaker embeddings
    retuns the average speaker embedding (array)
    """
    #decide the length of embeddings_array
    max = len(spk_embeddings[0])
    for embedding in spk_embeddings:
        if len(embedding) > max:
            max = len(embedding)

    embeddings_array = [0 for i in range(max)]

    #fill the array with sums
    for embedding in spk_embeddings:
        for i in range(len(embedding)):
            embeddings_array[i] += embedding[i]

    #take the average
    for index in range(len(embeddings_array)):
        embeddings_array[index] /= len(spk_embeddings)

    return np.array(embeddings_array)

def anonymize_audio(source_audios, target_audio):
    """
    Changes the pitch of the target audio to a specific pitch from the source audio,
    and changes the speaker embeddings of the target audio to the average speaker embeddings
    of the source audios.
    source_audios (list): a list of source audio path names
    target_audio (audio): the target audio path name

    Returns:
    modified_audio: the anonymized audio
    """

    coder = load_model("en", device= "cpu")    # Use PENN for pitch tracker

    #load the audios
    # audio_files = ['sample1.wav', 'sample2.wav'] #fill this with list of audio file pathnames
    spk_embeddings = []

    #getting speaker embeddings + pitch values
    pitch_values = 0

    for audio in source_audios:
        code = coder.encode(audio)
        if audio == audio_files[0]:
            pitch_values = code['pitch_stats']
        spk_embeddings.append(code['spk_emb'])

        #final step: changing an audio to have the avg speaker embedding + set pitch
        audio_to_anonymize = coder.encode(target_audio)  # Choose the audio to anonymize
        audio_to_anonymize['pitch_stats'] = pitch_values
        audio_to_anonymize['spk_emb'] = get_average(spk_embeddings)
        anonymized_audio = coder.decode(**audio_to_anonymize)

    #save the modified audio
    return anonymized_audio

#for testing: sf.write("output.wav", anonymized_audio, coder.sr)
