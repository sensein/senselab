# Functionalities

VIDEO
- example
    1. to showcase how a generic video module works.
        - INPUT: 
            1. a list of video file paths (.mp4, .avi)
            2. a service with a name
        - PROCESSING: nothing.
        - OUTPUT: a json object placeholder as a response

- input/output:
    1. to extract the audio (all channels) from the given videos. 
        - INPUT: 
            1. a list of video file paths (.mp4, .avi)
            2. a service with a name for extracting the audio from a video
        - PREPROCESSING:
            1. check files exist
            2. check their format is ok
        - PROCESSING: 
            1. extract audio channels from video
            2. embed those into an HF dataset object 
        - OUTPUT: 
            1. a HF datasets object with the audio recordings in the "audio" column

AUDIO
- example
    1. to showcase how the module works.
        - INPUT: 
            1. a list of audio file paths (.mp3, .wav)
            2. a service with a name
        - PROCESSING: nothing.
        - OUTPUT: a json object placeholder as a response

- input/output
    1. to read some audio recordings from disk. 
        - INPUT: 
            1. a list of audio file paths (.mp3, .wav)
            2. a service with a name (the default is "soundfile")
        - PREPROCESSING:
            1. check files exist
            2. check their format is ok
        - PROCESSING: 
            1. extract audio channels from video
            2. embed the extracted audio channels into an HF dataset object
        - OUTPUT: a datasets object with the audio recordings in the "audio" column

    2. to save HF dataset object to disk
        - INPUT:
            1. a datasets object with the audio recordings, 
            2. the output path
        - PREPROCESSING:
            1. check if the dataset exists already (TODO: decide how to manage this scenario)
            2. create all folders to path if they don't exist
        - PROCESSING:
            1. save HF dataset to disk
        - OUTPUT: -

    3. to upload HF dataset object to the remote platform
        - INPUT: 
            1. a datasets object with the audio recordings
            2. the remote ID (and maybe the version?)
        - PROCESSING:
            1. upload the dataset to the remote platform
        - OUTPUT: -

- speech to text
    1. to transcribe speech into text
        - INPUT: 
            1. a datasets object with the audio recordings in the "audio" column (optionally including the language spoken in the audio) 
            2. the speech to text service to use (including the name, the version, optionally the revision, and - for some services only and sometimes it's optional - the language of the transcription model we want to use)
        - PREPROCESSING:
            1. adapt the language to the service format
            2. organize the dataset into batches
        - PROCESSING:
            1. transcribe the dataset
        - POSTPROCESSING: 
            1. formatting the transcripts to follow a standard organization
        - OUTPUT:
            1. a new dataset including only the transcripts of the audios in a standardized json format (plus an index?)

    2. to compute word error rate
        - INPUT: 
            1. a dataset object with the "transcript" and the "groundtruth" columns 
            2. a service with a name
        - PROCESSING:
            1. computing the per-row WER between the 2 columns
        - OUTPUT: 
            1. a dataset with the "WER" column

[TODO]
- raw signal processing
- speaker diarization

- data_augmentation 
- data_representation
- speech emotion recognition
- speech enhancement
- speech verification
- text to speech
- voice conversion