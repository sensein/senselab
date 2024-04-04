from pipepal.audio.tasks import IOTask, RawSignalProcessingTask

# Instantiate your class
io_task = IOTask()

io_response = io_task.run({
    "data": {
        "files": [#"/Users/fabiocat/Documents/git/soup/pipepal/data/02___121##0.wav",
                  #"/Users/fabiocat/Documents/git/soup/pipepal/data/03___144##0.wav",
                  #"/Users/fabiocat/Documents/git/soup/pipepal/data/04___80##0.wav",
                  "/Users/fabiocat/Documents/git/soup/pipepal/data/diarization.wav"]
    },
    "service": {
        "service_name": "Datasets"
    }
})

audio_dataset = io_response['output']

print(audio_dataset[-1]['audio']['array'].shape)
print(audio_dataset[-1]['audio']['array'])

rawSignalProcessingTask = RawSignalProcessingTask()

asp_response = rawSignalProcessingTask.run({
    "data": {
        "dataset": audio_dataset, 
        "channeling": {
            "method": "selection", # alternative is "average"
            "channels_to_keep": [0]
        },
        "resampling": {
            "rate": 16000,
        }
    },
    "service": {
        "service_name": "torchaudio"
    }
})
new_audio_dataset = asp_response['output']
print(new_audio_dataset)
print(new_audio_dataset[-1]['audio']['array'].shape)
print(new_audio_dataset[-1]['audio']['array'])