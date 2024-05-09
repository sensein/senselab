"""This script is used to test the audio tasks."""

from typing import Any, Dict

import pydra

from senselab.audio.tasks.preprocessing import resample_hf_dataset
from senselab.audio.tasks.preprocessing_pydra import resample_hf_dataset_pt
from senselab.audio.tasks.speech_to_text import transcribe_dataset_with_hf
from senselab.audio.tasks.speech_to_text_pydra import transcribe_dataset_with_hf_pt
from senselab.utils.decorators import get_response_time
from senselab.utils.tasks.input_output import read_files_from_disk
from senselab.utils.tasks.input_output_pydra import read_files_from_disk_pt


@get_response_time
def workflow(data: Dict[str, Any]) -> None:
    """This function reads audio files from disk, and transcribes them using Whisper."""
    print("Starting to read files from disk...")
    dataset = read_files_from_disk(data["files"])
    print(f"Dataset loaded with {len(dataset)} records.")

    print("Resampling dataset...")
    dataset = resample_hf_dataset(dataset, 16000)
    print("Resampled dataset.")

    #print("Pushing dataset to the hub...")
    #push_dataset_to_hub(dataset, remote_repository="fabiocat/test", split="train")
    #print("Dataset pushed to the hub successfully.")

    print("Transcribing dataset...")
    _ = transcribe_dataset_with_hf(dataset=dataset, model_id="openai/whisper-tiny", language="en") # facebook/wav2vec2-base-960h
    print("Transcribed dataset.")

    #print("Pushing dataset to the hub...")
    #push_dataset_to_hub(transcript_dataset, remote_repository="fabiocat/transcript")
    #print("Dataset pushed to the hub successfully.")

@get_response_time
def pydra_workflow(data: Dict[str, Any]) -> None:
    """This function reads audio files from disk, and transcribes them using Whisper."""
    wf0 = pydra.Workflow(name='wf0', input_spec=['x'], x=data['files'])
    wf0.add(read_files_from_disk_pt(name='read_files_from_disk_name', files=wf0.lzin.x))
    wf0.add(resample_hf_dataset_pt(name='resample_hf_dataset_name', dataset=wf0.read_files_from_disk_name.lzout.out, resample_rate=16000))
    #wf0.add(push_dataset_to_hub_pt(name='push_audio_dataset_to_hub_name', dataset=wf0.resample_hf_dataset_name.lzout.out, remote_repository="fabiocat/test", split="train"))

    wf0.add(transcribe_dataset_with_hf_pt(name='transcribe_dataset_name', dataset=wf0.resample_hf_dataset_name.lzout.out, model_id="openai/whisper-tiny", language="en"))
    #wf0.add(push_dataset_to_hub_pt(name='push_transcript_dataset_to_hub_name', dataset=wf0.transcribe_dataset_name.lzout.out, remote_repository="fabiocat/transcript"))
    
    wf0.set_output([('out', wf0.transcribe_dataset_name.lzout.out)])

    # PYDRA RUN
    with pydra.Submitter(plugin='serial') as sub:
        sub(wf0)

    _ = wf0.result()    


@get_response_time
def pydra_workflow2(data: Dict[str, Any]) -> None:
    """This function reads audio files from disk, and transcribes them using Whisper."""
    wf0 = pydra.Workflow(name='wf0', input_spec=['x'], x=data['files'])

    wf0.add(read_files_from_disk_pt(name='read_files_from_disk_name', files=wf0.lzin.x).split('files', files=wf0.lzin.x))
    wf0.add(resample_hf_dataset_pt(name='resample_hf_dataset_name', dataset=wf0.read_files_from_disk_name.lzout.out, resample_rate=16000))
    # wf0.add(push_dataset_to_hub_pt(name='push_audio_dataset_to_hub_name', dataset=wf0.resample_hf_dataset_name.lzout.out, remote_repository="fabiocat/test", split="train"))

    wf0.add(transcribe_dataset_with_hf_pt(name='transcribe_dataset_name', dataset=wf0.resample_hf_dataset_name.lzout.out, model_id="openai/whisper-tiny", language="en"))
    wf0.combine('x')

    # wf0.add(push_dataset_to_hub_pt(name='push_transcript_dataset_to_hub_name', dataset=wf0.transcribe_dataset_name.lzout.out, remote_repository="fabiocat/transcript"))
    # TODO: create a dataset object from the combined transcripts
    wf0.set_output([('out', wf0.transcribe_dataset_name.lzout.out)])

    # PYDRA RUN
    with pydra.Submitter(plugin='serial') as sub:
        sub(wf0)

    _ = wf0.result()    


data = {"files": 
            ["/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav", 
             "/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
             "/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
             "/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
             "/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
             "/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
             "/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
             "/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav",
             "/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav"
             ]
        }

workflow(data)
print("\n\n")
pydra_workflow(data)
print("\n\n")
pydra_workflow2(data)
