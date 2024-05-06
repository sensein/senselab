"""This script is used to test the audio tasks."""

from typing import Any, Dict

import pydra

from senselab.audio.tasks.speech_to_text import transcribe_dataset
from senselab.audio.tasks.speech_to_text_pydra import transcribe_dataset_pt
from senselab.utils.decorators import get_response_time
from senselab.utils.tasks.input_output import push_dataset_to_hub, read_files_from_disk
from senselab.utils.tasks.input_output_pydra import push_dataset_to_hub_pt, read_files_from_disk_pt


@get_response_time
def workflow(data: Dict[str, Any]) -> None:
    """This function reads audio files from disk, and transcribes them using Whisper."""
    print("Starting to read files from disk...")
    dataset = read_files_from_disk(data["files"])
    print(f"Dataset loaded with {len(dataset)} records.")

    print("Pushing dataset to the hub...")
    push_dataset_to_hub(dataset, remote_repository="fabiocat/test", split="train")
    print("Dataset pushed to the hub successfully.")

    print("Transcribing dataset...")
    transcript_dataset = transcribe_dataset(dataset, "openai/whisper-tiny")
    print("Transcribed dataset.")

    print("Pushing dataset to the hub...")
    push_dataset_to_hub(transcript_dataset, remote_repository="fabiocat/transcript")
    print("Dataset pushed to the hub successfully.")

@get_response_time
def pydra_workflow(data: Dict[str, Any]) -> None:
    """This function reads audio files from disk, and transcribes them using Whisper."""
    wf0 = pydra.Workflow(name='wf0', input_spec=['x'], x=data['files'])
    wf0.add(read_files_from_disk_pt(name='read_files_from_disk_name', files=wf0.lzin.x))
    wf0.add(push_dataset_to_hub_pt(name='push_audio_dataset_to_hub_name', dataset=wf0.read_files_from_disk_name.lzout.out, remote_repository="fabiocat/test", split="train"))

    wf0.add(transcribe_dataset_pt(name='transcribe_dataset_name', dataset=wf0.read_files_from_disk_name.lzout.out, model_id="openai/whisper-tiny"))
    wf0.add(push_dataset_to_hub_pt(name='push_transcript_dataset_to_hub_name', dataset=wf0.transcribe_dataset_name.lzout.out, remote_repository="fabiocat/transcript"))
    
    wf0.set_output([('out', wf0.read_files_from_disk_name.lzout.out)])

    # PYDRA RUN
    with pydra.Submitter(plugin='serial') as sub:
        sub(wf0)

    _ = wf0.result()
    

data = {"files": ["/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav"], "type": "audio"}

workflow(data)
pydra_workflow(data)


# TODO: 
# CHANGE NAME TO THE PACKAGE
# PUBLISH THE NEW PACKAGE

# CHECK INPUTS AND OUTPUTS!! 
# TODO: SPEECH TO TEXT ON MULTIPLE FILES

# CHECK BETTER LOGIN WITH HF
# SETUP CACHE WITH HF