"""This module defines a pydra API for the input output tasks."""
import pydra

from senselab.utils.tasks.input_output import push_dataset_to_hub, read_dataset_from_disk, read_dataset_from_hub, read_files_from_disk, save_dataset_to_disk

read_files_from_disk_pt = pydra.mark.task(read_files_from_disk)
read_dataset_from_disk_pt = pydra.mark.task(read_dataset_from_disk)
read_dataset_from_hub_pt = pydra.mark.task(read_dataset_from_hub)
push_dataset_to_hub_pt = pydra.mark.task(push_dataset_to_hub)
save_dataset_to_disk_pt = pydra.mark.task(save_dataset_to_disk)