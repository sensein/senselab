"""This module defines an abstract service for the video IO task."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class AbstractService(ABC):
    """Abstract base class for video I/O services.

    This class serves as a template for defining services that handle
    the input/output operations related to video files, specifically
    focusing on extracting audio components from video data.
    
    Methods defined here outline the expected interface for such services,
    including the extraction of audio from video. Implementations should
    provide specific logic to handle various video formats and ensure the
    integrity and quality of the extracted audio.

    """
    
    @abstractmethod
    def extract_audios_from_videos(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts the audio track from the video data provided in the input object.

        This method should be implemented by subclasses to extract audio data from
        a video file encapsulated within `input_obj`. The specifics of `input_obj`
        (such as format and contents) should be clearly defined in subclass implementations.

        Parameters:
        input_obj (Dict[str, Any]): A dictionary containing the list of video files
            and metadata necessary for the extraction process. The expected format
            of this dictionary needs to be defined by the concrete subclass.
        
        Returns:
        Dict[str, Any]: A dictionary containing the list of extracted audio data and
            potentially additional metadata pertaining to the audio track.
        
        """
        pass  # implementation of this method is required by subclasses
