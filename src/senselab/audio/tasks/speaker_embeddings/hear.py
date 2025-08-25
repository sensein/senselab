"""
HeAR (Holistic Evaluation of Audio Representations) Embedding Extraction Script

This script extracts embeddings from audio files using the HeAR PyTorch model.
For files longer than 2 seconds, it creates overlapping 2-second windows,
extracts embeddings from each window, and averages them to create a single
representative embedding per file.

Features:
    - Processes full-length audio files (not just first 2 seconds)
    - Creates overlapping 2-second windows with 50% overlap
    - Averages embeddings across all windows for comprehensive representation
    - Uses senselab audio module with librosa fallback
    - Efficient processing with progress tracking and error handling

Usage:
    python b2ai_run_hear_pytorch.py /path/to/audio/files [options]

Output:
    - Individual .npy files containing averaged embeddings per audio file
    - Progress tracking with window count information
    - Error logging for failed files
"""

import os
import sys
import argparse
import importlib
import numpy as np
import pandas as pd
import librosa
import torch
from pathlib import Path
from typing import Set, List, Optional, Tuple
from tqdm import tqdm
from transformers import AutoModel
from huggingface_hub.utils import HfFolder
from huggingface_hub import notebook_login

# Try to import senselab for future use
try:
    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.preprocessing import resample_audios
    SENSELAB_AVAILABLE = True
except ImportError:
    SENSELAB_AVAILABLE = False
    print("Warning: senselab not available, falling back to librosa only")

from utils import get_audio_files_from_directory

# HeAR model constants
SAMPLE_RATE = 16000
CLIP_DURATION = 2
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION  # 32000 samples
EMBEDDING_DIM = 512  # HeAR model pooler_output dimension

# Configuration
AUDIO_DIR = ""  # Set this to your audio directory
OUTPUT_DIR = "output"
BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N files


class HeARProcessor:
    """HeAR embedding processor with PyTorch backend."""

    def __init__(self, device: Optional[str] = None):
        """Initialize the HeAR processor.

        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detection)
        """
        self.device = self._setup_device(device)
        self.model = None
        self.preprocess_audio = None
        self._setup_model()

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup compute device."""
        if device is None:
            # Auto-detect best available device
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        device = torch.device(device)
        print(f"Using device: {device}")
        
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            gpu_props = torch.cuda.get_device_properties(device)
            memory_gb = gpu_props.total_memory / 1e9
            print(f"Memory: {memory_gb:.1f} GB")
        elif device.type == "mps":
            print("Apple Silicon GPU (Metal Performance Shaders)")
            # MPS doesn't have direct memory query, but we can show it's available
            print("MPS backend available for acceleration")

        return device

    def _setup_model(self):
        """Setup HeAR model and preprocessing."""
        print("Setting up HeAR model...")

        # Setup Hugging Face authentication
        if HfFolder.get_token() is None:
            print("Hugging Face token not found. Please login:")
            notebook_login()

        # Add hear module to path
        hear_path = Path(__file__).parent / "hear"
        if hear_path.exists():
            sys.path.append(str(hear_path))

        # Import audio preprocessing
        try:
            module_name = "hear.python.data_processing.audio_utils"
            audio_utils = importlib.import_module(module_name)
            self.preprocess_audio = audio_utils.preprocess_audio
            print("✓ Audio preprocessing loaded")
        except ImportError as e:
            raise ImportError(f"Failed to import HeAR audio utils: {e}")

        # Load HeAR model
        try:
            self.model = AutoModel.from_pretrained("google/hear-pytorch")
            self.model.to(self.device)
            self.model.eval()
            print("✓ HeAR model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load HeAR model: {e}")

    def load_audio_full(self, file_path: str) -> Audio:
        """Load full audio file using senselab with librosa fallback.

        Args:
            file_path: Path to audio file

        Returns:
            Audio object resampled to target sample rate

        Raises:
            RuntimeError: If audio loading fails
        """
        if SENSELAB_AVAILABLE:
            try:
                # Load audio with senselab
                audio_obj = Audio(filepath=file_path)
                
                # Resample to target sample rate if needed
                if audio_obj.sampling_rate != SAMPLE_RATE:
                    [audio_obj] = resample_audios(
                        [audio_obj], resample_rate=SAMPLE_RATE
                    )
                
                return audio_obj
                
            except Exception as e:
                print(f"Senselab failed for {file_path}: {e}")
                print("Falling back to librosa...")
        
        # Fallback to librosa and create Audio object
        audio_array, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        # Normalize to [-1, 1] range
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to torch tensor and create Audio object
        audio_tensor = torch.from_numpy(audio_array).float()
        audio_obj = Audio(
            waveform=audio_tensor.unsqueeze(0),  # Add batch dimension
            sampling_rate=SAMPLE_RATE
        )
        
        return audio_obj

    def create_windows(self, audio_obj: Audio) -> List[Audio]:
        """Create 2-second windows from audio using senselab windowing.

        Args:
            audio_obj: Audio object from senselab

        Returns:
            List of Audio objects representing 2-second windows
        """
        windows = []
        
        # Get audio length in samples
        audio_length = audio_obj.waveform.shape[-1]
        
        # If audio is shorter than 2 seconds, return the original audio
        if audio_length <= CLIP_LENGTH:
            windows.append(audio_obj)
            return windows
        
        # Create overlapping windows using senselab's window_generator
        step_size = CLIP_LENGTH // 2  # 50% overlap
        
        for windowed_audio in audio_obj.window_generator(
            window_size=CLIP_LENGTH, step_size=step_size
        ):
            windows.append(windowed_audio)
        
        return windows

    def extract_embeddings(self, audio_obj: Audio) -> np.ndarray:
        """Extract HeAR embeddings from audio.

        Args:
            audio_obj: Audio object from senselab

        Returns:
            Embedding vector
        """
        try:
            # Get waveform from Audio object
            audio_tensor = audio_obj.waveform
            
            # Ensure tensor is on correct device and has batch dimension
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)

            # Preprocess audio to spectrogram
            spectrogram = self.preprocess_audio(audio_tensor)

            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(
                    spectrogram, return_dict=True
                )
                # Use pooler_output for proper 512d embeddings
                embeddings = outputs.pooler_output

            # Convert to numpy and remove batch dimension
            embeddings = embeddings.cpu().numpy().squeeze()

            return embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to extract embeddings: {e}")

    def process_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Process audio file to extract averaged embeddings from all windows.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (averaged_embedding_vector, number_of_windows)
        """
        # Load full audio file as Audio object
        full_audio = self.load_audio_full(file_path)
        
        # Create 2-second windows using senselab's windowing
        windows = self.create_windows(full_audio)
        
        # Extract embeddings for each window
        all_embeddings = []
        for window in windows:
            embeddings = self.extract_embeddings(window)
            all_embeddings.append(embeddings)
        
        # Average all embeddings
        if len(all_embeddings) == 1:
            averaged_embeddings = all_embeddings[0]
        else:
            averaged_embeddings = np.mean(all_embeddings, axis=0)
        
        return averaged_embeddings, len(windows)


def save_embeddings_numpy(embeddings: np.ndarray, file_stem: str,
                          output_dir: str):
    """Save embeddings as numpy file.

    Args:
        embeddings: Embedding vector
        file_stem: Base filename without extension
        output_dir: Output directory
    """
    numpy_path = Path(output_dir) / f"{file_stem}_embedding.npy"
    np.save(numpy_path, embeddings)


def save_embeddings_csv(embeddings_data: List[dict], output_file: str):
    """Save embeddings data as CSV file.

    Args:
        embeddings_data: List of dictionaries with file and embedding data
        output_file: Output CSV file path
    """
    df = pd.DataFrame(embeddings_data)
    df.to_csv(output_file, index=False)


def load_existing_results(output_file: str) -> Tuple[List[dict], Set[str]]:
    """Load existing results if available.

    Args:
        output_file: Path to existing CSV file

    Returns:
        Tuple of (existing_data, processed_files_set)
    """
    if os.path.exists(output_file):
        print(f"Found existing {output_file}, loading previous results...")
        existing_df = pd.read_csv(output_file)
        existing_data = existing_df.to_dict('records')
        processed_files = set(existing_df['file'].values)
        print(f"Loaded {len(existing_data)} existing embeddings")
        return existing_data, processed_files
    else:
        return [], set()


def create_embedding_result_path(file_path: str, output_dir: str) -> str:
    """Create path for embedding result file.

    Args:
        file_path: Original audio file path
        output_dir: Output directory

    Returns:
        Path to embedding .npy file
    """
    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(output_dir, f"{file_stem}_embedding.npy")


def is_already_extracted(file_path: str, output_dir: str) -> bool:
    """Check if embeddings have already been extracted for a file.

    Args:
        file_path: Original audio file path
        output_dir: Output directory

    Returns:
        True if embeddings already exist
    """
    result_path = create_embedding_result_path(file_path, output_dir)
    return os.path.exists(result_path)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract HeAR embeddings from audio files"
    )
    
    parser.add_argument(
        "audio_dir",
        nargs='?',
        default=AUDIO_DIR,
        help="Directory containing audio files to process"
    )
    
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help=f"Output directory for embeddings (default: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (cpu, cuda, mps). If not specified, auto-detects best available"
    )
    
    return parser.parse_args()


def main():
    """Main processing function."""
    args = parse_arguments()
    
    # Use command line arguments or defaults
    audio_dir = args.audio_dir
    output_dir = args.output
    max_files = args.max_files
    verbose = args.verbose
    device = args.device
    
    # Validate configuration
    if not audio_dir:
        print("ERROR: Please provide audio directory path")
        return

    if not os.path.exists(audio_dir):
        print(f"ERROR: Audio directory does not exist: {audio_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"Audio directory: {audio_dir}")
        print(f"Output directory: {output_dir}")
        if max_files:
            print(f"Max files to process: {max_files}")

    # Initialize HeAR processor
    try:
        processor = HeARProcessor(device=device)
    except Exception as e:
        print(f"ERROR: Failed to initialize HeAR processor: {e}")
        return

    # Get audio file paths
    if verbose:
        print(f"Scanning for audio files in: {audio_dir}")
    audio_file_paths = get_audio_files_from_directory(audio_dir)

    if not audio_file_paths:
        print("No audio files found!")
        return

    # Limit files if max_files is specified
    if max_files:
        audio_file_paths = audio_file_paths[:max_files]
        if verbose:
            print(f"Limited to {len(audio_file_paths)} files")

    print(f"Found {len(audio_file_paths)} audio files to process")

    # Filter out already processed files
    remaining_files = []
    for path in audio_file_paths:
        if not is_already_extracted(path, output_dir):
            remaining_files.append(path)

    print(f"Remaining files to process: {len(remaining_files)}")

    if not remaining_files:
        print("All files already processed!")
        return

    # Process files
    failed_files = []

    progress_desc = "Processing audio files"
    total_windows = 0
    
    for i, file_path in enumerate(tqdm(remaining_files, desc=progress_desc)):
        try:
            # Extract averaged embeddings from all windows
            embeddings, num_windows = processor.process_audio_file(file_path)
            total_windows += num_windows

            # Prepare data for saving
            file_name = os.path.basename(file_path)
            file_stem = os.path.splitext(file_name)[0]

            # Save individual numpy file
            save_embeddings_numpy(embeddings, file_stem, output_dir)

            # Save checkpoint every N files
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                if verbose:
                    avg_windows = total_windows / (i + 1)
                    print(f"\nProcessed {i + 1} files...")
                    print(f"Average windows per file: {avg_windows:.1f}")

        except Exception as e:
            print(f"\nFailed to process {file_path}: {e}")
            failed_files.append(file_path)
            continue

    # Final summary
    total_processed = len(remaining_files) - len(failed_files)
    print("\nProcessing complete!")
    print(f"Total embeddings extracted: {total_processed}")
    print(f"Total windows processed: {total_windows}")
    if total_processed > 0:
        avg_windows = total_windows / total_processed
        print(f"Average windows per file: {avg_windows:.1f}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Results saved to: {output_dir}")

    if failed_files:
        failed_file = os.path.join(output_dir, "failed_files.txt")
        with open(failed_file, 'w') as f:
            for file_path in failed_files:
                f.write(f"{file_path}\n")
        print(f"Failed files list saved to: {failed_file}")


if __name__ == "__main__":
    main()