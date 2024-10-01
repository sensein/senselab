# Audio data augmentation

<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/audio_data_augmentation.ipynb'">Tutorial</button>

## Task Overview
Data augmentation involves creating synthetic audio samples by adding some perturbations to the original data. This technique helps mimic real-world variations, making the audio data more robust and versatile for different contexts, whether for creative, practical, or analytical purposes. Importantly, in the machine learning space, these perturbations must maintain the same label as the original training sample.

### Common Techniques

Here are some commonly used audio data augmentation techniques:

- **Noise Injection**: Adds background noise to simulate real-world environments, such as crowds, traffic, or machinery.
- **Pitch Shifting**: Changes the pitch by altering the sound frequency without affecting its speed, often used to modify the tonal quality or mood.
- **Time Stretching**: Speeds up or slows down the audio without changing its pitch, useful for adjusting the timing of audio to meet specific requirements.
- **Volume Adjustment**: Increases or decreases loudness to simulate different recording conditions or adjust audio levels.
- **Reverb**: Applies echo or reverberation effects to simulate different acoustic environments, adding depth and space to the audio.

### Libraries: `audiomentations` and `torch-audiomentations`

- [audiomentations](https://github.com/iver56/audiomentations): A CPU-based Python library offering a wide variety of audio augmentation transforms. It's inspired by albumentations and optimized for deep learning tasks, such as speech processing and noise-robustness testing. It supports both mono and multichannel audio, with an easy-to-use interface.

- [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations): A GPU-accelerated augmentation library for PyTorch, allowing for efficient real-time augmentation. This is ideal for speeding up model training and reducing data loading times. It supports a subset of the techniques from audiomentations but is optimized for high-speed processing.
Compared to `audiomentations`, `torch-audiomentations` offers a more limited set of augmentation types.

In `senselab`, when these libraries run on a CPU, they utilize concurrent futures through `Pydra` for optimization.

For more information on CPU vs GPU audio data augmentation, including the pros and cons, see [this guide](https://iver56.github.io/audiomentations/guides/cpu_vs_gpu/).
