# Video SSL embeddings

`extract_ssl_embeddings_from_video(video, model, *, frames_per_clip, fps, pooling)` embeds a video with a
self-supervised Hugging Face video model such as V-JEPA 2. The video is resampled to `fps`, cut into consecutive
clips of `frames_per_clip` frames, and each clip becomes one vector (`pooling="mean"` averages its token states).

The model is an `HFModel` whose `revision` is a resolved 40-hex commit; a branch or tag name is refused, so the
returned provenance record names the exact weights. The record also holds the clip settings, the `torch` and
`transformers` versions and the device.

```python
from senselab.utils.data_structures import HFModel
from senselab.video.data_structures import Video
from senselab.video.tasks.ssl_embeddings import extract_ssl_embeddings_from_video

model = HFModel(path_or_uri="facebook/vjepa2-vitl-fpc16-256-ssv2", revision="<40-hex commit>")
result = extract_ssl_embeddings_from_video(
    Video(filepath="clip.mp4"), model, frames_per_clip=16, fps=4.0, pooling="mean"
)
result.embeddings.shape  # (n_clips, hidden)
```
