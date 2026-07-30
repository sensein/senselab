"""Dataset-specific metadata providers for :class:`~senselab.audio.data_structures.AudioPlus`.

Each module here implements the
:class:`~senselab.audio.data_structures.audio_plus.MetadataProvider` protocol for one
dataset layout. Nothing in the rest of the library imports from this package — providers
are injected by the caller — so an unavailable dataset or an optional dependency can never
break a plain senselab install.
"""
