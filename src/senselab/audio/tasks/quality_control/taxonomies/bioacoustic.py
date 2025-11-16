"""Bioacoustic activity taxonomy tree implementation."""

from senselab.audio.tasks.quality_control.checks import (
    completely_silent_check,
    high_proportion_clipped_check,
    low_spectral_gating_snr_check,
    very_low_amplitude_modulation_depth_check,
    very_low_peak_snr_from_spectral_check,
)
from senselab.audio.tasks.quality_control.metrics import (
    amplitude_modulation_depth_metric,
    peak_snr_from_spectral_metric,
    proportion_clipped_metric,
    proportion_silent_metric,
    spectral_gating_snr_metric,
)
from senselab.audio.tasks.quality_control.taxonomy import TaxonomyNode


def build_bioacoustic_activity_taxonomy() -> TaxonomyNode:
    """Build the complete bioacoustic activity taxonomy tree.

    Returns:
        The root node of the bioacoustic activity taxonomy tree
    """
    # Create root node
    root = TaxonomyNode(
        name="bioacoustic",
        checks=[
            high_proportion_clipped_check,
            completely_silent_check,
            very_low_peak_snr_from_spectral_check,
            low_spectral_gating_snr_check,
            very_low_amplitude_modulation_depth_check,
        ],
        metrics=[
            proportion_clipped_metric,
            proportion_silent_metric,
            peak_snr_from_spectral_metric,
            spectral_gating_snr_metric,
            amplitude_modulation_depth_metric,
        ],
    )

    return root


# Create the global taxonomy tree instance
BIOACOUSTIC_ACTIVITY_TAXONOMY = build_bioacoustic_activity_taxonomy()
