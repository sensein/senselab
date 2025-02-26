"""Taxonomy of bioacoustic tasks emphasizing human tasks."""

BIOACOUSTIC_TASK_TREE = {
    "bioacoustic_task": {
        "human": {
            "respiration": {
                "breathing": {
                    "quiet": None,
                    "deep": None,
                    "rapid": None,
                    "sigh": None,
                },
                "exhalation": {
                    "cough": {
                        "voluntary": None,
                        "reflexive": None,
                    },
                },
                "inhalation": {
                    "sniff": None,
                    "gasp": None,
                },
            },
            "vocalization": {
                "speech": {
                    "spontaneous_speech": None,
                    "read_speech": None,
                    "repetitive_speech": {
                        "diadochokinesis": None,
                        "counting": None,
                    },
                    "sustained_phonation": None,
                },
                "non_speech": {
                    "laughter": None,
                    "crying": None,
                    "humming": None,
                    "throat_clearing": None,
                },
            },
        }
    }
}
