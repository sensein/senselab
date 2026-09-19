def test_triage_module_resolves_into_worktree():
    import senselab.audio.workflows.triage.nodes.ddk as d
    assert "/agent-a2b266196f55c28a1/" in d.__file__, d.__file__
