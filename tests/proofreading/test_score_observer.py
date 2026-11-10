from scripts.proofreading.watch_comparison_scores import eligible


def test_observer_only_scores_terminal_safe_session_paths():
    assert eligible({'status':'complete','id':'abc-123'})
    assert eligible({'status':'error','id':'abc-123'})
    for status in ['queued','running','ready',None]:
        assert not eligible({'status':status,'id':'abc-123'})
    assert not eligible({'status':'complete','id':'../outside'})
    assert not eligible({'status':'complete'})
