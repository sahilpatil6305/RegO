"""Feature extraction module stub"""

def extract(args):
    """
    Stub function for feature extraction.
    Returns args unchanged.
    """
    print("[INFO] Feature extraction stub called - returning args unchanged")
    return args


def collator_audio(*args, **kwargs):
    """
    Stub for collator_audio - should use collate_fn.collator_audio instead.
    """
    from collate_fn import collator_audio as real_collator
    return real_collator(*args, **kwargs)
