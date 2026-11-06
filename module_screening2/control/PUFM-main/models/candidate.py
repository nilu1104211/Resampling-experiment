"""Unmodified PUFM controls; no auxiliary branch or external observations."""
def build_candidate(config):
    if config['variant'] not in {'b0', 'b_extra'}:
        raise ValueError('The control copy supports b0/b_extra only')
    return None
