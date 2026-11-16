"""Tissue strength: the lab-measured half of a stress / strength comparison.

``mechanics`` holds the unit and stress-measure conversions (stdlib only, so
the dataset parser can import it without numpy). ``strength`` loads the
packaged strength library and computes the strength index band.
Nothing is imported here, so ``import meshforge.tissue.mechanics`` stays
dependency-free.
"""
