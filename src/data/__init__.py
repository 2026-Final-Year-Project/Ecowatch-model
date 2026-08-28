"""Dataset discovery, preprocessing, splitting, and loading utilities."""

# Submodules are intentionally not imported eagerly. This keeps lightweight
# inference independent from scikit-learn, which is needed only to create splits.
