"""MDAT dataset loading and whole-document selection."""

from .selection import DatasetSelection

__all__ = ['MatformerDataset', 'DatasetSelection']


def __getattr__(name):
    # A lazy import also keeps ``python -m ...mdat.mdat`` warning-free.
    if name == 'MatformerDataset':
        from .mdat import MatformerDataset
        return MatformerDataset
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
