"""Exceptions shared by MDAT storage, preprocessing, and loading."""

class MDatNotFound(Exception):
    """Raised when dataset is not found."""

class MDatAlreadyPresent(Exception):
    """Raised when dataset already exists and overwrite is False."""

class SubmDatNotFound(Exception):
    """Raised when sub-dataset is not found."""

class SubMdatMissesStrategy(Exception):
    """SubMdatMissesStrategy"""

class MDatIsReadOnly(Exception):
    """Raised when trying to modify readonly dataset."""

class TooManySubmDats(Exception):
    """Raised when too many submdats."""

class MDatAlreadyShuffled(Exception):
    """Raised when mdat is already shuffled."""

class NameAlreadyUsed(Exception):
    """Raised when name is already used."""

class StrategyNotRegistered(Exception):
    """Raised when strategy is not registered."""

class MDatNotShuffled(Exception):
    """MDatNotShuffled"""

class MdatIsWriteOnly(Exception):
    """MdatIsWriteOnly"""

class MissingStrategyKey(Exception):
    """MissingStrategyKey"""

class StrategyNotFound(Exception):
    """StrategyNotFound"""

class SplitterClassNotFound(Exception):
    """SplitterClassNotFound"""

class MissingDataKey(Exception):
    """MissingDataKey"""

class MissingSplitterOutput(Exception):
    """MissingSplitterOutput"""

class InvalidChunksFormat(Exception):
    """InvalidChunksFormat"""

class SubMdatAlreadyExists(Exception):
    """The requested subdataset already exists."""

# Historical spelling retained for callers.
MdatIsReadOnly = MDatIsReadOnly
