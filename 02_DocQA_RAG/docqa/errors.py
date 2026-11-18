"""Typed exceptions raised on purpose by the DocQA pipeline."""


class DocQAError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(DocQAError):
    """Missing or invalid configuration."""


class UnsupportedFileError(DocQAError):
    """The file type is not one the pipeline can read."""


class DocumentReadError(DocQAError):
    """A file is missing, unreadable, corrupt or encrypted."""


class InputTooLargeError(DocQAError):
    """A file is bigger than the configured size limit."""


class EmptyDocumentError(DocQAError):
    """A document has no extractable text (blank, or a scanned PDF)."""


class EmbeddingError(DocQAError):
    """An embedding request failed or returned an unusable response."""


class StoreError(DocQAError):
    """The vector store could not be opened, written or queried."""
