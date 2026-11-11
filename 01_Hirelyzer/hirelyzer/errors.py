"""Exception hierarchy for Hirelyzer.

Every error the pipeline raises on purpose inherits from HirelyzerError,
so callers can catch one base class and still tell the cases apart.
"""


class HirelyzerError(Exception):
    """Base class for all Hirelyzer errors."""


class ConfigError(HirelyzerError):
    """Required configuration is missing or invalid (raised at startup)."""


class UnsupportedFileError(HirelyzerError):
    """The input file type is not supported."""


class EmptyDocumentError(HirelyzerError):
    """The file, or the text extracted from it, contains nothing to analyse."""


class InputTooLargeError(HirelyzerError):
    """The input file exceeds the configured size limit."""


class LLMCallError(HirelyzerError):
    """The LLM request failed for good: a non-transient error, or retries ran out."""


class OutputValidationError(HirelyzerError):
    """The model output could not be parsed or validated, even after repair attempts."""
