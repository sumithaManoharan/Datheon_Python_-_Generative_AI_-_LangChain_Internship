"""Typed exceptions raised on purpose by the DocQA pipeline."""


class DocQAError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(DocQAError):
    """Missing or invalid configuration."""
