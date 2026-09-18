"""Exception hierarchy for the framework core.

All framework errors derive from PytopovizError so a frontend can catch one base.
Kept flat and small; grows only when a caller needs to distinguish a new case.

Author: B.G.
"""

from __future__ import annotations


class PytopovizError(Exception):
    """Base for every framework error."""


class RegistrationError(PytopovizError):
    """Duplicate or invalid registration (type, converter, process)."""


class TypeError_(PytopovizError):
    """Type lookup / identification failure."""


class ConversionError(PytopovizError):
    """No converter path, or a converter raised."""


class ValidationError(PytopovizError):
    """A process call or workflow document failed validation."""


class SessionError(PytopovizError):
    """Handle not found, released, or misused."""


class WorkflowError(PytopovizError):
    """Malformed workflow document or runtime failure."""
