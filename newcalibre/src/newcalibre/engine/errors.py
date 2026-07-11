"""Define public engine failures shared by deep engine modules."""


class EngineError(ValueError):
    """Report an invalid engine input or stage result."""


__all__ = ["EngineError"]
