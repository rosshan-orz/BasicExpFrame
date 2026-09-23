"""Common type definitions for dataset samples."""

from __future__ import annotations

from typing import Any, Dict, TypedDict


class SampleDict(TypedDict, total=False):
    """A sample returned by a :class:`torch.utils.data.Dataset`.

    ``inputs`` and ``targets`` are the stable framework keys.  ``metadata`` is
    optional and can carry subject or acquisition information without coupling
    the trainer to a concrete dataset implementation.
    """

    inputs: Any
    targets: Any
    metadata: Dict[str, Any]
