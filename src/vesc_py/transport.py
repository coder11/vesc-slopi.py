"""Byte transport abstraction for VESC communication."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Transport(ABC):
    """Abstract byte transport for VESC communication."""

    @abstractmethod
    def send(self, data: bytes) -> None:
        """Send raw framed bytes."""

    @abstractmethod
    def recv(self, timeout: float) -> bytes:
        """Receive raw bytes, returning empty bytes on timeout."""

    @abstractmethod
    def close(self) -> None:
        """Close the transport and release associated resources."""


__all__ = ["Transport"]
