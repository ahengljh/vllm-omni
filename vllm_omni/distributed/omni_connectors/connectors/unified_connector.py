# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass
from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


@dataclass(frozen=True)
class _BackendSpec:
    name: str
    connector: OmniConnectorBase
    min_bytes: int | None = None
    max_bytes: int | None = None

    def matches_size(self, size: int) -> bool:
        if self.min_bytes is not None and size < self.min_bytes:
            return False
        if self.max_bytes is not None and size > self.max_bytes:
            return False
        return True


class UnifiedConnector(OmniConnectorBase):
    """Tiered connector that unifies inline, SHM, and remote backends.

    Config format (example):
    {
        "inline_threshold_bytes": 65536,
        "max_retries": 2,
        "retry_backoff_s": 0.05,
        "backends": [
            {"name": "SharedMemoryConnector", "extra": {"shm_threshold_bytes": 1048576}, "max_bytes": 1048576},
            {"name": "MooncakeConnector", "extra": {...}},
            {"name": "YuanrongConnector", "extra": {...}},
        ],
    }
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.inline_threshold_bytes = int(config.get("inline_threshold_bytes", 0))
        self.max_retries = int(config.get("max_retries", 1))
        self.retry_backoff_s = float(config.get("retry_backoff_s", 0.05))

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "inline_puts": 0,
            "inline_gets": 0,
            "backend_puts": 0,
            "backend_gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "fallbacks": 0,
        }

        self._backends = self._init_backends(config)
        self._backend_map = {b.name: b for b in self._backends}

    def _init_backends(self, config: dict[str, Any]) -> list[_BackendSpec]:
        raw_backends = config.get("backends", [])
        if not raw_backends:
            raw_backends = []
            for key in ["primary", "secondary", "tertiary"]:
                if key in config:
                    raw_backends.append(config[key])

        if not raw_backends:
            raise ValueError("UnifiedConnector requires at least one backend configuration")

        from ..factory import OmniConnectorFactory
        from ..utils.config import ConnectorSpec

        backends: list[_BackendSpec] = []
        for entry in raw_backends:
            if not isinstance(entry, dict):
                raise ValueError("UnifiedConnector backend entries must be dicts")

            name = entry.get("name")
            if not name:
                raise ValueError("UnifiedConnector backend missing 'name'")
            if name == self.__class__.__name__:
                raise ValueError("UnifiedConnector cannot include itself as a backend")

            extra = entry.get("extra", {})
            if not isinstance(extra, dict):
                raise ValueError("UnifiedConnector backend 'extra' must be a dict")

            connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=name, extra=extra))
            backends.append(
                _BackendSpec(
                    name=name,
                    connector=connector,
                    min_bytes=entry.get("min_bytes"),
                    max_bytes=entry.get("max_bytes"),
                )
            )

        return backends

    def _select_backends(self, size: int) -> list[_BackendSpec]:
        matches = [b for b in self._backends if b.matches_size(size)]
        return matches if matches else list(self._backends)

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            payload = self.serialize_obj(data)
            size = len(payload)

            if self.inline_threshold_bytes > 0 and size <= self.inline_threshold_bytes:
                self._metrics["puts"] += 1
                self._metrics["inline_puts"] += 1
                self._metrics["bytes_transferred"] += size
                return True, size, {"inline_bytes": payload, "size": size, "backend": "inline"}

            backends = self._select_backends(size)
            last_error: Exception | None = None
            for backend in backends:
                for attempt in range(self.max_retries):
                    try:
                        success, _, metadata = backend.connector.put(from_stage, to_stage, put_key, data)
                        if success:
                            self._metrics["puts"] += 1
                            self._metrics["backend_puts"] += 1
                            self._metrics["bytes_transferred"] += size
                            return True, size, {
                                "backend": backend.name,
                                "backend_metadata": metadata,
                                "size": size,
                            }
                        self._metrics["fallbacks"] += 1
                    except Exception as exc:
                        last_error = exc
                        self._metrics["fallbacks"] += 1
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_backoff_s * (2**attempt))

            if last_error:
                raise last_error

            return False, 0, None
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("UnifiedConnector put failed: %s", exc)
            return False, 0, None

    def get(
        self, from_stage: str, to_stage: str, get_key: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        try:
            if metadata is not None:
                if not isinstance(metadata, dict) or not metadata:
                    return None

                if "inline_bytes" in metadata:
                    payload = metadata["inline_bytes"]
                    obj = self.deserialize_obj(payload)
                    self._metrics["gets"] += 1
                    self._metrics["inline_gets"] += 1
                    return obj, len(payload)

                backend_name = metadata.get("backend")
                backend_metadata = metadata.get("backend_metadata")
                if backend_name and backend_name in self._backend_map:
                    result = self._backend_map[backend_name].connector.get(
                        from_stage, to_stage, get_key, metadata=backend_metadata
                    )
                    if result:
                        self._metrics["gets"] += 1
                        self._metrics["backend_gets"] += 1
                    return result

            # No metadata or unknown backend: try all backends in order.
            for backend in self._backends:
                try:
                    result = backend.connector.get(from_stage, to_stage, get_key, metadata=None)
                    if result:
                        self._metrics["gets"] += 1
                        self._metrics["backend_gets"] += 1
                        return result
                except Exception:
                    self._metrics["fallbacks"] += 1
                    continue

            return None
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("UnifiedConnector get failed: %s", exc)
            return None

    def cleanup(self, request_id: str) -> None:
        for backend in self._backends:
            try:
                backend.connector.cleanup(request_id)
            except Exception:
                continue

    def health(self) -> dict[str, Any]:
        health = {"status": "healthy", **self._metrics}
        backend_health = {}
        for backend in self._backends:
            try:
                backend_health[backend.name] = backend.connector.health()
            except Exception as exc:
                backend_health[backend.name] = {"status": "unhealthy", "error": str(exc)}
        health["backends"] = backend_health
        health["inline_threshold_bytes"] = self.inline_threshold_bytes
        return health

    def close(self) -> None:
        for backend in self._backends:
            close_fn = getattr(backend.connector, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    continue
