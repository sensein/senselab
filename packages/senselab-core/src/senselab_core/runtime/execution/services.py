"""Services: servers registered under renewed leases, consumed under leases of their own (ADR-003).

A `ServiceRegistry` is a directory, one per host and set of visible devices (`service_registry`). A service whose
identity is an `Attestation` is named `svc-<identity digest>`:

- `<name>.json` is its `service-entry@1.0` entry, `{identity, endpoint}`;
- `<name>.lease` is the `Lease` its server's process holds and renews (`Lease.keep_alive`); the entry is live while
  that lease is, so the entry carries no heartbeat;
- `<name>.consumers/<uuid>.lease` is one consumer's lease, renewed the same way.

A `svc:<id>` resource class is served by a pool worker that starts each server it needs, registers it
(`ServiceLease`) and keeps it until no consumer lease of it has been live for the profile's `services.idle_exit_s`.
A consumer (`ConsumerLease`) sends each request to the entry's endpoint and checks the reply against the entry.
"""

import contextlib
import logging
import math
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ValidationError

from senselab_core.kernel._base import NonEmptyStr
from senselab_core.kernel.formats import SERVICE_ENTRY_FORMAT, FormatTag, Formatted
from senselab_core.kernel.identity import Attestation
from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.execution_settings import JobLease, VllmSettings
from senselab_core.runtime.hardware import device_set, host_token
from senselab_core.runtime.locks import LEASE_SUFFIX, Lease, LeaseState, LeaseTimeout, LeaseUnavailable
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.storage import write_atomically
from senselab_core.runtime.teardown import run_all
from senselab_core.runtime.vllm_server import Attested, StaleAttestation, attest_reply, chat, complete

logger = logging.getLogger("senselab")

CONSUMER_LEASE = re.compile(rf"^([0-9a-f]{{32}}){re.escape(LEASE_SUFFIX)}$")
"""A consumer's lease in a service's `consumers` directory."""


class ServiceEntry(Formatted):
    """A `service-entry@1.0` document: the identity a running server attests and the endpoint it answers at."""

    format: FormatTag = SERVICE_ENTRY_FORMAT
    port_value: ClassVar[bool] = False
    identity: Attestation
    endpoint: NonEmptyStr


class ServiceRegistry:
    """The services registered in the directory `root`, each live while its lease is.

    Args:
        root: The registry directory, made on first use.
        lease: The profile's `lease`: every registration and consumer lease beats under it.
        vllm: The profile's `vllm` section, whose `request_s` bounds a consumer's request; the packaged profile's
            when None.
        clock: The clock every lease and every consumed-since reading is timed on.
    """

    def __init__(
        self,
        root: Path,
        *,
        lease: JobLease,
        vllm: VllmSettings | None = None,
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
    ) -> None:
        """Use `root` under `lease`; nothing is made until a lease is taken."""
        self.root = Path(root)
        self.lease, self.clock = lease, clock
        self.vllm = vllm if vllm is not None else ExecutionProfile.packaged().vllm

    @staticmethod
    def name(identity: Attestation) -> str:
        """Return the registry name of the service attesting `identity`."""
        return f"svc-{identity.canonical_digest()}"

    def _lease(self, resource: Path, timeout: float) -> Lease:
        return Lease(resource, settings=self.lease, timeout=timeout, clock=self.clock)

    def _kept(self, resource: Path) -> tuple[Lease, contextlib.ExitStack, LeaseState]:
        """Take the lease over `resource` at once and keep it alive; return it, what releases it, and its state.

        Raises:
            LeaseTimeout: a live lease holds `resource`.
        """
        resource.parent.mkdir(parents=True, exist_ok=True)
        lease, stack = self._lease(resource, 0), contextlib.ExitStack()
        try:
            stack.enter_context(lease)
            state = stack.enter_context(lease.keep_alive(self.lease.renew_fraction))
        except BaseException:
            stack.close()
            raise
        return lease, stack, state

    def _live(self, resource: Path) -> bool:
        age = self._lease(resource, 0).age()
        return age is not None and age <= self.lease.stale_after_s

    def reserve(self, identity: Attestation) -> "ServiceLease":
        """Take the registration lease of `identity`, renewed until the `ServiceLease` closes, and publish nothing yet.

        An entry a dead holder left is removed, so no entry is live until `ServiceLease.publish`.

        Raises:
            LeaseTimeout: a live registration of this identity exists.
        """
        name = self.name(identity)
        lease, stack, state = self._kept(self.root / name)
        try:
            (self.root / f"{name}.json").unlink(missing_ok=True)
        except BaseException:
            stack.close()
            raise
        return ServiceLease(self, identity, lease, stack, state)

    def register(self, identity: Attestation, endpoint: str) -> "ServiceLease":
        """Register the server attesting `identity` at `endpoint`: `reserve`, then `ServiceLease.publish`.

        Raises:
            LeaseTimeout: a live registration of this identity exists.
            LeaseUnavailable: the lease was lost before the entry was written.
        """
        service = self.reserve(identity)
        try:
            service.publish(endpoint)
        except BaseException:
            service.close()
            raise
        return service

    def lookup(self, identity: Attestation) -> ServiceEntry | None:
        """Return the entry of the service attesting `identity`, None unless it is registered under a live lease."""
        name = self.name(identity)
        try:
            entry = ServiceEntry.model_validate_json((self.root / f"{name}.json").read_bytes())
        except FileNotFoundError:
            return None
        except ValidationError as err:
            logger.warning("service entry %s cannot be read: %s", self.root / f"{name}.json", err)
            return None
        if entry.identity != identity or not self._live(self.root / name):
            return None
        return entry

    def consume(self, identity: Attestation) -> "ConsumerLease":
        """Take a consumer lease on the service attesting `identity`, renewed until the `ConsumerLease` stops."""
        _, stack, state = self._kept(self.root / f"{self.name(identity)}.consumers" / uuid.uuid4().hex)
        return ConsumerLease(self, identity, stack, state)

    def live_consumers(self, identity: Attestation) -> int:
        """Return how many consumer leases of the service attesting `identity` are live; stale ones are removed."""
        directory = self.root / f"{self.name(identity)}.consumers"
        try:
            names = [m.group(1) for n in sorted(p.name for p in directory.iterdir()) if (m := CONSUMER_LEASE.match(n))]
        except FileNotFoundError:
            return 0
        live = 0
        for token in names:
            if self._live(directory / token):
                live += 1
                continue
            with contextlib.suppress(LeaseTimeout, LeaseUnavailable, OSError), self._lease(directory / token, 0):
                pass
        return live


class ServiceLease:
    """A service's registration of `identity`, kept by its lease's renewer until `close`; its entry once published.

    Attributes:
        identity: What the service attests.
        entry: The published entry; None until `publish`.
        state: The lease's renewer state; once it is lost the entry is not live.
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        identity: Attestation,
        lease: Lease,
        stack: contextlib.ExitStack,
        state: LeaseState,
    ) -> None:
        """Hold the registration of `identity` by `lease`, released by `stack`; consumed as of now."""
        self.registry, self.identity, self.state = registry, identity, state
        self.entry: ServiceEntry | None = None
        self._lease, self._stack = lease, stack
        self._consumed = registry.clock.now()

    def publish(self, endpoint: str) -> ServiceEntry:
        """Write the entry `{identity, endpoint}` while the lease is proven held; consumed as of now.

        Raises:
            LeaseUnavailable: the lease was lost.
        """
        entry = ServiceEntry(identity=self.identity, endpoint=endpoint)
        if self.state.lost or not self._lease.prove():
            raise LeaseUnavailable(
                self._lease.path, f"the service lease {self._lease.path} was lost before it published"
            )
        write_atomically(
            self.registry.root / f"{self.registry.name(self.identity)}.json", entry.model_dump_json().encode()
        )
        self.entry, self._consumed = entry, self.registry.clock.now()
        return entry

    def touch(self) -> None:
        """Count the service as consumed now: its own worker ran jobs on it."""
        self._consumed = self.registry.clock.now()

    def idle_s(self) -> float:
        """Return the seconds since a consumer lease was last seen live (since registration or `touch` if later).

        Infinite once the registration's lease is lost.
        """
        if self.state.lost:
            return math.inf
        if self.registry.live_consumers(self.identity):
            self._consumed = self.registry.clock.now()
        return self.registry.clock.now() - self._consumed

    def close(self) -> None:
        """Remove the entry while the lease still holds it, then release the lease; safe to call twice.

        The release runs even when the removal raises; the first error is raised after both.
        """
        run_all(self._withdraw, self._stack.close)

    def _withdraw(self) -> None:
        if self._lease.owns() and self._lease.prove():
            (self.registry.root / f"{self.registry.name(self.identity)}.json").unlink(missing_ok=True)


class ConsumerLease:
    """A consumer of the service attesting `identity`, live while its lease beats; a `workers.Server` of it.

    Each request goes to the endpoint of the live entry. Its reply is refused (`StaleAttestation`) unless it names the
    attested model and the entry, read again after it, still names that endpoint.

    Attributes:
        label: The served model, as a message names it.
    """

    def __init__(
        self, registry: ServiceRegistry, identity: Attestation, stack: contextlib.ExitStack, state: LeaseState
    ) -> None:
        """Hold the consumer lease released by `stack`."""
        self.registry, self.identity, self.label = registry, identity, identity.served_model
        self._stack, self._state, self._stopped = stack, state, False
        self._request_s = registry.vllm.request_s

    @property
    def attestation(self) -> Attestation | None:
        """The identity consumed; None once stopped. The provider registers only a server that attests it."""
        return None if self._stopped else self.identity

    @property
    def url(self) -> str | None:
        """The live entry's endpoint; None when none is live or once stopped."""
        entry = None if self._stopped else self.registry.lookup(self.identity)
        return None if entry is None else entry.endpoint

    def _entry(self) -> ServiceEntry:
        entry = None if self._stopped or self._state.lost else self.registry.lookup(self.identity)
        if entry is None:
            raise StaleAttestation(f"the service of {self.label} is not registered under a live lease")
        return entry

    def context_fields(self) -> dict[str, Any]:
        """Return the service's attestation as `ExecutionContext.driver` fields.

        Raises:
            StaleAttestation: the service's entry is not live, or this consumer's lease was lost.
        """
        self._entry()
        return {"driver": self.identity.driver()}

    def _request(self, send: Callable[[str], Any]) -> Attested:
        entry = self._entry()
        body = send(entry.endpoint)
        if self._entry().endpoint != entry.endpoint:
            raise StaleAttestation(f"the service of {self.label} was replaced while it answered")
        return attest_reply(body, self.identity, self.label)

    def chat(self, messages: Sequence[Mapping[str, Any]], *, sampling: Mapping[str, Any]) -> Attested:
        """`chat` with the service's model; the reply carries the service's attestation.

        Raises:
            StaleAttestation: the entry is not live, was replaced, or the reply names another model.
            ServerRejected, ServerOutOfMemory, ServerFailed, ServiceUnavailable: the request failed (`server_fault`).
        """
        return self._request(
            lambda url: chat(url, self.identity.served_model, messages, sampling=sampling, timeout_s=self._request_s)
        )

    def complete(self, prompt_token_ids: Sequence[int], *, sampling: Mapping[str, Any]) -> Attested:
        """`complete` with the service's model; the reply carries the service's attestation.

        Raises:
            StaleAttestation: the entry is not live, was replaced, or the reply names another model.
            ServerRejected, ServerOutOfMemory, ServerFailed, ServiceUnavailable: the request failed (`server_fault`).
        """
        return self._request(
            lambda url: complete(
                url, self.identity.served_model, prompt_token_ids, sampling=sampling, timeout_s=self._request_s
            )
        )

    def stop(self) -> None:
        """Release the consumer lease; safe to call twice."""
        self._stopped = True
        self._stack.close()


def service_registry(
    profile: ExecutionProfile,
    environ: Mapping[str, str] | None = None,
    clock: Clock = SystemClock(),  # noqa: B008 - stateless
) -> ServiceRegistry:
    """Return the registry of this host's devices under `profile`: `<server env root>/services/<host token>/<devices>`.

    `<devices>` names the devices this process sees (`device_set` of `environ`, `os.environ` when None), so the
    servers of each card are registered, and consumed, apart. Its leases are timed on `clock`.
    """
    root = profile.env_root("server") / "services" / host_token() / device_set(environ=environ)
    return ServiceRegistry(root, lease=profile.lease, vllm=profile.vllm, clock=clock)


__all__ = [
    "CONSUMER_LEASE",
    "ConsumerLease",
    "ServiceEntry",
    "ServiceLease",
    "ServiceRegistry",
    "service_registry",
]
