"""Network guard for senselab subprocess-venv workers; imported by ``site`` before the worker's own code.

Active only when ``SENSELAB_BLOCK_NETWORK`` is set to a true value. It then refuses every
``connect``/``connect_ex``/``sendto`` to a non-loopback address and every name lookup of a
non-loopback host, raising ``PermissionError`` / ``socket.gaierror`` without sending a packet.
``AF_UNIX`` sockets and loopback addresses (``127.0.0.0/8``, ``::1``, ``localhost``) are allowed.
Standard library only: the worker venvs do not have senselab installed. A ``sitecustomize`` later
on ``sys.path`` (the venv's own, if any) is still executed afterwards when this file runs as
``sitecustomize``.
"""

import os
import sys

_TRUE = ("1", "true", "yes", "on")


def _install() -> None:
    import ipaddress
    import socket

    def loopback_host(host: object) -> bool:
        if isinstance(host, bytes):
            host = host.decode("ascii", "ignore")
        if not isinstance(host, str):
            return False
        name = host.strip("[]").split("%", 1)[0].lower()
        if name in ("localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"):
            return True
        try:
            return ipaddress.ip_address(name).is_loopback
        except ValueError:
            return False

    def allowed(sock: "socket.socket", address: object) -> bool:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return True
        return isinstance(address, tuple) and bool(address) and loopback_host(address[0])

    def refuse(address: object) -> PermissionError:
        return PermissionError(f"SENSELAB_BLOCK_NETWORK: non-loopback connection refused: {address!r}")

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex

    def connect(self: "socket.socket", address: object) -> None:
        if not allowed(self, address):
            raise refuse(address)
        return original_connect(self, address)  # type: ignore[arg-type]

    def connect_ex(self: "socket.socket", address: object) -> int:
        if not allowed(self, address):
            raise refuse(address)
        return original_connect_ex(self, address)  # type: ignore[arg-type]

    def sendto(self: "socket.socket", data: bytes, *args: object) -> int:
        address = args[-1] if args else None
        if not allowed(self, address):
            raise refuse(address)
        return original_sendto(self, data, *args)  # type: ignore[arg-type,call-overload]

    def getaddrinfo(host: object, *args: object, **kwargs: object) -> list:
        if host is not None and not loopback_host(host):
            raise socket.gaierror(socket.EAI_NONAME, f"SENSELAB_BLOCK_NETWORK: lookup of {host!r} refused")
        return original_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    def gethostbyname(host: str) -> str:
        if not loopback_host(host):
            raise socket.gaierror(socket.EAI_NONAME, f"SENSELAB_BLOCK_NETWORK: lookup of {host!r} refused")
        return original_gethostbyname(host)

    def gethostbyname_ex(host: str) -> tuple:
        if not loopback_host(host):
            raise socket.gaierror(socket.EAI_NONAME, f"SENSELAB_BLOCK_NETWORK: lookup of {host!r} refused")
        return original_gethostbyname_ex(host)

    socket.socket.connect = connect  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign,assignment]
    socket.socket.sendto = sendto  # type: ignore[method-assign,assignment]
    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
    socket.gethostbyname = gethostbyname
    socket.gethostbyname_ex = gethostbyname_ex  # type: ignore[assignment]
    socket.senselab_network_guard = True  # type: ignore[attr-defined]


def _chain() -> None:
    import importlib.machinery
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


if os.environ.get("SENSELAB_BLOCK_NETWORK", "").strip().lower() in _TRUE:
    _install()
if __name__ == "sitecustomize":
    _chain()
