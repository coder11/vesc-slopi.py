#!/usr/bin/env python3
"""Minimal VESC TCP packet multiplexer.

Start VESC Tool's TCP bridge first, then run this mux:
    vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer 65102
    uv run examples/vesc_mux.py --in 65102 --out 65122

Point clients at the mux output port:
    uv run examples/imu_live_plot.py --tcp 127.0.0.1:65122
    uv run examples/config_tui.py --tcp 127.0.0.1:65122
"""

from __future__ import annotations

import argparse
import queue
import signal
import socket
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from vesc_py.packet import PacketDecoder, encode_packet

DEFAULT_HOST = "127.0.0.1"
RECV_SIZE = 4096


@dataclass(frozen=True)
class QueuedPacket:
    """A complete decoded VESC request from one downstream client."""

    client: socket.socket
    client_name: str
    payload: bytes


def parse_endpoint(value: str, *, default_host: str = DEFAULT_HOST) -> tuple[str, int]:
    """Parse PORT or HOST:PORT."""
    host, sep, port_text = value.rpartition(":")
    if not sep:
        host = default_host
        port_text = value

    try:
        port = int(port_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("endpoint must be PORT or HOST:PORT") from exc

    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in range 1..65535")

    return host, port


class VescMux:
    """Queue complete VESC packets and serialize access to one upstream socket."""

    def __init__(
        self,
        *,
        upstream: tuple[str, int],
        bind: tuple[str, int],
    ) -> None:
        self._upstream_addr = upstream
        self._bind_addr = bind
        self._requests: queue.Queue[QueuedPacket] = queue.Queue()
        self._stop = threading.Event()
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        self._upstream_decoder = PacketDecoder()
        self._upstream: socket.socket | None = None
        self._server: socket.socket | None = None

    def serve_forever(self) -> None:
        """Connect upstream, listen for clients, and block until stopped."""
        self._upstream = socket.create_connection(self._upstream_addr)
        self._upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(self._bind_addr)
        self._server.listen()
        self._server.settimeout(0.5)

        worker = threading.Thread(target=self._process_requests, name="vesc-mux-worker")
        worker.start()

        print(
            f"Connected upstream {self._upstream_addr[0]}:{self._upstream_addr[1]}; "
            f"listening on {self._bind_addr[0]}:{self._bind_addr[1]}"
        )

        try:
            self._accept_loop()
        finally:
            self.stop()
            worker.join(timeout=2.0)

    def stop(self) -> None:
        """Stop accepting work and close sockets."""
        self._stop.set()
        for sock in (self._server, self._upstream):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()

        for client in clients:
            self._close_socket(client)

    def _accept_loop(self) -> None:
        if self._server is None:
            raise RuntimeError("server socket is not initialized")

        while not self._stop.is_set():
            try:
                client, address = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise

            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_name = f"{address[0]}:{address[1]}"
            with self._clients_lock:
                self._clients.add(client)

            print(f"client connected: {client_name}")
            thread = threading.Thread(
                target=self._read_client,
                args=(client, client_name),
                name=f"vesc-mux-client-{client_name}",
                daemon=True,
            )
            thread.start()

    def _read_client(self, client: socket.socket, client_name: str) -> None:
        decoder = PacketDecoder()
        try:
            while not self._stop.is_set():
                data = client.recv(RECV_SIZE)
                if not data:
                    return
                for payload in decoder.process(data):
                    self._requests.put(
                        QueuedPacket(client=client, client_name=client_name, payload=payload)
                    )
        except OSError:
            return
        finally:
            with self._clients_lock:
                self._clients.discard(client)
            self._close_socket(client)
            print(f"client disconnected: {client_name}")

    def _process_requests(self) -> None:
        while not self._stop.is_set():
            try:
                request = self._requests.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                response = self._round_trip(request.payload)
            except OSError as exc:
                print(f"upstream error while handling {request.client_name}: {exc}")
                self.stop()
                return

            if not self._is_client_connected(request.client):
                continue

            try:
                request.client.sendall(encode_packet(response))
            except OSError:
                with self._clients_lock:
                    self._clients.discard(request.client)
                self._close_socket(request.client)

    def _round_trip(self, payload: bytes) -> bytes:
        if self._upstream is None:
            raise RuntimeError("upstream socket is not initialized")

        self._upstream.sendall(encode_packet(payload))

        while not self._stop.is_set():
            for response in self._upstream_decoder.process(b""):
                return response

            data = self._upstream.recv(RECV_SIZE)
            if not data:
                raise ConnectionError("upstream closed")

            for response in self._upstream_decoder.process(data):
                return response

        raise ConnectionError("mux stopped before upstream response")

    def _is_client_connected(self, client: socket.socket) -> bool:
        with self._clients_lock:
            return client in self._clients

    @staticmethod
    def _close_socket(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Serialize multiple local VESC TCP clients through one upstream connection."
    )
    parser.add_argument(
        "--in",
        dest="upstream",
        type=parse_endpoint,
        required=True,
        metavar="PORT|HOST:PORT",
        help="Upstream VESC Tool TCP server endpoint, e.g. 65102.",
    )
    parser.add_argument(
        "--out",
        dest="bind",
        type=parse_endpoint,
        required=True,
        metavar="PORT|HOST:PORT",
        help="Local endpoint to listen on for clients, e.g. 65122.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mux."""
    args = build_parser().parse_args(argv)
    mux = VescMux(upstream=args.upstream, bind=args.bind)

    def _handle_signal(_signum: int, _frame: object) -> None:
        mux.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        mux.serve_forever()
    except KeyboardInterrupt:
        mux.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
