#!/usr/bin/env python3
"""Forward a tailnet-only TCP listener to the loopback MLflow tunnel."""

from __future__ import annotations

import argparse
import select
import socket
import threading


def relay(client: socket.socket, backend_host: str, backend_port: int) -> None:
    try:
        with socket.create_connection(
            (backend_host, backend_port), timeout=10
        ) as backend:
            backend.settimeout(None)
            peers = {client: backend, backend: client}
            open_sockets = set(peers)
            while open_sockets:
                readable, _, _ = select.select(open_sockets, (), ())
                for source in readable:
                    destination = peers[source]
                    try:
                        data = source.recv(65536)
                    except OSError:
                        data = b""
                    if data:
                        destination.sendall(data)
                    else:
                        open_sockets.discard(source)
                        try:
                            destination.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
    except OSError:
        pass
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=15000)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.listen_host, args.listen_port))
        listener.listen(128)
        while True:
            client, _ = listener.accept()
            threading.Thread(
                target=relay,
                args=(client, args.backend_host, args.backend_port),
                daemon=True,
            ).start()


if __name__ == "__main__":
    main()
