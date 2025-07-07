from typing import Tuple

from starlette.types import ASGIApp, Receive, Scope, Send


class ProxyHeadersMiddleware:
    """
    Middleware to handle proxy headers, such as X-Forwarded-For,
    X-Forwarded-Proto, and X-Forwarded-Host.
    This is necessary when running behind a reverse proxy (like Nginx or Traefik)
    that terminates TLS and forwards requests to the application.
    """
    def __init__(self, app: ASGIApp, trusted_hosts: str = "127.0.0.1"):
        self.app = app
        self.trusted_hosts = {host.strip() for host in trusted_hosts.split(",")}

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])

            # Handle X-Forwarded-Proto
            if b"x-forwarded-proto" in headers:
                x_forwarded_proto = headers[b"x-forwarded-proto"].decode("latin-1")
                scope["scheme"] = x_forwarded_proto

            # Handle X-Forwarded-Host
            if b"x-forwarded-host" in headers:
                x_forwarded_host = headers[b"x-forwarded-host"].decode("latin-1")
                host, port = self.parse_host_port(x_forwarded_host, scope)
                scope["server"] = (host, port)

            # Handle X-Forwarded-For
            if b"x-forwarded-for" in headers:
                x_forwarded_for = headers[b"x-forwarded-for"].decode("latin-1")
                # The client IP is the first IP in the list
                client_ip = x_forwarded_for.split(",")[0].strip()
                scope["client"] = (client_ip, 0)

        await self.app(scope, receive, send)

    def parse_host_port(self, host_header: str, scope: Scope) -> Tuple[str, int]:
        """
        Parses the host header to extract host and port.
        """
        if ":" in host_header:
            host, port_str = host_header.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                # Handle cases where the part after ':' is not a valid port
                host = host_header
                port = 80  # Default to 80 if port is invalid
        else:
            host = host_header
            # Default to 443 for https, 80 for http
            port = 443 if scope.get("scheme") == "https" else 80
        return host, port 