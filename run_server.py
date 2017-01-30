#!/usr/bin/env python3

from http.server import HTTPServer, CGIHTTPRequestHandler


def main():
    port = 8080
    server = HTTPServer(('127.0.0.1', port), CGIHTTPRequestHandler)
    print("Starting simple http server on:")
    print("http://localhost:" + str(server.server_port))
    server.serve_forever()

if __name__ == '__main__':
    main()
