#!/usr/bin/env bash

set -euo pipefail

port="${1:-4001}"
host="${JEKYLL_HOST:-127.0.0.1}"

if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    printf 'Usage: %s [port]\n' "$0" >&2
    exit 2
fi

if ! ruby -rsocket -e 'server = TCPServer.new(ARGV.fetch(0), Integer(ARGV.fetch(1))); server.close' "$host" "$port" 2>/dev/null; then
    printf 'Port %s is already in use on %s.\n' "$port" "$host" >&2
    printf 'Stop the existing preview or try: %s %s\n' "$0" "$((port + 1))" >&2
    exit 1
fi

printf 'Building and serving the site at http://%s:%s\n' "$host" "$port"
printf 'Press Ctrl-C to stop the preview.\n\n'

jekyll_args=(serve --host "$host" --port "$port")

if [[ -f Gemfile ]] && command -v bundle >/dev/null 2>&1; then
    exec bundle exec jekyll "${jekyll_args[@]}"
fi

if command -v jekyll >/dev/null 2>&1; then
    exec jekyll "${jekyll_args[@]}"
fi

if command -v ruby >/dev/null 2>&1 && ruby -e 'exit Gem::Specification.find_all_by_name("jekyll").empty? ? 1 : 0' 2>/dev/null; then
    exec ruby -e 'load Gem.bin_path("jekyll", "jekyll")' -- "${jekyll_args[@]}"
fi

printf 'Jekyll is not installed. Install it with: gem install jekyll\n' >&2
exit 1