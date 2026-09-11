#!/bin/bash
# Bring up a screen and a window manager, then stay alive so `docker exec` can
# drive it. Nothing here is Otto-specific: Otto reaches in through the same
# command runner it uses for bash and the file tools.
set -e
Xvfb "$DISPLAY" -screen 0 "$SCREEN_GEOMETRY" -nolisten tcp &
for _ in $(seq 1 50); do
    xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && break
    sleep 0.2
done
fluxbox >/dev/null 2>&1 &
exec tail -f /dev/null
