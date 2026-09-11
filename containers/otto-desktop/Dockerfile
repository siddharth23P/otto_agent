# A throwaway desktop for Otto to look at and click on.
#
# Otto's other surfaces are sandboxes and this one is too, deliberately. A
# screen-control tool pointed at a real machine can click anything its owner
# has open; pointed at a container it can click the things somebody put in the
# container. That difference is the whole reason this file exists.
#
# Built on the Claw-Eval agent image so the desktop inherits a browser and the
# shell tooling Otto already drives -- which matters, because the measured
# advice is that the CODE path beats the GUI path (agents preferring code take
# about a third fewer steps at a higher score). A desktop with no shell on it
# would force the worse path.
FROM claw-eval-agent:latest

# The base image points apt at a regional mirror that is slow from elsewhere.
RUN sed -i 's|mirrors.ustc.edu.cn|deb.debian.org|g' /etc/apt/sources.list.d/* 2>/dev/null || true

# Xvfb is the screen; fluxbox gives windows something to be managed by;
# xdotool is the mouse and keyboard; imagemagick's `import` is the camera.
# xterm and the x11-apps demos exist so there is something to actually do.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11-utils fluxbox xdotool imagemagick xterm x11-apps \
    && rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:99 \
    SCREEN_GEOMETRY=1280x800x24

COPY start-desktop.sh /usr/local/bin/start-desktop.sh
RUN chmod +x /usr/local/bin/start-desktop.sh

ENTRYPOINT ["/usr/local/bin/start-desktop.sh"]
