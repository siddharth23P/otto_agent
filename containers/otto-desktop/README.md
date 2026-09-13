# containers/otto-desktop/

A throwaway desktop for Otto to look at and click on. A screen-control tool
pointed at a real machine can click anything its owner happens to have open;
pointed at a container it can click only the things somebody deliberately
put in it, which is why the `look` and `look_act` tools reach a desktop only
here.

- `Dockerfile` builds on the image Otto already drives for shell and file
  work, so the code path stays available beside the GUI path, and adds Xvfb,
  fluxbox, xdotool and ImageMagick.
- `start-desktop.sh` starts the virtual display and window manager.

`look <question>` captures the screen and answers the question through the
vision model, so the conversation gets a description and never the pixels.
`look_act` clicks and types by coordinate behind the mutation gate. Two
things a live run taught: `base64` wraps at 76 columns and strict decoding
rejects the newline, so captures use `-w0`; and `xdotool type` needs
`--clearmodifiers`, because a modifier stuck from an earlier click turns
ordinary text into shortcuts.
