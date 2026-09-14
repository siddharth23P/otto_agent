# agent/phone/

The phone as a place Otto works. The device-agnostic half of the phone
helper: what a phone must provide, the tools the agent gets over it, the
screen digest the model reads, and the money guard's rules. The Android app
supplies the other half (the accessibility service, the enforcement, the UI)
and binds these through [agent/embed.py](../embed.py).

| module | what it is |
| --- | --- |
| `backend.py` | `PhoneBackend`, the Protocol a host implements (tree, tap, type, press, swipe, scroll, screenshot, apps, launch, settings, install); `PhoneError` with a code and a hand-over flag; `JsonBackend`, the adapter over a bridge whose methods return JSON envelopes |
| `digest.py` | the accessibility snapshot as bounded, inert text: one `[index] "label" role flags @x,y` line per element, password fields never shown; `find_node` resolves a text target (exact, then unique substring, else the candidates) |
| `guard.py` + `assets/guard_rules.json` | the rules the phone enforces and this side pre-checks: denied packages, money words, sensitive-screen patterns (two signals required), pay words (never tappable), commit words (only through `phone_commit`) |
| `tools.py` | `phone_tools(backend)`: eight `ExtraTool`s, `PHONE_GUIDANCE`, and the standing tools a phone cannot run |

## The tools

| tool | mutates | body |
| --- | --- | --- |
| `phone_screen` | no | `{}` -- the app in front and every element with a number |
| `phone_act` | no | `{op, ...}` -- `tap`/`tap_text`/`long_press` by text or `x,y`, `type`, `press` (`back`, `home`, `recents`, `enter`), `swipe`, `scroll`; the screen after |
| `phone_commit` | yes | `{target}` -- a Send/Delete/Confirm; held once by the mutation gate |
| `phone_open` | no | `{app}` -- by label or package |
| `phone_apps` | no | `{query?}` |
| `phone_look` | no | `{question}` -- a screenshot through the vision seat, words back |
| `phone_settings` | no | `{page, package?}` -- a Settings page by intent |
| `phone_install` | yes | `{package?, query?}` -- the Play listing, Install tapped; free apps only |

Every result is third-party content to the loop, exactly as a web page is.
The mutation gate keys on the first line of the body, so a one-line JSON
body holds `phone_install` once per package and never holds a tap.

## The stop rules

Code, on both sides, and the phone's verdict wins:

- A **denied or money-worded package** in front: nothing is described or
  touched; `press back`/`home` are the way out.
- A **sensitive screen** ("UPI PIN", "OTP", "CVV", "Pay ₹499"): refused only
  with a second signal -- an input field that asks for it, a money-worded
  package, or a secure window -- so a chat that mentions an OTP is not one.
- A **pay word** on the target ("Pay now", "Place order", "Buy now"): never
  tapped, whatever tool asks. Reaching checkout is allowed; paying is not.
- A **commit word** ("Send", "Delete", "Confirm"): only `phone_commit`.
- A **password field**: never typed into.

A refusal comes back as a result beginning `GUARD:`; when the phone itself
refused with a hand-over, the result says the person has taken over. The
rules file is package data; the app vendors a copy and its CI asserts the
two are identical, so a change here fails the app's bump until it copies it.
