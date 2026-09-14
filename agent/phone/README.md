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
| `guard.py` + `assets/guard_rules.json` | the rules the phone enforces and this side pre-checks: denied packages, money words, sensitive-screen patterns (two signals required), pay words (never tappable), forward words that become pay words next to a checkout signal, commit words (only through `phone_commit`); a broken rules file is a `GuardRulesError` on every verdict |
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
- A **pay word** on the target ("Pay now", "Place order", and the bare
  "Pay", "Buy", "Purchase", "Subscribe"): never tapped, whatever tool asks.
  A **forward word** ("Continue", "Next", "Confirm", "Done") is a pay word
  when the screen shows a **checkout signal** (a total, "payment", a card
  field): that is what the last button of a checkout is usually called.
- A **commit word** ("Send", "Delete", "Confirm", and "Checkout": reaching
  the payment page is the person's call, once): only `phone_commit`.
- `press enter` is the keyboard's submit. It has no label to judge, so the
  screen is judged instead: the sensitive-screen verdict, then a checkout
  signal or any pay button on the screen (what Enter would submit) refuses
  it. `back`, `home` and `recents` are the way out and always allowed.
- A tap by coordinates is judged by the element under the point. A point
  with no element under it is refused on a screen that has clickable
  elements (a checkout drawn on a canvas inside an ordinary page is exactly
  the case). On a screen with none (a game, a canvas app) it needs a
  `phone_look` taken on the current capture (every action installs a new
  one, so: look, then tap) that described nothing payment-like. The phone
  enforces the same capture-id rule.
- Matching folds NFKC, strips invisible characters, and maps Cyrillic and
  Greek look-alike letters to Latin, so "Pаy now" spelled with a Cyrillic а
  is still "pay now". A label in a script the lists do not carry is the
  limit; the lists are data and meant to grow.
- A **password field**: never typed into.

A refusal comes back as a result beginning `GUARD:`; when the phone itself
refused with a hand-over, the result says the person has taken over. The
rules file is package data; the app vendors a copy and its CI asserts the
two are identical, so a change here fails the app's bump until it copies it.
