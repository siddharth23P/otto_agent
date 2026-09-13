# Security

Otto runs shell commands, edits files and drives a browser on the machine it
is installed on. The workspace boundary confines the file tools; the shell is
not confined, and the README says so. Please read that section before
running it against anything you have not committed.

## Reporting a vulnerability

Report privately through
[GitHub's security advisory form](https://github.com/siddharth23P/otto_agent/security/advisories/new)
rather than a public issue. Include what you ran, what happened, and why it
matters. You will get an acknowledgement within a few days.

Issues of this kind are in scope: a tool reaching outside the workspace root,
a URL check that can be bypassed, a prompt that lets a page steer the next
request, secrets written anywhere but `.env`.
