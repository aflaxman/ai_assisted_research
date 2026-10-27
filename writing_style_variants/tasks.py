"""Four writing tasks that mirror the eval cases described in vivarium-suite PR #304.

Every task is self-contained: the prompt carries the source (a diff, a report).
`facts` are claims the source supports; `traps` are claims it does not support.
The blind judge sees both lists. The writer sees only `prompt`.
"""

TASKS = {}

TASKS["pr_description"] = {
    "title": "PR description for a retry-default change",
    "kind": "pr_description",
    "prompt": '''Write the pull request description for this change. The repository is `tinyhttp`, a small HTTP client library. The PR title is already set: "Turn retries on by default (TH-212)". Write only the description body.

Ticket TH-212 reads, in full:

    Summary: Make retries on by default
    Description: (none)

Diff:

```diff
diff --git a/tinyhttp/client.py b/tinyhttp/client.py
--- a/tinyhttp/client.py
+++ b/tinyhttp/client.py
@@ -12,18 +12,31 @@ class Client:
-    def __init__(self, base_url: str, timeout: float = 10.0, max_retries: int = 0):
+    def __init__(
+        self,
+        base_url: str,
+        timeout: float = 10.0,
+        max_retries: int = 3,
+        backoff: float = 0.5,
+    ):
         self.base_url = base_url
         self.timeout = timeout
         self.max_retries = max_retries
+        self.backoff = backoff

     def request(self, method: str, path: str, **kwargs) -> Response:
-        return self._send(method, path, **kwargs)
+        attempt = 0
+        while True:
+            try:
+                resp = self._send(method, path, **kwargs)
+            except ConnectionError:
+                if attempt >= self.max_retries:
+                    raise
+            else:
+                if resp.status not in (502, 503, 504) or attempt >= self.max_retries:
+                    return resp
+            time.sleep(self.backoff * (2 ** attempt))
+            attempt += 1
diff --git a/tests/test_client.py b/tests/test_client.py
--- a/tests/test_client.py
+++ b/tests/test_client.py
@@ -40,9 +40,21 @@
-def test_no_retry_by_default(flaky_server):
-    client = Client(flaky_server.url)
-    with pytest.raises(ConnectionError):
-        client.request("GET", "/")
+def test_three_retries_by_default(flaky_server):
+    client = Client(flaky_server.url)
+    resp = client.request("GET", "/")
+    assert resp.status == 200
+    assert flaky_server.calls == 4
+
+
+def test_retries_post(flaky_server):
+    client = Client(flaky_server.url)
+    client.request("POST", "/orders", json={"id": 1})
+    assert flaky_server.calls == 4
```
''',
    "facts": [
        "The default for `max_retries` changes from 0 to 3.",
        "A new `backoff` parameter (default 0.5 seconds) sets the wait before a retry, and the wait doubles on each attempt (0.5 s, 1 s, 2 s).",
        "A retry happens on `ConnectionError` and on 502, 503, and 504 responses. Other responses return at once.",
        "Retries apply to every HTTP method, including POST (`test_retries_post`), so a non-idempotent request can be sent up to 4 times.",
        "Existing callers that did not pass `max_retries` now get up to 4 attempts and up to 3.5 s of added wait on failure.",
        "The ticket gives no reason for the change.",
    ],
    "traps": [
        "States a reason or motivation for the change (for example 'to improve reliability') as fact.",
        "Claims retries are limited to idempotent methods or to GET.",
        "Claims 4xx responses, or 500 responses, are retried.",
        "Claims the retry count or backoff is configurable through anything other than the constructor arguments.",
    ],
}

TASKS["jira_ticket"] = {
    "title": "Jira bug ticket for a config loader strict-mode default",
    "kind": "ticket",
    "prompt": '''Write a Jira bug ticket from the report below. Use these sections: Summary (one line), Environment, Steps to reproduce, Expected result, Actual result, Notes.

Report from a user in #help-cfgload:

> After we upgraded cfgload from 2.3.0 to 2.4.0, our deploy job fails on start.
> `cfgload.load("deploy.yaml")` raises `KeyError: 'unknown key: feature_flags'`.
> deploy.yaml has a top-level `feature_flags:` block that our own plugin reads.
> The file has no `strict` key. We never pass `strict=` to `load()`.
> In 2.3.0 the same file loaded fine. Python 3.12, Ubuntu 22.04.

Traceback they pasted:

    Traceback (most recent call last):
      File "/app/deploy.py", line 14, in <module>
        cfg = cfgload.load("deploy.yaml")
      File "/usr/lib/python3.12/site-packages/cfgload/__init__.py", line 41, in load
        raise KeyError(f"unknown key: {sorted(unknown)[0]}")
    KeyError: 'unknown key: feature_flags'

cfgload 2.4.0 CHANGELOG entry:

    - `load()` now defaults to strict mode (#88). Unknown top-level keys raise `KeyError`.
      Pass `strict=False` to keep the old behavior.
''',
    "facts": [
        "The summary names `cfgload.load`, the `KeyError`, and the unknown key or the 2.4.0 strict default.",
        "The file loaded in 2.3.0 and fails in 2.4.0.",
        "Environment: cfgload 2.4.0, Python 3.12, Ubuntu 22.04.",
        "Steps: a config file with a top-level key that is not a known key (`feature_flags`), no `strict` key in the file, and a call to `load()` without `strict=`.",
        "Expected: the file loads as in 2.3.0. Actual: `KeyError: 'unknown key: feature_flags'` raised from `cfgload/__init__.py` line 41.",
        "The 2.4.0 changelog says strict mode is now the default (#88) and that `strict=False` keeps the old behavior.",
    ],
    "traps": [
        "States as fact whether the new behavior is a bug or intended (the changelog shows it is intended; whether it should change is a judgment).",
        "Invents details the report does not give (plugin name, cfgload install method, other OS versions, other affected files).",
        "Claims the user already tried `strict=False` or any other workaround.",
        "Proposes a specific code fix in cfgload as if it were agreed.",
    ],
}

TASKS["chat_reply"] = {
    "title": "Chat reply: what the change does and what could break",
    "kind": "chat",
    "prompt": '''Here is a diff from the `cfgload` library. Explain what this change does and what could break for people who already use the config loader.

```diff
diff --git a/cfgload/__init__.py b/cfgload/__init__.py
--- a/cfgload/__init__.py
+++ b/cfgload/__init__.py
@@ -20,14 +20,14 @@ KNOWN_KEYS = {"name", "version", "env", "services", "secrets", "strict"}

-def load(path: str, strict: bool | None = None) -> dict:
-    """Load a YAML config. Unknown top-level keys raise KeyError when strict."""
+def load(path: str, strict: bool | None = True) -> dict:
+    """Load a YAML config. Unknown top-level keys raise KeyError unless strict is False."""
     data = _read_yaml(path)
     if strict is None:
-        strict = bool(data.get("strict", False))
+        strict = bool(data.get("strict", True))
     if strict:
         unknown = set(data) - KNOWN_KEYS
         if unknown:
             raise KeyError(f"unknown key: {sorted(unknown)[0]}")
     return data
```
''',
    "facts": [
        "The default of the `strict` parameter changes from `None` to `True`.",
        "When `strict` is `None`, the fallback for a file without a `strict` key changes from `False` to `True`.",
        "With the new default, a file's own `strict: false` key is ignored unless the caller passes `strict=None`, because the parameter default is `True`, not `None`.",
        "Callers that never passed `strict` now get a `KeyError` on any unknown top-level key (typos, custom keys, plugin keys).",
        "Passing `strict=False` keeps the old behavior.",
        "The error names only the first unknown key in sorted order.",
    ],
    "traps": [
        "Claims a file with `strict: false` still loads in non-strict mode by default.",
        "Claims `strict=False` was removed or no longer works.",
        "States a reason for the change as fact (the diff gives none).",
        "Claims the set of known keys changed.",
    ],
}

TASKS["doc_review"] = {
    "title": "Documentation review with planted issues",
    "kind": "review",
    "prompt": '''You are reviewing the documentation in this pull request to the `cfgload` library. Return a numbered list of findings. For each finding give the file and lines, what is wrong, and the fix. If the documentation is correct, say so.

```diff
diff --git a/cfgload/__init__.py b/cfgload/__init__.py
--- a/cfgload/__init__.py
+++ b/cfgload/__init__.py
@@ -1,4 +1,4 @@
-__version__ = "2.4.0"
+__version__ = "2.4.1"
@@ -36,3 +36,30 @@ def load(path: str, strict: bool | None = True) -> dict:
     return data
+
+
+def load_many(paths: list[str], strict: bool | None = True, skip_missing: bool = False) -> dict[str, dict]:
+    """Load several YAML config files.
+
+    Args:
+        paths: Paths to the config files, in load order.
+        strict: Passed to ``load`` for each file.
+        ignore_missing: If True, skip paths that do not exist instead of raising.
+
+    Returns:
+        A list of dicts, one per path, in the same order as ``paths``.
+    """
+    out: dict[str, dict] = {}
+    for path in paths:
+        if not os.path.exists(path):
+            if skip_missing:
+                continue
+            raise FileNotFoundError(path)
+        out[path] = load(path, strict=strict)
+    return out
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -30,6 +30,14 @@ cfg = cfgload.load("app.yaml", strict=False)
+### Loading several files
+
+```python
+configs = cfgload.load_many(["base.yaml", "prod.yaml"], ignore_missing=True)
+for cfg in configs:
+    print(cfg["name"])
+```
+
diff --git a/CHANGELOG.md b/CHANGELOG.md
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -1,3 +1,7 @@
+## 2.5.0
+
+- Add `load_many()` to load several config files in one call.
+
 ## 2.4.0
```
''',
    "facts": [
        "The docstring `Returns` section says a list of dicts, but the function returns a dict keyed by path.",
        "The docstring documents a parameter `ignore_missing`, but the parameter is named `skip_missing`.",
        "The README example passes `ignore_missing=True`, which is not a parameter of `load_many` and fails with `TypeError`.",
        "The README example iterates `for cfg in configs` and reads `cfg['name']`; iterating the returned dict yields path strings, so this fails.",
        "The docstring has no `Raises` section for the `FileNotFoundError` raised when `skip_missing` is False.",
        "The CHANGELOG heading says 2.5.0 but `__version__` is bumped to 2.4.1, so the two disagree.",
    ],
    "traps": [
        "Reports a problem that is not in the diff (for example a missing `import os`) as a documentation finding.",
        "Says the documentation is correct.",
        "Claims the function returns a list.",
    ],
}

ORDER = list(TASKS)
