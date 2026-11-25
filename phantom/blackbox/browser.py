"""BLACKBOX safe browser (Days 21-22).

Playwright (headless Chromium) wrapped in hard safety limits:
- allowlist: only the local origin it was opened on (127.0.0.1); every other
  request is aborted and logged as a blocked request;
- downloads are never accepted; file links (.apk, .pdf...) are never opened;
- forms are filled with DUMMY data only and only submitted to the allowlisted
  origin (a form posting elsewhere is observed, never submitted);
- per-action timeout and a hard action budget enforced by the explorer.

It captures page state (URL, title, visible text, links with their page region,
forms with field types/labels, countdown/overlay elements, meta refresh) as a
structured PageState, plus screenshots on request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlencode, urlparse

ACTION_TIMEOUT_MS = 5000
FILE_EXTENSIONS = (".apk", ".exe", ".msi", ".bat", ".scr", ".dmg", ".pdf", ".zip", ".doc", ".docx")
DUMMY = {"email": "test.user@example.test", "tel": "9000000000", "password": "Dummy#Pass1",
         "date": "1990-01-01", "number": "123456", "text": "Test User"}

_EXTRACT_JS = r"""() => {
  const abs = h => { try { return new URL(h, location.href).href } catch (e) { return h } };
  const region = a => a.closest('nav') ? 'nav' : a.closest('footer') ? 'footer'
                    : (a.classList.contains('btn') ? 'button' : 'body');
  return {
    url: location.href, title: document.title,
    text: document.body ? document.body.innerText.slice(0, 5000) : '',
    links: [...document.querySelectorAll('a[href]')].map(a => ({
      text: a.innerText.trim(), href: abs(a.getAttribute('href')), region: region(a),
      download: a.hasAttribute('download')})),
    forms: [...document.forms].map(f => ({
      action: abs(f.getAttribute('action') || location.href), method: (f.method || 'get'),
      submit: ((f.querySelector('button,[type=submit]') || {}).innerText || '').trim(),
      fields: [...f.querySelectorAll('input,select,textarea')].filter(i => i.type !== 'submit')
        .map(i => ({name: i.name || '', type: i.type || 'text',
                    label: ((i.closest('label') || {}).innerText
                            || i.placeholder || '').trim()}))})),
    countdown: !!document.querySelector('.countdown,[data-deadline]'),
    overlay: [...document.querySelectorAll('.overlay,.alert,.modal,[role=alertdialog]')]
               .map(e => e.innerText).join(' '),
    meta_refresh: (document.querySelector('meta[http-equiv=refresh]') || {}).content || null,
  };
}"""


@dataclass
class PageState:
    url: str
    requested_url: str
    title: str
    text: str
    links: list[dict]
    forms: list[dict]
    countdown: bool
    overlay: str
    meta_refresh: str | None
    redirected_from: str | None = None

    @property
    def path(self) -> str:
        return urlparse(self.url).path


@dataclass
class Action:
    kind: str  # link | form | back
    target: str  # absolute URL to load
    text: str = ""
    region: str = ""
    form: dict | None = None

    def key(self) -> tuple:
        return (self.kind, self.target)


@dataclass
class SafeBrowser:
    origin: str  # e.g. http://127.0.0.1:53121
    blocked: list[str] = field(default_factory=list)

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._ctx = self._browser.new_context(accept_downloads=False,
                                              user_agent="PHANTOM-BLACKBOX/1.0 (local test only)")
        self._ctx.set_default_timeout(ACTION_TIMEOUT_MS)
        self._ctx.route("**/*", self._guard)
        self.page = self._ctx.new_page()
        return self

    def __exit__(self, *exc):
        self._ctx.close()
        self._browser.close()
        self._pw.stop()

    def _guard(self, route):
        url = route.request.url
        if self.allowed(url):
            return route.continue_()
        self.blocked.append(url)
        return route.abort()

    def allowed(self, url: str) -> bool:
        u = urlparse(url)
        return f"{u.scheme}://{u.netloc}" == self.origin

    def load(self, url: str) -> PageState:
        if not self.allowed(url):
            raise PermissionError(f"blocked: {url} is outside the allowlist")
        for attempt in range(3):
            try:
                self.page.goto(url, wait_until="load")
                break
            except Exception as e:  # noqa: BLE001
                # A redirect still in flight from the previous page can interrupt us.
                if "interrupted by another navigation" not in str(e) or attempt == 2:
                    raise
                self.page.wait_for_load_state("load")
        data = self._extract()
        if data["meta_refresh"] and data["url"] == url:
            try:  # an instant meta-refresh redirect: wait for it, then read the new page
                self.page.wait_for_url(lambda u: u != url, timeout=2000)
                self.page.wait_for_load_state("load")
                data = self._extract()
            except Exception:  # noqa: BLE001 - no redirect within 2 s
                pass
        state = PageState(requested_url=url, **data)
        if urlparse(state.url).path != urlparse(url).path:
            state.redirected_from = url
        return state

    def _extract(self) -> dict:
        """Read the DOM; retry if a navigation (e.g. a redirect) replaces the page mid-read."""
        for _ in range(4):
            try:
                return self.page.evaluate(_EXTRACT_JS)
            except Exception as e:  # noqa: BLE001
                if "context was destroyed" not in str(e) and "navigat" not in str(e):
                    raise
                self.page.wait_for_load_state("load")
        return self.page.evaluate(_EXTRACT_JS)

    def screenshot(self) -> bytes:
        return self.page.screenshot(full_page=False)

    def actions(self, state: PageState) -> list[Action]:
        """Permitted next actions: same-origin page links and same-origin form submits."""
        acts, seen = [], set()
        for lk in state.links:
            href = lk["href"].split("#")[0]
            if not self.allowed(href) or href.lower().endswith(FILE_EXTENSIONS):
                continue  # external or a file: observe only, never open
            a = Action("link", href, lk["text"], lk["region"])
            if a.key() not in seen and href != state.url:
                seen.add(a.key())
                acts.append(a)
        for f in state.forms:
            if not self.allowed(f["action"]):
                continue  # posting data elsewhere: observe only, never submit
            query = urlencode({fl["name"]: DUMMY.get(fl["type"], DUMMY["text"])
                               for fl in f["fields"] if fl["name"]})
            target = f["action"].split("?")[0] + ("?" + query if query else "")
            acts.append(Action("form", target, f["submit"], "form", form=f))
        return acts
