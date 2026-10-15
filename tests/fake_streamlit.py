"""A stand-in for the parts of Streamlit the dashboard uses, so every page can be executed in tests.

It is not a renderer: it records what a page shows and plays back scripted clicks and widget values. Each
function takes the same parameter names as the real Streamlit (>= 1.40) and rejects anything else, so a
misspelt or unsupported argument fails a test instead of failing on the coach's machine.

Usage in a test:
    fake.reset(); fake.click("runner_demo"); fake.run(page.render)
    fake.interact("weight_R1_0_0", 0.2)      # move a slider: sets session state and calls its on_change
"""
from __future__ import annotations

import datetime as _dt
import sys
import types
from typing import Any, Callable

from matplotlib.figure import Figure


class StopRun(Exception):
    pass


class Rerun(StopRun):
    pass


class SwitchPage(StopRun):
    def __init__(self, page):
        super().__init__(getattr(page, "url_path", page))
        self.page = page


class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class _App:
    def __init__(self):
        self.session_state = SessionState()
        self.reset_run()
        self.clicks: set[str] = set()
        self.values: dict[str, Any] = {}           # scripted values for widgets, by key or label
        self.callbacks: dict[str, tuple] = {}

    def reset_run(self):
        self.out: list[tuple[str, Any]] = []       # (kind, payload) in display order
        self.widgets: dict[str, str] = {}


A = _App()
session_state = A.session_state


def reset() -> None:
    global A, session_state
    A = _App()
    session_state = A.session_state
    mod.session_state = session_state


def click(*keys: str) -> None:
    A.clicks.update(keys)


def set_value(key: str, value: Any) -> None:
    A.values[key] = value


def interact(key: str, value: Any) -> None:
    """What happens when the coach moves a widget: its state changes and its callback runs (before the rerun)."""
    A.session_state[key] = value
    if key in A.callbacks:
        fn, args, kwargs = A.callbacks[key]
        fn(*args, **(kwargs or {}))


def run(page: Callable[[], None], max_reruns: int = 8):
    """Run a page like Streamlit does, following st.rerun(). Returns None or the page switched to."""
    for _ in range(max_reruns):
        A.reset_run()
        try:
            page()
            return None
        except Rerun:
            continue
        except SwitchPage as e:
            return e.page
    raise AssertionError("page kept calling st.rerun()")


def texts() -> str:
    return "\n".join(str(p) for k, p in A.out if isinstance(p, str))


def kinds(kind: str) -> list:
    return [p for k, p in A.out if k == kind]


# ---- helpers -------------------------------------------------------------------------------------
def _check(name: str, kwargs: dict, allowed: set[str]) -> None:
    bad = set(kwargs) - allowed
    if bad:
        raise TypeError(f"st.{name}() got unexpected keyword argument(s) {sorted(bad)}")


def _widget(kind: str, label: str, key: str | None, default: Any, on_change=None, args=(), kwargs=None) -> Any:
    if not isinstance(label, str) or label == "":
        raise TypeError(f"st.{kind}() needs a non-empty label (Streamlit warns on empty labels)")
    wkey = key if key is not None else f"{kind}:{label}"
    if wkey in A.widgets:
        raise ValueError(f"duplicate widget key {wkey!r} (Streamlit raises DuplicateWidgetID)")
    A.widgets[wkey] = kind
    if key in A.values or label in A.values:          # scripted input = what the coach typed or picked
        value = A.values[key] if key in A.values else A.values[label]
    elif wkey in A.session_state:
        value = A.session_state[wkey]
    else:
        value = default
    A.session_state[wkey] = value
    if on_change is not None:
        A.callbacks[wkey] = (on_change, tuple(args or ()), kwargs)
    A.out.append((kind, label))
    return value


COMMON = {"key", "help", "on_change", "args", "kwargs", "disabled", "label_visibility"}


# ---- layout --------------------------------------------------------------------------------------
class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        return getattr(mod, name)


def columns(spec, *, gap="small", vertical_alignment="top", border=False):
    n = spec if isinstance(spec, int) else len(spec)
    if n < 1:
        raise ValueError("st.columns needs at least one column")
    return [_Ctx() for _ in range(n)]


def container(*, height=None, border=None, key=None):
    return _Ctx()


def expander(label, expanded=False, *, icon=None):
    A.out.append(("expander", label))
    return _Ctx()


def form(key, clear_on_submit=False, *, enter_to_submit=True, border=True):
    return _Ctx()


def tabs(labels):
    return [_Ctx() for _ in labels]


sidebar = _Ctx()


# ---- text and status -----------------------------------------------------------------------------
def _text(kind):
    def f(body="", *a, **kw):
        _check(kind, kw, {"anchor", "help", "divider", "unsafe_allow_html", "icon", "width"})
        A.out.append((kind, str(body)))
    f.__name__ = kind
    return f


title, header, subheader, caption, markdown, text = (_text(k) for k in ("title", "header", "subheader", "caption", "markdown", "text"))
info, success, warning, error = (_text(k) for k in ("info", "success", "warning", "error"))


def write(*args, **kw):
    for a in args:
        A.out.append(("write", str(a)))


def divider():
    A.out.append(("divider", ""))


def toast(body, *, icon=None, duration="short"):
    A.out.append(("toast", str(body)))


class _Progress:
    def progress(self, value, text=None):
        if not 0 <= value <= 1:
            raise ValueError("progress value must be between 0 and 1")
        A.out.append(("progress", text or ""))
        return self


def progress(value, text=None):
    return _Progress().progress(value, text)


def metric(label, value, delta=None, delta_color="normal", *, help=None, label_visibility="visible", border=False):
    if delta_color not in ("normal", "inverse", "off"):
        raise ValueError("bad delta_color")
    A.out.append(("metric", f"{label}: {value}" + (f" ({delta})" if delta else "")))


def dataframe(data=None, *args, **kw):
    _check("dataframe", kw, {"hide_index", "column_config", "key", "height", "on_select", "selection_mode", "width"})
    A.out.append(("dataframe", data))


def pyplot(fig=None, clear_figure=None, **kw):
    _check("pyplot", kw, set())
    if not isinstance(fig, Figure):
        raise TypeError("st.pyplot expects a matplotlib Figure")
    A.out.append(("pyplot", fig))


def json(body, *, expanded=True):
    A.out.append(("json", body))


def chat_message(name, *, avatar=None):
    if not isinstance(name, str) or not name:
        raise TypeError("st.chat_message needs a name such as 'user' or 'assistant'")
    A.out.append(("chat_message", name))
    return _Ctx()


def chat_input(placeholder="Your message", *, key=None, max_chars=None, disabled=False, on_submit=None,
               args=None, kwargs=None):
    """Returns the submitted text once (like Streamlit), when a test scripted one with set_value(key, text)."""
    wkey = key or "chat_input"
    if wkey in A.widgets:
        raise ValueError(f"duplicate widget key {wkey!r}")
    A.widgets[wkey] = "chat_input"
    A.out.append(("chat_input", placeholder))
    return A.values.pop(wkey, None)


def spinner(text="In progress...", *, show_time=False):
    return _Ctx()


def page_link(page, *, label=None, icon=None, help=None, disabled=False):
    A.out.append(("page_link", label))


def set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None):
    pass


# ---- widgets -------------------------------------------------------------------------------------
def button(label, key=None, help=None, on_click=None, args=None, kwargs=None, *, type="secondary", icon=None,
           disabled=False, use_container_width=None):
    wkey = key or f"button:{label}"
    if wkey in A.widgets:
        raise ValueError(f"duplicate widget key {wkey!r}")
    A.widgets[wkey] = "button"
    A.out.append(("button", label))
    hit = not disabled and (wkey in A.clicks or label in A.clicks)
    if hit:
        A.clicks.discard(wkey)
        A.clicks.discard(label)
        if on_click:
            on_click(*(args or ()), **(kwargs or {}))
    return hit


def form_submit_button(label="Submit", help=None, on_click=None, args=None, kwargs=None, *, type="secondary",
                       icon=None, disabled=False, use_container_width=None):
    return button(label, key=f"submit:{label}", on_click=on_click, args=args, kwargs=kwargs, disabled=disabled)


def download_button(label, data, file_name=None, mime=None, key=None, help=None, on_click="rerun", args=None,
                    kwargs=None, *, type="secondary", icon=None, disabled=False, use_container_width=None):
    if not isinstance(data, (str, bytes)) or not data:
        raise TypeError("download data must be non-empty str or bytes")
    A.out.append(("download", file_name))
    return False


def slider(label, min_value=None, max_value=None, value=None, step=None, format=None, **kw):
    _check("slider", kw, COMMON)
    if value is None or not (min_value <= value <= max_value):
        raise ValueError(f"slider {label!r}: value {value} outside {min_value}..{max_value}")
    v = _widget("slider", label, kw.get("key"), value, kw.get("on_change"), kw.get("args"), kw.get("kwargs"))
    return v


def _choice(kind, label, options, index, format_func, kw):
    opts = list(options)
    for o in opts:
        format_func(o)                       # a broken format_func fails here, as it would in Streamlit
    if index is not None and not 0 <= index < len(opts):
        raise IndexError(f"{kind} {label!r}: index {index} out of range")
    v = _widget(kind, label, kw.get("key"), None if index is None else opts[index], kw.get("on_change"),
                kw.get("args"), kw.get("kwargs"))
    if v is not None and v not in opts:
        raise ValueError(f"{kind} {label!r}: {v!r} is not one of the options")
    return v


def radio(label, options, index=0, format_func=str, **kw):
    _check("radio", kw, COMMON | {"horizontal", "captions"})
    return _choice("radio", label, options, index, format_func, kw)


def selectbox(label, options, index=0, format_func=str, **kw):
    _check("selectbox", kw, COMMON | {"placeholder", "accept_new_options"})
    return _choice("selectbox", label, options, index, format_func, kw)


def multiselect(label, options, default=None, format_func=str, **kw):
    _check("multiselect", kw, COMMON | {"placeholder", "max_selections", "accept_new_options"})
    opts = list(options)
    for d in default or []:
        if d not in opts:
            raise ValueError(f"multiselect {label!r}: default {d!r} not in options")
    return _widget("multiselect", label, kw.get("key"), list(default or []), kw.get("on_change"), kw.get("args"))


def text_input(label, value="", max_chars=None, key=None, type="default", **kw):
    _check("text_input", kw, COMMON | {"placeholder", "autocomplete"})
    return _widget("text_input", label, key, value, kw.get("on_change"), kw.get("args"))


def text_area(label, value="", height=None, max_chars=None, key=None, **kw):
    _check("text_area", kw, COMMON | {"placeholder"})
    return _widget("text_area", label, key, value, kw.get("on_change"), kw.get("args"))


def number_input(label, min_value=None, max_value=None, value="min", step=None, format=None, key=None, **kw):
    _check("number_input", kw, COMMON | {"placeholder", "icon"})
    default = min_value if value == "min" else value
    return _widget("number_input", label, key, default, kw.get("on_change"), kw.get("args"))


def checkbox(label, value=False, key=None, **kw):
    _check("checkbox", kw, COMMON)
    return _widget("checkbox", label, key, value, kw.get("on_change"), kw.get("args"))


def toggle(label, value=False, key=None, **kw):
    _check("toggle", kw, COMMON)
    return _widget("toggle", label, key, value, kw.get("on_change"), kw.get("args"))


def date_input(label, value="today", min_value=None, max_value=None, key=None, **kw):
    _check("date_input", kw, COMMON | {"format"})
    default = _dt.date.today() if value == "today" else value
    return _widget("date_input", label, key, default, kw.get("on_change"), kw.get("args"))


class UploadedFile:
    """What st.audio_input and st.file_uploader return: a name and getvalue()."""

    def __init__(self, data: bytes, name: str = "audio.wav"):
        self.data, self.name = data, name

    def getvalue(self) -> bytes:
        return self.data


def audio_input(label, *, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False,
                label_visibility="visible"):
    """Returns the recording a test scripted with set_value(key, UploadedFile(...)), else None."""
    return _widget("audio_input", label, key, None, on_change, args, kwargs)


def file_uploader(label, type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None,
                  kwargs=None, *, disabled=False, label_visibility="visible"):
    return _widget("file_uploader", label, key, None, on_change, args, kwargs)


def data_editor(data, *, width=None, height=None, use_container_width=None, hide_index=None, column_order=None,
                column_config=None, num_rows="fixed", disabled=False, key=None, on_change=None, args=None,
                kwargs=None, row_height=None):
    if num_rows not in ("fixed", "dynamic"):
        raise ValueError("bad num_rows")
    wkey = key or "data_editor"
    if wkey in A.widgets:
        raise ValueError(f"duplicate widget key {wkey!r}")
    A.widgets[wkey] = "data_editor"
    A.out.append(("data_editor", data))
    return A.values.get(wkey, data.copy())


column_config = types.SimpleNamespace(
    TextColumn=lambda label=None, **kw: {"type": "text", **kw},
    NumberColumn=lambda label=None, **kw: {"type": "number", **kw},
)


# ---- execution control ---------------------------------------------------------------------------
def rerun(*, scope="app"):
    raise Rerun()


def switch_page(page):
    raise SwitchPage(page)


def cache_resource(func=None, *, ttl=None, max_entries=None, show_spinner=True, validate=None, hash_funcs=None):
    def deco(f):
        cache = {}

        def wrapper(*a):
            if a not in cache:
                cache[a] = f(*a)
            return cache[a]
        wrapper.clear = cache.clear
        return wrapper
    return deco(func) if func else deco


def fragment(func=None, *, run_every=None):
    def deco(f):
        f.run_every = run_every
        return f
    return deco(func) if func else deco


class Page:
    def __init__(self, page, *, title=None, icon=None, url_path=None, default=False):
        self.page, self.title, self.icon, self.url_path, self.default = page, title, icon, url_path, default

    def run(self):
        self.page()


class _Nav:
    def __init__(self, pages):
        self.pages = [p for group in (pages.values() if isinstance(pages, dict) else [pages]) for p in group]

    def run(self):
        next((p for p in self.pages if p.default), self.pages[0]).run()


def navigation(pages, *, position="sidebar", expanded=False):
    return _Nav(pages)


mod = sys.modules[__name__]
