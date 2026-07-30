"""HTML representation of TuiML components for notebooks.

Renders a component — most usefully a :class:`~tuiml.workflow.Workflow` — as a
nested diagram of boxes, so a pipeline's structure is visible at a glance
instead of buried in a long ``repr`` string.

Any object may control how it is drawn by implementing
``_tuiml_visual_block_()`` and returning a :class:`VisualBlock`. Composite
components (a pipeline, a column router) use this to expose their children,
which are then drawn recursively.

The diagram is pure HTML + CSS — collapsible boxes use a hidden checkbox
toggled by its ``<label>``, so no JavaScript is needed and the output renders
in any notebook front end. The CSS is scoped to a unique container id so
several diagrams on one page cannot affect each other.
"""

import html
import itertools
from string import Template
from typing import Any, List, Optional

_CONTAINER_IDS = itertools.count()
_ELEMENT_IDS = itertools.count()


class VisualBlock:
    """How a component should be drawn.

    Parameters
    ----------
    kind : {"single", "serial", "parallel"}
        ``"single"`` draws one box. ``"serial"`` stacks children vertically
        (data flows top to bottom). ``"parallel"`` places children side by
        side (children each see the same input).
    items : object or list of object
        The component itself when ``kind="single"``, otherwise its children.
    names : str or list of str, optional
        Box captions — a single name for ``"single"``, one per child otherwise.
    details : str or list of str, optional
        Text revealed when a box is expanded (typically the child's ``repr``).
    title : str, optional
        Caption for the frame drawn around the children. Only used when
        ``kind`` is ``"serial"`` or ``"parallel"``.
    framed : bool, default=True
        Whether to draw a dashed frame around the children.
    """

    def __init__(
        self,
        kind: str,
        items: Any,
        *,
        names: Any = None,
        details: Any = None,
        title: Optional[str] = None,
        framed: bool = True,
    ):
        if kind not in ("single", "serial", "parallel"):
            raise ValueError(
                f"kind must be 'single', 'serial', or 'parallel', got {kind!r}."
            )
        self.kind = kind
        self.items = items
        self.title = title
        self.framed = framed

        if kind in ("serial", "parallel"):
            n = len(items)
            if names is None:
                names = [None] * n
            if details is None:
                details = [None] * n
        self.names = names
        self.details = details


def get_visual_block(component: Any) -> VisualBlock:
    """Return the :class:`VisualBlock` describing how to draw ``component``.

    Parameters
    ----------
    component : object
        Any component. Objects implementing ``_tuiml_visual_block_()``
        control their own layout; anything else is drawn as a single box.

    Returns
    -------
    VisualBlock
        The layout description.
    """
    if hasattr(component, "_tuiml_visual_block_"):
        try:
            return component._tuiml_visual_block_()
        except Exception:
            pass  # fall through to a plain single box

    if component is None or isinstance(component, str):
        label = "passthrough" if component is None else component
        return VisualBlock("single", component, names=label, details=label)

    return VisualBlock(
        "single",
        component,
        names=type(component).__name__,
        details=repr(component),
    )


def _write_box(out: List[str], name: str, details: Optional[str], *,
               outer_class: str, inner_class: str, expanded: bool) -> None:
    """Append one labelled box, collapsible when it has details to show."""
    name = html.escape(str(name))
    out.append(f'<div class="{outer_class}"><div class="{inner_class} tuiml-toggleable">')
    if details is None:
        out.append(f"<label>{name}</label>")
    else:
        eid = f"tuiml-el-{next(_ELEMENT_IDS)}"
        checked = " checked" if expanded else ""
        out.append(
            f'<input class="tuiml-toggle-input" type="checkbox" id="{eid}"{checked}>'
            f'<label class="tuiml-toggle-label" for="{eid}">{name}</label>'
            f'<div class="tuiml-toggle-content"><pre>{html.escape(str(details))}</pre></div>'
        )
    out.append("</div></div>")


def _write_component(out: List[str], component: Any, name: Optional[str],
                     details: Optional[str], *, first_call: bool = False) -> None:
    """Recursively append the HTML for ``component`` and its children."""
    block = get_visual_block(component)

    if block.kind == "single":
        _write_box(
            out, block.names, block.details,
            outer_class="tuiml-item",
            inner_class="tuiml-estimator",
            expanded=first_call,
        )
        return

    # Composite: an optional caption, then the children.
    frame = " tuiml-framed" if block.framed else ""
    out.append(f'<div class="tuiml-group{frame}">')
    title = block.title if block.title is not None else name
    if title is not None:
        _write_box(
            out, title, None,
            outer_class="tuiml-label-container",
            inner_class="tuiml-label",
            expanded=False,
        )

    out.append(f'<div class="tuiml-{block.kind}">')
    for child, child_name, child_details in zip(block.items, block.names, block.details):
        if block.kind == "serial":
            _write_component(out, child, child_name, child_details)
        else:
            out.append('<div class="tuiml-parallel-item">')
            # A caption above each branch, then the branch itself.
            if child_name is not None:
                _write_box(
                    out, child_name, child_details,
                    outer_class="tuiml-label-container",
                    inner_class="tuiml-label",
                    expanded=False,
                )
            _write_component(out, child, None, None)
            out.append("</div>")
    out.append("</div></div>")


def component_html_repr(component: Any) -> str:
    """Build the HTML diagram for a component.

    Parameters
    ----------
    component : object
        The component to draw (typically a :class:`~tuiml.workflow.Workflow`).

    Returns
    -------
    str
        Self-contained HTML: a scoped ``<style>`` block, a plain-text
        fallback for front ends that strip CSS, and the diagram itself.

    Examples
    --------
    >>> from tuiml import Workflow
    >>> html_str = component_html_repr(Workflow(["StandardScaler", "NaiveBayesClassifier"]))
    >>> html_str.startswith("<style>")
    True
    """
    container_id = f"tuiml-container-{next(_CONTAINER_IDS)}"
    style = Template(_CSS).substitute(id=container_id)

    fitted = " tuiml-fitted" if getattr(component, "_is_fitted", False) else ""
    status = "fitted" if fitted else "not fitted"

    out: List[str] = [
        f"<style>{style}</style>",
        f'<div id="{container_id}" class="tuiml-top">',
        # Shown only when the CSS is stripped (e.g. an untrusted notebook),
        # since the stylesheet hides it and reveals the diagram instead.
        f'<div class="tuiml-fallback"><pre>{html.escape(repr(component))}</pre></div>',
        '<div class="tuiml-container" hidden>',
        f'<div class="tuiml-status{fitted}">{status}</div>',
    ]
    _write_component(out, component, type(component).__name__, repr(component),
                     first_call=True)
    out.append("</div></div>")
    return "".join(out)


# CSS is templated on $id so every diagram is scoped to its own container and
# cannot leak styles into the surrounding notebook.
_CSS = """
#$id {
  --tuiml-line: #b0b7c3;
  --tuiml-border: #b0b7c3;
  --tuiml-bg: #fff;
  --tuiml-box: #f4f6f9;
  --tuiml-box-hover: #e6ebf2;
  --tuiml-text: #1c1e21;
  --tuiml-accent: #1668c1;
  color: var(--tuiml-text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 1.3;
}
@media (prefers-color-scheme: dark) {
  #$id {
    --tuiml-line: #5c6470;
    --tuiml-border: #5c6470;
    --tuiml-bg: #1e2126;
    --tuiml-box: #2a2f37;
    --tuiml-box-hover: #353b45;
    --tuiml-text: #e6e6e6;
    --tuiml-accent: #6bb0ff;
  }
}
#$id .tuiml-fallback { display: none; }
#$id .tuiml-container {
  display: inline-block;
  position: relative;
  background: var(--tuiml-bg);
  padding: 0.4em;
  border-radius: 4px;
  box-sizing: border-box;
}
#$id pre { margin: 0; padding: 0; white-space: pre-wrap; overflow-x: auto; }
#$id .tuiml-status {
  font-size: 0.8em;
  color: var(--tuiml-line);
  text-align: right;
  padding-bottom: 0.2em;
}
#$id .tuiml-status.tuiml-fitted { color: #1a7f37; }
#$id .tuiml-group { display: flex; flex-direction: column; align-items: stretch; }
#$id .tuiml-framed {
  border: 1px dashed var(--tuiml-border);
  border-radius: 4px;
  padding: 0.4em;
  margin: 0.25em;
  box-sizing: border-box;
}
/* Serial: children stacked vertically, joined by a vertical connector. */
#$id .tuiml-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  padding: 0 0.6em;
}
#$id .tuiml-serial::before {
  content: "";
  position: absolute;
  top: 0; bottom: 0; left: 50%;
  border-left: 2px solid var(--tuiml-line);
  z-index: 0;
}
/* Parallel: branches side by side under a shared horizontal connector. */
#$id .tuiml-parallel {
  display: flex;
  flex-direction: row;
  align-items: stretch;
  position: relative;
}
#$id .tuiml-parallel::before {
  content: "";
  position: absolute;
  top: 0; left: 6%; right: 6%;
  border-top: 2px solid var(--tuiml-line);
  z-index: 0;
}
#$id .tuiml-parallel-item {
  display: flex;
  flex-direction: column;
  position: relative;
  padding: 0 0.2em;
}
#$id .tuiml-parallel-item::before {
  content: "";
  position: absolute;
  top: 0; height: 0.5em; left: 50%;
  border-left: 2px solid var(--tuiml-line);
  z-index: 0;
}
#$id .tuiml-item, #$id .tuiml-label-container { position: relative; z-index: 1; }
#$id .tuiml-label-container { text-align: center; }
#$id .tuiml-label { font-weight: 600; padding: 0.15em 0; }
#$id .tuiml-label label { cursor: default; }
#$id .tuiml-estimator {
  background: var(--tuiml-box);
  border: 1px solid var(--tuiml-border);
  border-radius: 4px;
  margin: 0.25em 0;
  box-sizing: border-box;
}
#$id .tuiml-estimator:hover { background: var(--tuiml-box-hover); }
#$id .tuiml-estimator > label,
#$id .tuiml-estimator > .tuiml-toggle-label { padding: 0.3em 0.6em; display: block; }
/* Collapsible boxes: a visually hidden checkbox driven by its label. */
#$id .tuiml-toggle-input {
  position: absolute;
  width: 1px; height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}
#$id .tuiml-toggle-label { cursor: pointer; user-select: none; }
#$id .tuiml-toggle-label::before {
  content: "\\25b8";
  color: var(--tuiml-accent);
  padding-right: 0.4em;
}
#$id .tuiml-toggle-input:checked + .tuiml-toggle-label::before { content: "\\25be"; }
#$id .tuiml-toggle-input:focus-visible + .tuiml-toggle-label {
  outline: 2px solid var(--tuiml-accent);
  outline-offset: 1px;
}
#$id .tuiml-toggle-content { display: none; }
#$id .tuiml-toggle-input:checked ~ .tuiml-toggle-content {
  display: block;
  padding: 0.4em 0.6em;
  border-top: 1px solid var(--tuiml-border);
  font-size: 0.92em;
  max-height: 20em;
  overflow: auto;
}
"""
