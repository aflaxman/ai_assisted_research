# Graphviz Online for compartmental models

A copy of [dreampuf/GraphvizOnline](https://github.com/dreampuf/GraphvizOnline)
(BSD-3, see `LICENSE`) that opens on an SIR diagram instead of the stock
"process #1 / process #2" example, for infectious disease modeling students.

Live page: https://aflaxman.github.io/ai_assisted_research/graphviz/

The default example:

```dot
digraph {
  graph [rankdir = LR]
  node [shape = box, fixedsize = true,
        width = 1, height = 1]
  S -> I [label = "λS"]
  I -> R [label = "νI"]
}
```

## What changed from upstream

- `index.html`: the editor starts with the SIR digraph; the page title and
  the GitHub corner link point here; an **Example** dropdown sits first in
  the options bar.
- `examples.js` (new): the dropdown's five models (SIR, SIS, SEIR, SIR with
  births and deaths, SIR with vaccination). Add a model by adding an entry
  to the `examples` object and an `<option>` to `index.html`.
- `ace/`: trimmed from the full ACE editor bundle (8.6 MB) to the four
  modules this page loads (editor, DOT mode, Twilight theme, search box).

`main.js`, `style.css`, `viz-global.js`, `svg-pan-zoom.min.js`, and
`hammer.min.js` are unmodified copies of upstream commit `ecaf161`.

## Sharing a diagram

Everything upstream supports still works: the **Share** button, and the
`?url=`, `?presentation=`, and `#<url-encoded source>` query forms described
in the [upstream README](https://github.com/dreampuf/GraphvizOnline#readme).

## Making this a proper fork

This copy lives in `graphviz/` at the repository root, which is where
GitHub Pages for this repository serves from. To have it at `aflaxman.github.io/GraphvizOnline` instead:

1. Fork https://github.com/dreampuf/GraphvizOnline on GitHub.
2. Copy `index.html` and `examples.js` from this directory over the fork's.
3. In the fork, Settings → Pages → Build and deployment → Branch: `master`,
   folder `/ (root)`.
