'use strict';

// Compartmental-model examples for infectious disease modeling students.
// Picking one from the "Example" dropdown replaces the editor contents.
(function (document) {
  const examples = {
    "SIR": `digraph {
  graph [rankdir = LR]
  node [shape = box, fixedsize = true,
        width = 1, height = 1]
  S -> I [label = "λS"]
  I -> R [label = "νI"]
}`,

    "SIS": `digraph {
  graph [rankdir = LR]
  node [shape = box, fixedsize = true,
        width = 1, height = 1]
  S -> I [label = "λS"]
  I -> S [label = "νI"]
}`,

    "SEIR": `digraph {
  graph [rankdir = LR]
  node [shape = box, fixedsize = true,
        width = 1, height = 1]
  S -> E [label = "λS"]
  E -> I [label = "σE"]
  I -> R [label = "νI"]
}`,

    "SIR with births and deaths": `digraph {
  graph [rankdir = LR]
  node [shape = box, fixedsize = true,
        width = 1, height = 1]
  S -> I [label = "λS"]
  I -> R [label = "νI"]

  // births flow in from the left, deaths flow out below each box
  node [shape = none, width = 0, height = 0, label = ""]
  birth -> S [label = "μN"]
  S -> deathS [label = "μS"]
  I -> deathI [label = "μI"]
  R -> deathR [label = "μR"]
  { rank = same; S; deathS }
  { rank = same; I; deathI }
  { rank = same; R; deathR }
}`,

    "SIR with vaccination": `digraph {
  graph [rankdir = LR]
  node [shape = box, fixedsize = true,
        width = 1, height = 1]
  S -> I [label = "λS"]
  I -> R [label = "νI"]
  S -> R [label = "ρS"]
}`,
  };

  const select = document.querySelector("#examples select");
  select.addEventListener("change", () => {
    const source = examples[select.value];
    if (source === undefined) { return; }
    GraphvizOnline.editor.getSession().setValueAndRenderGraphImmediately(source);
  });
})(document);
