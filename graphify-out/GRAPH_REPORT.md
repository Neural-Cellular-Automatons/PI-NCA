# Graph Report - .  (2026-06-16)

## Corpus Check
- Corpus is ~746 words - fits in a single context window. You may not need a graph.

## Summary
- 10 nodes · 11 edges · 3 communities (1 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_DeepFlux NCA Core|DeepFlux NCA Core]]
- [[_COMMUNITY_Heat Equation Simulation|Heat Equation Simulation]]

## God Nodes (most connected - your core abstractions)
1. `HeatEquationSolver` - 4 edges
2. `DeepFluxNCA` - 3 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (3 total, 2 thin omitted)

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HeatEquationSolver` connect `Heat Equation Simulation` to `DeepFlux NCA Core`, `PI-NCA State & Energy`?**
  _High betweenness centrality (0.472) - this node is a cross-community bridge._
- **Why does `DeepFluxNCA` connect `DeepFlux NCA Core` to `PI-NCA State & Energy`?**
  _High betweenness centrality (0.306) - this node is a cross-community bridge._