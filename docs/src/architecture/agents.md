# Keystone Agents

Keystone agents are defined using the Claude Agent SDK and orchestrate the classification workflow. Unlike the CrewAI-based Agent Studio, Atelier uses Claude as both the agent runtime and the SDK for programmatic control.

## Agent Types

- **Classifier** — Drives the classification pipeline against a taxonomy
- **Evidence Fuser** — Combines signals from multiple evidence sources using Dempster-Shafer theory
- **Visualization Director** — Curates embedding-atlas views for different personas

## Evolution

Keystone agents are designed to evolve alongside the classification workflow. As the signals pipeline produces new parquet outputs, agents adapt their strategies based on uncertainty metrics and evidence quality.
