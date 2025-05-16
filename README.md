# Crossword Solver Comparative Analysis

This repository contains the codebase for **crossword solvers** implemented using two distinct algorithmic approaches:

1. **[LBP (Loopy Belief Propagation)](https://web.stanford.edu/class/cs224n/final-reports/256725260.pdf)**
2. **[SweepClip Algorithm](https://arxiv.org/pdf/2406.09043)**

## Overview

The core logic for both solvers was already implemented in prior work:

- The original **LBP** implementation is available [here](https://web.stanford.edu/class/cs224n/final-reports/256725260.pdf).
- The original **SweepClip** implementation is available [here](https://arxiv.org/pdf/2406.09043).

In this repository, we focus on a **comparative analysis** of their performance and accuracy.

## Objectives

- Evaluate and compare the effectiveness of LBP and SweepClip in solving crossword puzzles.
- Measure and report the solver accuracy for each approach.

## Results

After conducting a series of evaluations:

- **SweepClip Algorithm** achieved an accuracy of approximately **87%**
- **LBP (Loopy Belief Propagation)** achieved an accuracy of approximately **68%**

These results highlight that the SweepClip algorithm significantly outperforms LBP for the given dataset and problem setup.
