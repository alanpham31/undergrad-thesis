# Olfactory Perception and Molecular Topologies: A Data Analysis Project

## Overview
This repository contains the dataset and analytical frameworks for investigating the complex relationship between molecular properties (specifically, vibrational structures) and human olfactory perception. Inspired by high-dimensional sensory inference, this project aims to map how physical chemical attributes translate into subjective human scent experiences.

By bridging molecular data (Infrared spectra) with human psychophysical ratings, this project enables advanced statistical modeling, machine learning, and network analysis to uncover the foundational structures of smell.

## Datasets

The project utilizes a comprehensive collection of datasets that capture molecules, human subjects, and perceptual ratings:

### 1. Perceptual Data
*   **`keller_ratings_long.csv`**: A comprehensive, long-format dataset containing individual perceptual ratings from subjects for various chemical stimuli. 
*   **`keller_qualities.csv`**: Contains scent descriptor data (e.g., "sweet", "garlic", "flower", "decayed"). Ideal for creating semantic networks of smell.
*   **`keller_pleasantness.csv`**: Focuses on the hedonic valence of smells, capturing how pleasant or unpleasant specific molecules are perceived to be.

### 2. Molecular & Physical Data
*   **`molecules.csv`**: Metadata and structural information regarding the chemical compounds (often identified by PubChem CIDs) used in the study.
*   **`stimuli.csv`**: Details the specific chemical stimuli presented to the subjects during the trials, including concentrations or mixtures.
*   **`ir_spectra.csv`**: Raw Infrared (IR) spectroscopy data capturing the vibrational frequencies (the "Shake" profile) of the molecules.
*   **`ir_spectra_matrix.csv`**: A structured matrix form of the IR spectra, optimized for machine learning algorithms, distance calculations, and clustering.

### 3. Subject Data
*   **`subjects.csv`**: Demographic and background information on the human participants. This is critical for assessing the subjectivity and demographic variance in olfactory perception.

## Analytical Framework & Network Analysis

This dataset is primed for advanced network analysis to explore three major domains:

1.  **The Perceptual Semantic Network:** 
    *   *Goal:* Map the relationships between scent descriptors using `keller_qualities.csv`.
    *   *Method:* Correlation networks connecting descriptors that frequently co-occur, identifying core "anchor" percepts in human olfaction.
2.  **Molecular Similarity Networks:**
    *   *Goal:* Cluster molecules based on physical properties using `ir_spectra_matrix.csv`.
    *   *Method:* Distance-based graphs (e.g., Cosine similarity) linking molecules with highly similar vibrational profiles to see if they predict similar smells.
3.  **Subjectivity and Bipartite Mapping:**
    *   *Goal:* Understand variance across human smellers using `subjects.csv` + `keller_ratings_long.csv`.
    *   *Method:* Bipartite graphs mapping subjects to the specific chemical clusters they rate similarly, identifying communities of smellers based on demographic or perceptual differences.

## Setup and Installation

To replicate the analyses or explore the dataset, ensure you have a standard Python data science environment configured. We recommend using Python 3.8+ and installing the following dependencies:

```bash
pip install pandas numpy scikit-learn networkx matplotlib seaborn