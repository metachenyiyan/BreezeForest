---
description: Generate a structured literature review on a sub-topic related to normalizing flows and density estimation
allowed-tools: WebSearch, WebFetch, Write, Read, Glob, Grep, Task
argument-hint: <sub-topic, e.g. "universal density estimation" or "Jacobian computation in normalizing flows">
---

# Literature Review

Generate a structured literature review on a research sub-topic.

**Topic**: $ARGUMENTS

## Task: Produce a Comprehensive Literature Review

### 1. Define Scope

Based on "$ARGUMENTS", define:
- **Core topic**: The central research question or technique
- **Boundaries**: What is included and excluded
- **Time span**: Focus on key works from foundational (pre-2018) to recent (2024-2026)
- **Connection to BreezeForest**: How this topic relates to the BreezeForest project (autoregressive normalizing flows, conditional CDF, Jacobian-free determinant, bisection inversion, mixture regularization)

### 2. Search for Literature

Use `WebSearch` with multiple queries:
- "[topic] survey" or "[topic] review"
- "[topic] normalizing flows"
- "[topic] arxiv" for recent preprints
- "[topic] density estimation"
- Key author names if known

Run at least 5 different search queries to ensure broad coverage.

### 3. Collect and Organize Papers

For each relevant paper found, record:
- Title, authors, year
- Venue (NeurIPS, ICML, ICLR, AISTATS, arXiv, etc.)
- Key contribution (1-2 sentences)
- Methodology category

Group papers into thematic categories relevant to the topic.

### 4. Trace Development Timeline

Identify:
- **Foundational works**: Papers that established the field or technique
- **Key breakthroughs**: Papers that significantly advanced the state of the art
- **Recent developments**: Latest papers pushing the frontier
- **Branching points**: Where the field diverged into different approaches

### 5. Write the Literature Review

Structure the review as follows:

```markdown
# Literature Review: [Topic]
**Date**: YYYY-MM-DD
**Scope**: [brief scope description]

## Abstract
[3-5 sentence summary of the review findings]

## 1. Introduction and Motivation
- Why this topic matters for normalizing flow research
- Connection to BreezeForest's approach
- Scope and organization of this review

## 2. Background and Foundations
- Core concepts and definitions
- Foundational works that established the field
- Key mathematical frameworks

## 3. Key Methods and Approaches

### 3.1 [Approach Category A]
- [Paper 1] (Author et al., Year): [contribution summary]
- [Paper 2] (Author et al., Year): [contribution summary]
- **Summary**: [synthesis of this approach]

### 3.2 [Approach Category B]
- ...

### 3.3 [Approach Category C]
- ...

## 4. Development Timeline
[Chronological narrative of how the field evolved]

## 5. Comparative Analysis
| Method | Key Idea | Strengths | Limitations | Relevance to BreezeForest |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## 6. Connection to BreezeForest
- Which ideas does BreezeForest build upon?
- How does BreezeForest differ from or extend existing methods?
- What techniques from the literature could benefit BreezeForest?

## 7. Open Problems and Future Directions
1. [Open problem]: [description and current state]
2. [Open problem]: [description and current state]
3. ...

## 8. Conclusion
[Summary of key findings and implications for BreezeForest research]

## References
1. [Author et al. (Year). "Title." Venue. URL]
2. ...
```

### 6. Save Results

Save the review to `notes/reviews/[topic_slug]_YYYY_MM_DD.md`.
Create the directory if it doesn't exist.
