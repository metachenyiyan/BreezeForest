---
description: Search for academic papers related to normalizing flows, density estimation, and generative models
allowed-tools: WebSearch, WebFetch, Write, Read, Glob
argument-hint: <keywords or topic, e.g. "autoregressive normalizing flows" or "Jacobian determinant computation">
---

# Search Papers

Search for academic papers relevant to BreezeForest and normalizing flow research.

**Search query**: $ARGUMENTS

## Task: Find and Organize Relevant Papers

### 1. Construct Search Queries

Based on "$ARGUMENTS", construct multiple search queries targeting:
- arXiv (site:arxiv.org)
- Google Scholar
- Semantic Scholar

Use combinations of the user's keywords with domain-specific terms:
- normalizing flows, autoregressive flows
- density estimation, universal density estimator
- Jacobian determinant, triangular Jacobian
- conditional CDF, cumulative distribution function
- generative models, flow-based models
- mixture models, Gaussian mixture

### 2. Execute Searches

Use `WebSearch` to search across academic sources. Run at least 3 different query variations to maximize coverage. For each search:
- Include "arxiv" or "paper" in queries to bias toward academic results
- Search for both recent papers (2023-2026) and foundational works

### 3. Fetch Paper Details

For the most relevant results (top 5-10), use `WebFetch` on the arxiv abstract page to extract:
- Title
- Authors
- Abstract summary
- Publication date
- arXiv ID

### 4. Assess Relevance to BreezeForest

For each paper, assess relevance to BreezeForest's key techniques:
- **Autoregressive architecture**: Does it use autoregressive decomposition of joint density?
- **Jacobian computation**: How does it handle the Jacobian determinant? (triangular, numerical, exact)
- **Universality**: Does it claim or prove universal approximation for densities?
- **Inversion/sampling**: How does it handle inverse mapping for generation? (bisection, analytical)
- **Regularization**: Does it address overfitting / sample memorization?
- **Mixture models**: Does it use mixture components?

### 5. Output Structured Results

Output results in this format:

```markdown
# Paper Search: [query topic]
**Date**: YYYY-MM-DD

## Summary
[1-2 sentence overview of findings]

## Papers Found

### [Relevance: High/Medium/Low] Paper Title
- **Authors**: ...
- **Year**: ...
- **arXiv**: https://arxiv.org/abs/XXXX.XXXXX
- **Key contribution**: [1 sentence]
- **Relation to BreezeForest**: [How this connects to BreezeForest's approach]
- **Techniques**: [e.g., autoregressive flow, coupling layers, continuous normalizing flow]

### [Next paper...]

## Key Takeaways
1. ...
2. ...

## Suggested Follow-up
- [Further search directions or papers to read in depth]
```

### 6. Save Results

Save the results to `notes/papers/search_YYYY_MM_DD_[topic_slug].md`.
Create the directory if it doesn't exist.
