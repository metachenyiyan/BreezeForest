---
description: Deep analysis of a related GitHub repository, comparing its architecture and algorithms with BreezeForest
allowed-tools: Bash, WebSearch, WebFetch, Read, Write, Glob, Grep, Task
argument-hint: <GitHub URL or local path, e.g. "https://github.com/nicola-decao/BNAF" or "./some-local-repo">
---

# Analyze Repo

Perform a deep analysis of a repository related to normalizing flows / density estimation.

**Target**: $ARGUMENTS

## Task: Comprehensive Repository Analysis

### 1. Access the Repository

Determine if `$ARGUMENTS` is a GitHub URL or local path:

**If GitHub URL**:
- Use `gh repo clone [url] /tmp/analyze-repo-target -- --depth=1` to shallow clone
- Use `gh api repos/{owner}/{repo}` to get repo metadata (stars, forks, description, language)
- Set analysis path to the cloned directory

**If local path**:
- Verify the path exists
- Set analysis path to the provided path

### 2. Repository Overview

Gather basic information:
```bash
# File structure
find [path] -type f -name "*.py" | head -50

# README
cat [path]/README.md

# Dependencies
cat [path]/requirements.txt or [path]/setup.py or [path]/pyproject.toml

# Git stats
git -C [path] log --oneline -20
git -C [path] shortlog -sn --no-merges | head -10
```

Record:
- **Language/Framework**: Python, PyTorch, TensorFlow, JAX, etc.
- **Project maturity**: Active development, stable, archived
- **Contributors**: Number and activity
- **Documentation quality**: README completeness, docstrings, examples

### 3. Architecture Analysis

Read the core source files to understand:

#### Model Architecture
- What type of normalizing flow is implemented? (autoregressive, coupling, continuous, residual)
- What are the building blocks? (layers, transforms, distributions)
- How is the model composed? (sequential, parallel, hierarchical)
- What base distributions are used?

#### Jacobian Computation
- How is the log-determinant of the Jacobian computed?
- Triangular structure exploited? LU decomposition? Trace estimation?
- Computational complexity

#### Training
- Loss function (negative log-likelihood, ELBO, other)
- Regularization techniques
- Optimization setup

#### Sampling / Inversion
- How are samples generated from the model?
- Analytical inverse? Iterative inversion? One-pass?

### 4. Code Quality Assessment

Evaluate:
- **Code organization**: Clean module structure? Separation of concerns?
- **Extensibility**: Easy to add new flow types or components?
- **Testing**: Test coverage and quality
- **Documentation**: Inline comments, docstrings, examples
- **Reproducibility**: Clear instructions to reproduce results

### 5. Comparison with BreezeForest

Compare against BreezeForest's key properties:

| Dimension | BreezeForest | [Target Repo] |
|---|---|---|
| Flow type | Autoregressive (conditional CDF) | ? |
| Architecture | Tree + breeze connections | ? |
| Jacobian | Numerical diff / exact jacrev, O(N^2) | ? |
| Universality | Yes (1 hidden layer) | ? |
| Sampling | Batched bisection | ? |
| Regularization | sap_w + MultiBF mixture | ? |
| Dimensions tested | 2D | ? |

### 6. Identify Transferable Ideas

Look for techniques or patterns that could benefit BreezeForest:
- Better numerical methods
- Training tricks or schedules
- Architecture innovations
- Testing approaches
- Benchmark datasets and evaluation metrics
- Code organization patterns

### 7. Output Format

```markdown
# Repository Analysis: [repo name]
**URL**: [GitHub URL]
**Date**: YYYY-MM-DD

## Overview
- **Description**: [what the project does]
- **Language**: [primary language/framework]
- **Stars**: [count] | **Forks**: [count]
- **Last active**: [date of last commit]
- **Maturity**: [active/stable/archived]

## Architecture

### Model Design
[Description of the model architecture with key insights]

### Key Components
- `[file]`: [what it does]
- `[file]`: [what it does]
- ...

### Jacobian Computation
[How log-det-Jacobian is computed, complexity]

### Training Pipeline
[Loss function, optimizer, regularization]

### Sampling
[How samples are generated]

## Code Quality
| Aspect | Rating | Notes |
|---|---|---|
| Organization | [Good/Fair/Poor] | [details] |
| Documentation | [Good/Fair/Poor] | [details] |
| Testing | [Good/Fair/Poor] | [details] |
| Reproducibility | [Good/Fair/Poor] | [details] |

## Comparison with BreezeForest
[Detailed comparison table and discussion]

### Similarities
- [similarity 1]
- [similarity 2]

### Differences
- [difference 1]
- [difference 2]

## Transferable Ideas
1. **[Idea]**: [description + how it could be applied to BreezeForest]
2. **[Idea]**: [description + how it could be applied to BreezeForest]
3. ...

## Summary
[2-3 paragraph summary of findings and recommendations]

## References
- [Paper or resource 1]
- [Paper or resource 2]
```

### 8. Save Results

Save to `notes/repo_analysis/[repo_name]_YYYY_MM_DD.md`.
Create the directory if it doesn't exist.

### 9. Cleanup

If a temporary clone was created, remove it:
```bash
rm -rf /tmp/analyze-repo-target
```
