---
description: Track updates and activity in peer normalizing flow open-source repositories
allowed-tools: Bash, WebSearch, WebFetch, Write, Read, Glob
argument-hint: "[time range, e.g. '7 days ago' or '1 month ago'] [optional: repo names]"
---

# Track Peer Repos

Track recent activity in normalizing flow and density estimation open-source repositories.

**Arguments**: $ARGUMENTS

## Tracked Repositories

The following are key repositories in the normalizing flow ecosystem:

| Short Name | GitHub Repo | Description |
|---|---|---|
| nflows | bayesiains/nflows | Normalizing flows in PyTorch |
| normflows | VincentStimper/normalizing-flows | Normalizing flows library |
| FrEIA | vislearn/FrEIA | Framework for Easily Invertible Architectures |
| BNAF | nicola-decao/BNAF | Block Neural Autoregressive Flow |
| survae-flows | didriknielsen/survae_flows | Surjective and surjective flows |
| zuko | probabilists/zuko | Normalizing flows in PyTorch |
| flowtorch | facebookresearch/flowtorch | Flow-based generative models |
| neural-spline-flows | bayesiains/nsf | Neural Spline Flows |
| FFJORD | rtqichen/ffjord | Free-form Jacobian of Reversible Dynamics |
| residual-flows | rtqichen/residual-flows | Residual Flows |

## Task: Generate Activity Report

### 1. Parse Arguments

Parse `$ARGUMENTS` to extract:
- **Time range**: e.g., "7 days ago", "1 month ago", "2025-01-01". Default to "14 days ago" if not specified.
- **Repo filter**: If specific repo names are given, only track those. Otherwise track all.

### 2. Collect Data for Each Repository

For each repository, use `gh` CLI to gather:

```bash
# Recent commits
gh api repos/{owner}/{repo}/commits --jq '.[0:20] | .[] | "\(.sha[0:7])|\(.commit.author.date)|\(.commit.message | split("\n")[0])"' 2>/dev/null

# Recent releases
gh api repos/{owner}/{repo}/releases --jq '.[0:5] | .[] | "\(.tag_name)|\(.published_at)|\(.name)"' 2>/dev/null

# Open issues count and recent issues
gh api repos/{owner}/{repo}/issues?state=open\&per_page=5 --jq '.[] | "\(.number)|\(.created_at)|\(.title)"' 2>/dev/null

# Repository stats
gh api repos/{owner}/{repo} --jq '"\(.stargazers_count) stars | \(.forks_count) forks | \(.open_issues_count) open issues"' 2>/dev/null
```

### 3. Analyze Activity

For each repository, assess:

#### Activity Level
- **Commit frequency**: commits per week in the time range
- **Release cadence**: any new releases?
- **Issue activity**: new issues opened/closed?
- **Rating**: Dormant / Low / Moderate / Active / Very Active

#### Technical Direction
Categorize commits by type:
- **feat**: New features, capabilities, model architectures
- **fix**: Bug fixes, numerical stability improvements
- **perf**: Performance optimizations, memory efficiency
- **refactor**: Code restructuring, API changes
- **docs**: Documentation updates
- **test**: Test additions/improvements

#### Relevance to BreezeForest
For significant updates, note:
- Does it relate to autoregressive flows?
- Any improvements to Jacobian computation?
- New training techniques or regularization methods?
- Sampling/inversion improvements?

### 4. Generate Report

Output the report in this format:

```markdown
# Peer Repository Activity Report
**Period**: [start date] to [end date]
**Generated**: YYYY-MM-DD

## Activity Overview

| Repository | Stars | Commits | Releases | Activity |
|---|---|---|---|---|
| nflows | ⭐ XXX | N commits | vX.Y.Z | 🟢 Active |
| normflows | ⭐ XXX | N commits | - | 🟡 Moderate |
| ... | ... | ... | ... | ... |

## Detailed Analysis

### [Repository Name]
**Activity**: [rating]
**Stats**: [stars] stars, [forks] forks

#### Key Updates
- [commit hash] [description] — **[category]**: [brief analysis]
- ...

#### Technical Direction
[Summary of where this project is heading]

#### Relevance to BreezeForest
[What BreezeForest can learn from or compare against]

### [Next repository...]

## Cross-Project Trends
1. [Trend observed across multiple repos]
2. ...

## Signals Worth Noting

### Notable
- [Signal]: [description + implication]

### Opportunities
- [Opportunity]: [description + how BreezeForest could benefit]

## Metadata
- Report generated: YYYY-MM-DD
- Data range: [start] to [end]
- Repositories tracked: [count]
```

### 5. Save Report

Save to `notes/peer_repos/activity_YYYY_MM_DD.md`.
Create the directory if it doesn't exist.
