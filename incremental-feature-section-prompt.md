<!-- ╔════════════════════════════════════════════════════════════╗
     ║  AI CODING PROMPT — ORG-WIDE STANDARD  · v2.0 · 2025-05-28 ║
     ╚════════════════════════════════════════════════════════════╝ -->
# 🤖 Unified AI-Coding Preamble

## 1 · Mission
Generate production-grade, test-driven, secure code that complies with our style, automates its own release, and documents *why* as well as *what*.

## 2 · Baseline Repositories
| Alias | URL | Branch | Role |
|-------|-----|--------|------|
| core-api | git@github.com:ORG/core-api.git | main | FastAPI service skeleton |
| terraform-iac | git@github.com:ORG/terraform-iac.git | prod | AWS/EKS infra |
| ui-components | git@github.com:ORG/ui-components.git | dev | React + shadcn UI kit |
| prompt-lib | git@github.com:ORG/prompt-library.git | main | Versioned prompt snippets |

**Reuse first:** import before re-inventing.

## 3 · Toolchain Matrix
```text
python 3.12 · fastapi 0.111 · pydantic 2.x
node 20 · typescript 5 · react 19 · vite 5
terraform ≥ 1.7   · docker compose 2.27
```text
## 4 · Dev Container
.devcontainer.json extends the python-3-node-lts template and installs Ruff, Black, Poetry and AWS CLI.

.vscode/extensions.json recommends ms-python.vscode-pylance, esbenp.prettier-vscode, and github.copilot.

## 5 · Coding & Commit Rules
Python – Black, Ruff, pyupgrade, full type hints.

TS/JS – ESLint (core + Airbnb), Prettier, import-sort.

Commits – Conventional Commits enforced by @commitlint/config-conventional.

Branch names – feat/{ticket}, fix/{ticket}, docs/{topic}.

## 6 · Pre-Commit
Ships with a project-root .pre-commit-config.yaml that activates:

trailing-whitespace, end-of-file-fixer, check-yaml, black, ruff, isort, detect-aws-credentials.

To install locally:

bash
Copiar
Editar
pip install pre-commit && pre-commit install
7 · Lint, Test, Build
Stage	Tool	Blocking?
super-linter	GitHub Action lint.yml	✅
unit (PyTest + Coverage ≥ 90 %)	test.yml	✅
e2e (Playwright)	e2e.yml	✅
build & SBOM	Docker Buildx + Trivy	✅

## 8 · Release & Versioning
semantic-release on main →

bumps CHANGELOG.md,

tags git,

publishes Docker image to GHCR,

creates GitHub Release.

For Python packages, bump-my-version mirrors the same semver.

## 9 · Dependency Automation
renovate.json (self-hosted Renovate App) auto-PRs outdated deps, grouped by ecosystem; security fixes marked priority-high.

## 10 · Docs & ADRs
MkDocs + mkdocstrings autogen API docs.

Architecture decisions live in /docs/adr/NNN-title.md (adr-tools).

## 11 · Security Baseline
Secrets via AWS Secrets Manager (ARN placeholders).

Trivy image & IaC scans on every PR.

Dependabot alerts ≤ 7 days open.

## 12 · Prompt Engineering Rules
Keep prompts in /prompts/{domain}/{slug}.md.

Include:

yaml
Copiar
Editar
---
intent: "Summarise user feedback"
input_schema: feedback[]
output_schema: summary, sentiment
test_cases:
  - input: [...]
    expect: [...]
---
Link each production prompt to code via ID.

Unit-test critical prompts with fixtures.

## 13 · AI Response Guidelines
Prefer diff patches for code.

≤ 250 LOC unless scope expanded.

Ask clarifying questions if ambiguous.

Inline // why: comments for non-obvious logic.

## 14 · Extensibility Tokens
{{PROJECT_NAME}}, {{AWS_REGION}}, {{PORT}}, {{PROMPT_ID}}

----
# Task

"Implement a hello world endpoint that calls the Open AI API for a dynamic response based on historical trivia for the current day"
