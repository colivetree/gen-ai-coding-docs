<!-- ╔════════════════════════════════════════════════════════════╗
     ║  AI BUGFIX PROMPT — ORG-WIDE STANDARD  · v1.0 · 2025-01-27 ║
     ╚════════════════════════════════════════════════════════════╝ -->
# 🐛 Automated AI Bugfix Preamble

## 1 · Mission
Generate **minimal, surgical** bug fixes that preserve existing functionality, maintain code quality standards, and include comprehensive validation of the fix.

## 2 · Baseline Repositories
| Alias | URL | Branch | Role |
|-------|-----|--------|------|
| core-api | git@github.com:ORG/core-api.git | main | FastAPI service skeleton |
| terraform-iac | git@github.com:ORG/terraform-iac.git | prod | AWS/EKS infra |
| ui-components | git@github.com:ORG/ui-components.git | dev | React + shadcn UI kit |
| prompt-lib | git@github.com:ORG/prompt-library.git | main | Versioned prompt snippets |

**Fix first:** understand the root cause before changing code.

## 3 · Toolchain Matrix

| Tool | Version |
|------|---------|
| Python | 3.12 |
| FastAPI | 0.111 |
| Pydantic | 2.x |
| Node.js | 20 |
| TypeScript | 5 |
| React | 19 |
| Vite | 5 |
| Terraform | ≥1.7 |
| Docker Compose | 2.27 |

## 4 · Bug Analysis Protocol
1. **Reproduce** the issue using provided steps/test cases
2. **Isolate** the root cause via debugging/logging
3. **Assess** impact scope (files, functions, dependencies)
4. **Plan** minimal change set
5. **Validate** fix doesn't break existing functionality

## 5 · Coding & Commit Rules
Python – Black, Ruff, pyupgrade, full type hints.

TS/JS – ESLint (core + Airbnb), Prettier, import-sort.

Commits – `fix(scope): brief description` format enforced by @commitlint/config-conventional.

Branch names – `fix/{{ISSUE_ID}}-{{SHORT_DESCRIPTION}}`

**Bugfix-specific rules:**
- Preserve existing code style and patterns
- Add comments explaining the fix rationale
- Include regression test if missing

## 6 · Pre-Commit Validation
All standard checks MUST pass:

trailing-whitespace, end-of-file-fixer, check-yaml, black, ruff, isort, detect-aws-credentials.

**Additional bugfix validation:**
- Existing tests still pass
- New test covers the bug scenario
- No unrelated formatting changes

## 7 · Testing Requirements
| Stage | Tool | Requirement |
|-------|------|-------------|
| Regression | Existing test suite | ✅ 100% pass |
| Bug-specific | New test case | ✅ Covers reported issue |
| Coverage | PyTest + Coverage | ✅ No decrease in coverage |
| Integration | Relevant e2e tests | ✅ Pass |

## 8 · Change Scope Constraints
**MINIMIZE CHANGES:**
- Touch only files directly related to the bug
- Preserve existing function signatures when possible
- Avoid refactoring unless essential to the fix
- No dependency updates unless bug-related

**MAXIMUM CHANGE LIMITS:**
- ≤ 50 LOC for simple bugs
- ≤ 150 LOC for complex bugs
- Request approval if exceeding limits

## 9 · Documentation Requirements
**Required in PR description:**
- Root cause analysis
- Files changed and why
- Test strategy
- Rollback plan if needed

**Code comments:**
```python
# BUG FIX: {{ISSUE_ID}} - Brief description of what was wrong
# REASON: Explain why this specific change fixes the issue
```

## 10 · Validation Checklist
Before submitting PR, verify:

- [ ] Bug is reproducible with provided steps
- [ ] Fix addresses root cause, not just symptoms  
- [ ] All existing tests pass
- [ ] New test case prevents regression
- [ ] No unrelated code changes
- [ ] Performance impact assessed (if applicable)
- [ ] Security implications reviewed
- [ ] Documentation updated if behavior changes

## 11 · Security & Safety
**Critical bug fixes:**
- Security vulnerabilities get priority review
- Breaking changes require explicit approval
- Database migrations need rollback scripts
- API changes require version compatibility check

**Safety nets:**
- Feature flags for risky fixes
- Gradual rollout for user-facing changes
- Monitoring alerts for key metrics

## 12 · AI Response Guidelines for Bugfixes
**Analysis phase:**
- Explain the root cause clearly
- Show reproduction steps
- Identify all affected components

**Implementation phase:**
- Provide exact diff patches
- Explain each change with inline comments
- Include test cases that verify the fix

**Validation phase:**
- List all tests that should pass
- Identify potential side effects
- Suggest monitoring points post-deployment

## 13 · Prompt Schema for Bug Reports
```yaml
---
bug_id: "{{ISSUE_ID}}"
severity: "critical|high|medium|low"
component: "{{COMPONENT_NAME}}"
reproduction_steps: 
  - step1
  - step2
expected_behavior: "What should happen"
actual_behavior: "What actually happens"
environment: "{{ENV_DETAILS}}"
logs: "{{ERROR_LOGS}}"
test_cases:
  - input: "{{TEST_INPUT}}"
    expected: "{{EXPECTED_OUTPUT}}"
    actual: "{{ACTUAL_OUTPUT}}"
---
```

## 14 · Extensibility Tokens
{{ISSUE_ID}}, {{COMPONENT_NAME}}, {{SEVERITY}}, {{ENV_DETAILS}}, {{ERROR_LOGS}}

----
# Bug Report Template

**Issue ID:** {{ISSUE_ID}}
**Severity:** {{SEVERITY}}
**Component:** {{COMPONENT_NAME}}

**Description:**
{{BUG_DESCRIPTION}}

**Reproduction Steps:**
1. {{STEP_1}}
2. {{STEP_2}}
3. {{STEP_3}}

**Expected Behavior:**
{{EXPECTED_BEHAVIOR}}

**Actual Behavior:**
{{ACTUAL_BEHAVIOR}}

**Environment:**
{{ENVIRONMENT_DETAILS}}

**Error Logs:**
```
{{ERROR_LOGS}}
```

**Additional Context:**
{{ADDITIONAL_CONTEXT}} 