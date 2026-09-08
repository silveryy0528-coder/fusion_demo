---
applyTo: "**"
---

# Project Context

- This is a Python implementation for PLS regression-based cross-modal fusion.
- Favor simple, maintainable solutions consistent with established Python practices.
- Follow the existing project structure unless there is a clear reason to change it.

# Coding

- Prefer straightforward implementations over unnecessary abstractions.
- Make the smallest coherent change needed for the task.
- Do not modify unrelated code or add dependencies without a clear benefit.
- During refactoring, preserve existing behavior and public interfaces unless a change is explicitly requested.
- Run the relevant tests after changes when possible; otherwise state which tests should be run.
- Preserve numerical behavior within appropriate floating-point tolerance during refactoring.

# Testing

- Use pytest.
- Do not test third-party library behavior directly; test the behavior introduced by this project.
- Prefer test names in the form:
  `test_<function>_WHEN_<condition>_THEN_<expected_behavior>`.
- Test observable behavior and important edge cases without unnecessary duplication.

# Documentation

- Follow PEP 257 and use NumPy-style docstrings.
- Keep docstrings and comments concise.
- Document non-obvious behavior; do not restate the code.
- Do not expose personal or sensitive information in documentation or comments.

# Working Style

- Inspect the relevant implementation and tests before making changes.
- After changes, identify relevant tests to run.
- Keep explanations concise and focused on:
  1. What changed.
  2. Why.
  3. What to do next, if anything.

# Recommendations

When comparing improvement options:

- Clearly distinguish the meaningful alternatives.
- Give up to three relevant pros and cons for each.
- Compare them directly using relevant criteria such as simplicity, maintainability, behavior, performance, or testability.
- Recommend one option, preferring the simplest solution that adequately solves the current problem.
