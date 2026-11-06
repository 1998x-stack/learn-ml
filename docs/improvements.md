# Portfolio Improvement History

### 2026-11-06 04:19 UTC — ChatGPT Portfolio Iteration
- Run ID: portfolio-20261106T0418Z-a11
- Repo / Base SHA: 1998x-stack/learn-ml / b4bd704023b31972b3b9f60f37faa4c0371fd168
- Historical context: docs/improvements.md was confirmed absent (GitHub 404) on default branch master; first initialization. Repository PR search returned zero PRs, so no same-topic open PR was observed. Recent commits addressed ML chapters, not split size validation.
- Product problem & expected user value: Learners calling train_test_split with invalid fractions or one-row datasets previously received obscure NumPy errors or empty partitions; explicit validation makes mistakes actionable and prevents misleading evaluation.
- Engineering diagnosis & evidence: src/mini_sklearn/model_selection/_split.py calculated ceil(n * test_size) without checking type, finite range, or nonempty train/test partitions; invalid fractions can silently produce empty splits or errors downstream.
- Options & decision: (1) invalid split fraction and empty partitions, P1, score 5+4+5+5-1-1=17; (2) regression metric empty input guard, P2, provisional score 4+3+4+4-2-2=11; (3) error-message consistency across estimators, P3, provisional score 2+2+2+3-3-2=4. Only option 1 implemented; other options require further source review.
- Changes: src/mini_sklearn/model_selection/_split.py validates finite numeric fraction in (0,1) and nonempty train/test; tests/model_selection/test_split.py adds invalid, singleton, and two-row cases. Existing valid fractional split order and RNG behavior preserved.
- Validation: A (isolated reproduction, not full repository): python -m pytest tests/model_selection/test_split.py -q in Python/NumPy/pytest local reconstructed minimal harness: 13 passed. D: committed repository test suite and full parity tests were not executed in a checked-out repository; independent CI status unverified. Reproduction reimplemented the fetched target module and minimal utility behavior, not a byte-identical full checkout.
- Security & regression review: No secrets, permissions, workflow, deployment, or production changes. Valid test_size=0.25 retains original order. Regression risk: callers relying on invalid/empty partitions now receive ValueError.
- Risk & rollback: Low; close this Draft PR or revert its commits on the feature branch; default branch untouched.
- Follow-up: Run full repository pytest and parity suite in a real checkout, including numpy scalar type edge cases.
- Commit: d83bac845bad2e42a6332c2680ad50eb2b0b16f2 (test change; history committed separately)
- PR: https://github.com/1998x-stack/learn-ml/pull/1
