# Project Context

## Purpose
This project is a small security-focused homework / lab repository. Its goals are:

- Provide a repeatable environment for experimenting with and demonstrating security concepts (vulnerabilities, exploit mitigation, protocol analysis, etc.).
- Contain exercises, vulnerable sample code, and automated tests or checks that verify learning objectives.
- Be easy to run locally and reproducible across contributors (students, TAs, reviewers).

If you are using this repository for a specific course or assignment, add the course name, assignment number, and learning objectives here.

## Tech Stack
- Primary language(s): Python 3.10+ for tooling, scripts, and backend exercises.
- Optional frontend: lightweight HTML/JS or minimal React app (if exercises include web components).
- Build & packaging: standard tooling (venv, pip, setuptools or Poetry).
- CI: GitHub Actions (recommended) for automated tests, linting, and artifact creation.
- Testing: pytest for unit/integration tests; hypothesis for property-based tests when useful.
- Static analysis: flake8/black/isort for Python formatting and linting; ESLint/Prettier for JS if present.
- Containerization: Dockerfiles for reproducible environments where exercises require specific system state.

> Note: These choices are reasonable defaults. If your environment uses Node, Go, or other languages, substitute the relevant tools and workflows.

## Project Conventions

### Code Style
- Python: follow PEP8 style with Black for automatic formatting and isort for imports.
- JavaScript: follow the Airbnb style guide or a minimal Prettier+ESLint setup if a frontend exists.
- Keep functions small and focused; prefer pure functions for logic-heavy code to simplify testing.
- Name tests to clearly express behavior: test_<module>__should_<behavior> or use pytest-style descriptive test names.

Formatting tools (run locally or via pre-commit):

- black .
- isort .
- flake8 .

Add a `.editorconfig` and recommended VS Code settings for consistent editor behavior across contributors.

### Architecture Patterns
- Keep the repository modular: separate exercises, vulnerable-app(s), tooling, and docs into top-level folders such as `exercises/`, `vulnerable_app/`, `tools/`, and `docs/`.
- Prefer small, well-documented scripts over monolithic binaries so students can read and modify code easily.
- If the project exposes networked components (HTTP, sockets), clearly document expected ports and provide Docker Compose files to start dependent services.

### Testing Strategy
- Unit tests: target pure logic and parsers with pytest.
- Integration tests: spin up lightweight test instances (using pytest fixtures, Docker Compose, or test clients) to validate end-to-end behavior.
- Security checks: include regression tests that exercise known vulnerable code paths so fixes can't regress.
- Test coverage: aim for meaningful coverage on core logic. For small teaching repositories 70–90% is reasonable; prioritize critical code paths.

Run tests locally:

- python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; pytest -q

### Git Workflow
- Branching: use GitHub flow—create a short-lived feature branch for each change: `feature/<short-desc>` or `fix/<short-desc>`.
- Pull requests: open PRs targeting `main` (or `master`) and include a description of changes, relevant issue link, and test/CI status.
- Commits: keep commits small and focused. Use imperative, present-tense commit messages (e.g., "Add X", "Fix Y").
- Conventional commits (optional): consider using Conventional Commits (feat/fix/docs/chore) if you want automated changelogs.

## Domain Context
This repository centers on security educational material. Important domain notes for contributors and AI assistants:

- Exercises may intentionally include insecure patterns (e.g., SQL injection, insecure deserialization). These are for learning only — do not deploy vulnerable examples to production.
- When adding fixes or mitigations, document the vulnerability, the exploit scenario, and the reasoning behind the chosen mitigation.
- Keep reproducible test cases that demonstrate both vulnerable and fixed behavior.

## Important Constraints
- Safety: never include real credentials, private keys, or sensitive production data in the repository. Use placeholders and environment variables.
- Licensing: include a LICENSE file (e.g., MIT or CC BY-SA for course material) and ensure third-party materials used in exercises permit redistribution.
- Resource limits: CI runners may have limited resources — avoid extremely long-running tests in the main workflow; mark longer experiments as optional.

## External Dependencies
- If exercises depend on external services (e.g., public APIs, package registries, or cloud services), document the dependency and provide mock or local alternatives when possible.
- Typical external dependencies in this project may include:
	- PyPI packages listed in `requirements.txt` or `pyproject.toml`.
	- Docker Hub images referenced by Dockerfiles or docker-compose.
	- Optional: simple external APIs for demonstration—provide recorded fixtures or mock servers for offline runs.

## Contributors & Ownership
- Owner: add the course instructor or repository maintainer name and contact email here.
- Contribution process: open issues for new exercises or problems, create feature branches, add tests, and open a PR for review.

## Onboarding / How to Run Locally
1. Clone the repo and create a Python virtual environment.

- git clone <repo-url>
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1
- pip install -r requirements.txt

2. Run unit tests:

- pytest -q

3. If there is a Docker setup:

- docker-compose up --build

Customize ports and environment variables in `.env.example` and add a `.env` locally (gitignored).

## Suggested Follow-ups
- Add a `README.md` at repository root with quick start steps and links to each exercise.
- Add CI configuration (GitHub Actions) to run linters and tests on each PR.
- Add a `CONTRIBUTING.md` with a small checklist for PRs and code review expectations.

---

If you'd like, I can: create or update a `README.md`, add a CI workflow file, generate a `.editorconfig`, or tailor this spec to a different tech stack you prefer—tell me which and I'll update the files.
