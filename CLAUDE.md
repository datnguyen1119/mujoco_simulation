# CLAUDE.md

## Goal
Portfolio robotics lab series in MuJoCo: planar arm → industrial arm → humanoid → VLA.
Roadmap: `plan/MASTER_PLAN.md`. Current progress: see README.

## Architecture Principle
- Pinocchio = analytical (FK, Jacobian, dynamics, IK)
- MuJoCo = simulation (step, contact, render)
- Never compute the same quantity twice; when both can, cross-validate.

## Per-Lab Workflow (in order)
1. Read `plan/LAB_XX.md` fully
2. Write `tasks/PLAN.md`, then `tasks/ARCHITECTURE.md`, before code
3. Keep `tasks/TODO.md` (Current Focus, Blockers), update after every step
4. Log bugs in `tasks/LESSONS.md` immediately: Symptom / Root cause / Fix / Takeaway
5. One phase at a time; tests pass before the next phase

## Commands
pip install mujoco numpy pin scipy matplotlib pytest   # `pin`, NOT `pinocchio`
pytest lab-1-2link-arm/tests/

## Code Standards
- Docstrings + type hints; pathlib; no absolute paths
- Numeric checks with np.allclose and explicit atol
- Docs in English (docs/) and Turkish (docs-turkish/)

## Known Issues
<!-- Add an entry only after a mistake happens (or nearly happens) twice.
     Format: symptom → rule → where it was learned. -->

## Session Start
1. This file → 2. lab brief → 3. tasks/TODO.md → 4. tasks/LESSONS.md