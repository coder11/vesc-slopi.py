# Agent Instructions

- Work inside `nix develop`.
- Use `uv sync`, `uv run ...`, and `uv run poe ...` for Python work.
- Do not use system `python`, `pip`, or ad hoc virtualenv tooling.
- Keep tests as unit tests. Do not add hardware integration tests; stub/mock VESC hardware boundaries.
- Firmware XMLs come from the `vesc_tool` submodule at `vesc_tool/res/config`.
