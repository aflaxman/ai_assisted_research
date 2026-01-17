# AI-Assisted Research Project

This repository contains notes and experiments for AI-assisted open source development work.

## Structure

- `mpld3_notes/` - Development environment and PR testing for mpld3/mplexporter

## Environment

- Running on WSL (Windows Subsystem for Linux)
- Use `python -P` to avoid path shadowing issues when working inside cloned repos
- Browser files open via `explorer.exe` on WSL

## Conventions

- Test scripts follow the pattern `test_pr<NUMBER>_<description>.py`
- Use `uv` for Python package management
- Commit messages include the Claude Code attribution footer
- **Quickstart sections**: Should be simple and direct - choose ONE recommended path, not multiple options. Don't force users to make decisions in quickstart; save alternatives for detailed instructions.
