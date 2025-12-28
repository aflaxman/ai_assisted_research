# AI-Assisted Research Project

This repository contains notes and experiments for AI-assisted open source development work.

## Structure

Use subdirectories for each research project, to keep things separated, e.g.

- `mpld3_notes/` - Development environment and PR testing for mpld3/mplexporter

## Environment

I would like to keep the environments easy to create and isolated, e.g. for mpld3_notes I will be

- Running on WSL (Windows Subsystem for Linux)
- Use `python -P` to avoid path shadowing issues when working inside cloned repos
- Browser files open via `explorer.exe` on WSL


## Conventions

- Use `uv` for Python package management
- Commit messages include the Claude Code attribution footer

Also for mpld3_notes

- Test scripts follow the pattern `test_pr<NUMBER>_<description>.py`
