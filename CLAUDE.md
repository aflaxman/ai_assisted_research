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

## Technical Blog Post Guidelines

When writing technical blog posts for healthyalgorithms.com:

### Project Structure

1. **Create a new subdirectory** for each blog post (e.g., `simple_fuzzy_checker_application/`)
2. **Use `uv` to set up a Python environment** in each subdirectory for isolated dependencies
3. **Put the blog draft in `README.md`** in the subdirectory

### Content Structure

1. **Keep it simple** - Focus on clarity over complexity
2. **Start with a hook** - Begin with a minimal, concrete code example that demonstrates the core topic
3. **Include a TL;DR section** - Provide quick takeaways at the beginning
4. **Include a graphic** - Add a visualization (animation, diagram, or plot) to illustrate the concept
5. **Provide runnable code** - Make the code accessible via:
   - Colab notebook (add badge: `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link)`)
   - Binder notebook
   - Self-contained scripts
6. **Separate concerns** - Keep simulation/implementation code separate from test code
7. **Link to specific code** - Reference code with pattern `file_path:line_number` or GitHub permalink

### File Organization

Each blog post directory should include:
- `README.md` - The blog post content
- `requirements.txt` - Python dependencies for `uv`
- Implementation files (e.g., `simulation.py`)
- Test files (e.g., `test_simulation.py`)
- Jupyter notebook (e.g., `tutorial.ipynb`)
