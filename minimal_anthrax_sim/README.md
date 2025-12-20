# Minimal Anthrax Sim (JAX + diffrax)

This project will house a JAX + diffrax investigation of a system of differential equations. The notebook content will be added once the write-up is ready.

## Quickstart (VS Code + WSL)

1. From WSL, create the virtual environment and install dependencies:

   ```bash
   uv venv
   uv sync
   ```

2. Open the project in VS Code (WSL):

   ```bash
   code .
   ```

3. In VS Code, select the kernel from `.venv` when opening the notebook in `notebooks/`, then run all cells.

## Reproducibility notes

- The environment is specified in `pyproject.toml`.
- If you need a locked environment, run:

  ```bash
  uv lock
  uv sync
  ```

## Next steps

- Drop the markdown write-up and code, and I will refactor it into a notebook.
