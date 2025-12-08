# mpld3 Development Notes

## Key Insight: Bundled mplexporter

**mpld3 bundles its own copy of mplexporter** at `mpld3/mpld3/mplexporter/`. The standalone mplexporter repo is NOT used at runtime. When testing mplexporter PRs, you must copy files to the bundled location.

## Directory Structure

```
mpld3_notes/
├── mpld3-dev/           # Development environment (gitignored)
│   ├── .venv/           # Python virtual environment
│   ├── mpld3/           # Cloned mpld3 repo
│   └── mplexporter/     # Cloned mplexporter repo (for PR testing)
├── test_pr68_*.py       # Test for mplexporter PR #68 (log-scale fix)
├── test_pr536_*.py      # Test for mpld3 PR #536 + mplexporter PR #70
└── README.md            # Setup instructions
```

## Common Commands

```bash
# Activate environment
cd /home/abie/ai_assisted_research/mpld3_notes/mpld3-dev
source .venv/bin/activate

# Run Python without path shadowing
python -P script.py

# Apply mplexporter PR to bundled copy
cp mplexporter/mplexporter/exporter.py mpld3/mpld3/mplexporter/exporter.py

# Restore bundled mplexporter
cd mpld3 && git checkout -- mpld3/mplexporter/
```

## Related PRs Often Need Both Repos

- mpld3 PR #536 (figure text) requires mplexporter PR #70
- mpld3 PR testing may require mplexporter PR #68 (log-scale)

## Branch Names

- mpld3: `master` (not `main`)
- mplexporter: `master`
