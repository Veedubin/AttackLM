"""CLI entry point for attacklm-gui."""

import sys
from pathlib import Path


def main() -> int:
    """Launch the AttackLM TUI."""
    from attacklm_gui.app import AttackLMApp

    app = AttackLMApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
