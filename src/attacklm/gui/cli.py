"""CLI entry point for attacklm-gui."""

import sys


def main() -> int:
    """Launch the AttackLM TUI."""
    from attacklm.gui.app import AttackLMApp

    app = AttackLMApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
