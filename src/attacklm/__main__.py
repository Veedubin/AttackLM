"""Allow `python -m attacklm …` (used by `attacklm queue start --detach`)."""

from attacklm.cli import main

if __name__ == "__main__":
    main()
