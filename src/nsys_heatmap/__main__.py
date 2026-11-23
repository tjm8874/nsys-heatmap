"""Allow `python3 -m nsys_heatmap` execution."""

import sys

from nsys_heatmap.cli import main

if __name__ == "__main__":
    sys.exit(main())
