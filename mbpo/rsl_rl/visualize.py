"""MJLab visualization entry point.

MJLab handles viewer selection through the play CLI. This wrapper keeps the
local script name available for existing workflows.
"""

from __future__ import annotations

import go1_tasks  # noqa: F401
from mjlab.scripts.play import main


if __name__ == "__main__":
    main()
