#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


SGBOT_DIR = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/SG-Bot")


def main() -> None:
    os.chdir(SGBOT_DIR)
    sys.path.insert(0, str(SGBOT_DIR))

    import pybullet

    pybullet.ER_BULLET_HARDWARE_OPENGL = pybullet.ER_TINY_RENDERER

    import sgbot_pybullet

    sgbot_pybullet.main()


if __name__ == "__main__":
    main()
