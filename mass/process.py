"""The single subprocess seam: every external command the driver launches goes through run().

Call sites use `process.run(...)` (module attribute lookup) so tests can patch one name.
"""

import shlex
import subprocess
from collections.abc import Iterable, Mapping

from .io import ROOT


def run(command: Iterable[object], env: Mapping[str, str] | None = None) -> None:
    """Echo and run `command` from the repository root; raise CalledProcessError on failure."""
    command = [str(x) for x in command]
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
