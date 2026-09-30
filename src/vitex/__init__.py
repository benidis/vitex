"""vitex — reconstruct tutorial videos into structured LaTeX documents.

The public entry point is the CLI (`vitex run <url|path>`), which drives the
staged pipeline in :mod:`vitex.pipeline`. The pipeline is also importable for
programmatic use.
"""

from vitex.config import Config
from vitex.pipeline import Pipeline

__version__ = "0.1.0"
__all__ = ["Config", "Pipeline", "__version__"]
