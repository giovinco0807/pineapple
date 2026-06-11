"""Streaming PyTorch trainer for HU-aware Turn2 teacher JSONL files.

The initial HU Turn2 model uses the same generic HU action-value feature
encoder as HU Turn3.  Multi-head delta/gate training is added after the first
teacher set is validated.
"""

from __future__ import annotations

from .train_torch_hu_turn3_streaming import main


if __name__ == "__main__":
    main()
