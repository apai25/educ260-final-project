from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TaxonomyConfig:
    num_subgroups: List[int] = field(default_factory=lambda: [10, 5, 5])
    max_courses: int = 1000
    batch_size: int = 50
    model: str = "o4-mini"
    cluster_fn: str = "llm"

    save_dir: Path = Path("outputs/taxonomy").resolve()
