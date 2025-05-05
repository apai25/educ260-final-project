from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    course_info_csv: Path = Path(
        "/data/groups/ccc/course_catalog_202401161416.csv"
    ).resolve()

    course_embed_dir: Path = Path(
        "/data/groups/ccc/working/openai_mpsr_ccc/0"
    ).resolve()

    data_save_dir: Path = Path("data").resolve()
    embed_dim: int = 1536
