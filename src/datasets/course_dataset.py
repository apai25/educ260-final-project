import os
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset

from configs.data_config import DataConfig


class CourseDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.data_path = self.cfg.data_save_dir / "course_data.pkl"

        if self.data_path.exists():
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = self._build_and_save()

    def _build_and_save(self):
        df = pd.read_csv(self.cfg.course_info_csv, encoding="utf-8", dtype=str)
        df = df[["control_number", "title", "catalogue_description"]].dropna()

        info_map = {
            row["control_number"].strip().lower(): {
                "title": str(row["title"]).strip(),
                "description": str(row["catalogue_description"]).strip(),
            }
            for _, row in df.iterrows()
        }

        embed_map = {}
        for vec_file in self.cfg.course_embed_dir.glob("vectors-*.txt"):
            with open(vec_file, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 1537:
                        continue
                    cid = parts[0].lower()
                    vec = torch.tensor(
                        [float(x) for x in parts[1:]], dtype=torch.float32
                    )
                    embed_map[cid] = vec

        common_ids = set(info_map).intersection(embed_map)
        final_data = []
        for cid in common_ids:
            final_data.append(
                {
                    "control_number": cid,
                    "title": info_map[cid]["title"],
                    "description": info_map[cid]["description"],
                    "embedding": embed_map[cid],
                }
            )

        os.makedirs(self.cfg.data_save_dir, exist_ok=True)
        with open(self.data_path, "wb") as f:
            pickle.dump(final_data, f)

        return final_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def course_collate_fn(batch):
    control_numbers = [item["control_number"] for item in batch]
    embeddings = torch.stack([item["embedding"] for item in batch])
    titles = [item["title"] for item in batch]
    descriptions = [item["description"] for item in batch]

    return {
        "control_number": control_numbers,
        "embedding": embeddings,
        "title": titles,
        "description": descriptions,
    }
