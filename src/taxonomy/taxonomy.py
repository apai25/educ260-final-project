from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from configs.taxonomy_config import TaxonomyConfig
from src.taxonomy.course import CourseNode
from src.taxonomy.llm_cluster import cluster_fn


@dataclass
class TaxonomyNode:
    topic: str
    parent: Optional[TaxonomyNode] = None
    children: List[TaxonomyNode] = field(default_factory=list)
    courses: Optional[List[CourseNode]] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, child: TaxonomyNode) -> None:
        self.children.append(child)

    def assign_course(self, course: CourseNode) -> None:
        if self.courses is None:
            self.courses = []
        self.courses.append(course)


class Taxonomy:
    def __init__(self, cfg: TaxonomyConfig, courses: List[CourseNode] = None):
        self.cfg = cfg
        self.save_dir = cfg.save_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if len(courses) > cfg.max_courses:
            courses = random.sample(courses, cfg.max_courses)
        self.root = TaxonomyNode(topic="root", courses=courses)

        self.ccn_to_taxonomy = {}

        if cfg.cluster_fn == "llm":
            self.cluster_fn = lambda courses, k, topic_path: cluster_fn(
                courses, k, topic_path, model=cfg.model, batch_size=cfg.batch_size
            )
        else:
            raise ValueError(f"Unknown clustering function: {cfg.cluster_fn}")

        self.build(
            self.root,
            cfg.num_subgroups,
        )

    def build(self, node: TaxonomyNode, levels_remaining: List[int]) -> None:
        if node.parent:
            node.parent.courses = []

        if not levels_remaining:
            return

        clusters = self.cluster_fn(
            node.courses, levels_remaining[0], self.get_topic_path(node)
        )

        for topic, courses in clusters.items():
            child = TaxonomyNode(topic=topic, parent=node, courses=courses)
            node.add_child(child)
            self.ccn_to_taxonomy.update(
                {course.control_number: child for course in courses}
            )
            self.build(child, levels_remaining[1:])

    def get_topic_path(self, node: TaxonomyNode) -> List[str]:
        path = []
        while node:
            path.append(node.topic)
            node = node.parent
        return list(reversed(path))

    def save(self) -> dict:
        with open(self.save_dir / "taxonomy.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        return self._to_dict(self.root)

    def to_dict(self) -> dict:
        return self._to_dict(self.root)

    def _to_dict(self, node: TaxonomyNode) -> dict:
        result = {
            "topic": node.topic,
            "courses": [course.control_number for course in node.courses]
            if node.courses
            else None,
            "children": [self._to_dict(child) for child in node.children],
        }
        return result
