from configs.config import Config
from src.datasets.course_dataset import CourseDataset
from src.taxonomy.taxonomy import CourseNode, Taxonomy


def to_course_nodes(dataset: CourseDataset) -> list[CourseNode]:
    return [
        CourseNode(
            control_number=row["control_number"],
            title=row["title"],
            description=row["description"],
        )
        for row in dataset
    ]


if __name__ == "__main__":
    cfg = Config()

    dataset = CourseDataset(cfg.data)
    course_nodes = to_course_nodes(dataset)

    taxonomy = Taxonomy(cfg.taxonomy, courses=course_nodes)
    taxonomy.save()
