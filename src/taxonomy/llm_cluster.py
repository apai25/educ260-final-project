import json
import os
import re
import time
from typing import Dict, List

from openai import OpenAI

from src.taxonomy.course import CourseNode

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


from math import ceil

def get_llm_response(
    prompt: str,
    model: str) -> str:
    SUPPORTED_MODELS = [
        "gpt-4o",
        "o3-mini",
        "o4-mini",
        "o1-mini",
    ]

    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model} is not supported. Supported models are: {SUPPORTED_MODELS}")
    

    if model == "gpt-4o":
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    
    response = completion.choices[0].message.content.strip()
    response = remove_wrappers(response)
    return response
    

def cluster_fn(
    courses: List[CourseNode],
    k: int,
    topic_path: List[str],
    model: str = "gpt-4o",
    batch_size: int = 50,
) -> Dict[str, List[CourseNode]]:
    ## Stage 1: Generate Topic Labels
    label_prompt = build_label_gen_prompt(courses, k, topic_path)
    for attempt in range(3):
        try:
            label_raw = get_llm_response(label_prompt, model)
            topics = json.loads(label_raw)
            if not isinstance(topics, list) or len(topics) != k:
                raise ValueError(
                    f"Expected list of {k} topics, got: {len(topics), topics}"
                )
            break
        except Exception as e:
            print(f"[cluster_fn] Label generation failed (attempt {attempt + 1}): {e}")
            time.sleep(2)
    else:
        raise RuntimeError("Failed to generate topic labels after 3 attempts.")

    ## Stage 2: Batched Course Assignment
    assignments: Dict[str, str] = {}
    total_batches = ceil(len(courses) / batch_size)

    for batch_idx in range(total_batches):
        batch_courses = courses[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        assign_prompt = build_assignment_prompt(batch_courses, topics)

        for attempt in range(3):
            try:
                assignment_raw = get_llm_response(assign_prompt, model)
                batch_assignments = json.loads(assignment_raw)

                if not isinstance(batch_assignments, dict) or len(
                    batch_assignments
                ) != len(batch_courses):
                    raise ValueError(
                        f"Expected dict of {len(batch_courses)} assignments, got: {len(batch_assignments), batch_assignments}"
                    )

                assignments.update(batch_assignments)
                break
            except Exception as e:
                print(
                    f"[cluster_fn] Assignment failed on batch {batch_idx + 1}/{total_batches} (attempt {attempt + 1}): {e}"
                )
                time.sleep(2)
        else:
            raise RuntimeError(
                f"Failed to assign courses for batch {batch_idx + 1} after 3 attempts."
            )

    ## Construct cluster dict
    topic_to_courses: Dict[str, List[CourseNode]] = {t: [] for t in topics}
    course_map = {c.control_number: c for c in courses}

    for cid, topic in assignments.items():
        if cid in course_map and topic in topic_to_courses:
            topic_to_courses[topic].append(course_map[cid])

    return topic_to_courses


def build_label_gen_prompt(
    courses: List[CourseNode], k: int, topic_path: List[str]
) -> str:
    path_str = " > ".join(topic_path) if topic_path else "All Courses"
    serialized_courses = "\n".join(
        f"- {c.control_number}: {c.title} — {c.description}" for c in courses
    )
    return f"""You are building a hierarchical course taxonomy.

    Current topic path: {path_str}
    You are at depth {len(topic_path)} of the taxonomy.

    Given the following university courses return exactly {k} distinct and
    coherent subtopic labels that could be used to cluster the courses. The granularity 
    of the subtopics should be appropriate for a university course taxonomy, and you should choose the level
    of specificity based on the current topic path, and the depth you are currently evaluating courses at. A deeper
    depth should have more specific subtopics, while a shallower depth should have broader subtopics.
    (I.e, dental assisting on the first level is too specific, but on the second level might be appropriate.)
    (I.e, mathematics on the first level might be appropriate)

    Courses:
    {serialized_courses}

    Respond in **valid Python list** format. Use double quotes for strings
    {["Topic A", "Topic B", ...]}

    Just return the raw string-representation of the list, without any additional
    formatting.
    """


def build_assignment_prompt(courses: List[CourseNode], topics: List[str]) -> str:
    serialized_courses = "\n".join(
        f"- {c.control_number}: {c.title} — {c.description}" for c in courses
    )
    return f"""You are building a hierarchical course taxonomy.

    Given the following university courses, assign each course to one of the following
    subtopics.
    Each course must be assigned to exactly one of the below subtopics. Spell each
    subtopic exactly as given (case-sensitive).

    Courses:
    {serialized_courses}

    Subtopics:
    {topics}

    Respond in **valid Python dict** format. Use double quotes for strings:
    {{"course1": "Topic A", "course2": "Topic B", ...}}

    Just return the raw string-representation of the dict, without any additional
    formatting. Be careful to ensure ALL courses are assigned to one of the subtopics; do not leave any of them out in your response.
    Go about this sequentially, assigning each course to one of the topics in order. 
    """


def remove_wrappers(text: str) -> str:
    match = re.search(r"```(?:json|python)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
