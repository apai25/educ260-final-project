from openai import OpenAI
import pandas as pd
import random
import re
import difflib  # added for fuzzy matching


client = OpenAI(api_key=API_KEY)
cc_data = pd.read_csv("/home/awang/educ260-final-proj-data/course_catalog_202401161416.csv")

def hierarchy(department, cc_data, num_layers):
    dept_level = cc_data[cc_data["department_name"] == department]

    course_dict = dict(zip(dept_level['title'], dept_level['catalogue_description']))

    def get_closest_title(given_title, course_dict_keys):
        matches = difflib.get_close_matches(given_title, course_dict_keys, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def recursive_labeling(course_dict, depth):
        #print(f"\nRecursive call at depth {depth} with {len(course_dict)} courses")

        if depth == 0 or not course_dict:
            return course_dict  # base case: return the raw courses

        sampled = dict(random.sample(list(course_dict.items()), min(100, len(course_dict))))

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                #start with 6 subtopic classifications
                {"role": "system", "content": "Provided below is a sample of courses from a course catalog database that are all offered through the same department at a community college. The objective of the task is to utilize the semantics of the course title and description to further classify the courses into smaller groups based on the specificity of the content covered. Using the course department, title, and description specified in the user prompt below, generate a bulleted list of only the 5 subtopics that best divide the sample of courses into subgroups, but still generalizable enough to apply to all courses offered through a particular department. "},
                {"role": "user", "content": f"Department = {department}, Course information = {course_dict}"}
            ]
        )

        # Debug: Print raw subtopic response
        #print("Subtopics raw response:\n", response.choices[0].message.content)

        subtopics = [line.strip("-• ") for line in response.choices[0].message.content.splitlines() if line.strip()]

        assignment_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Using only subtopics listed here {subtopics}, assign every course in the dictionary of the user prompt to the most similar subcategory using the course title and description. List every subtopic as a heading, and under each, include a bulleted list starting with ### of all course titles from the course dictionary, case sensitive, that belong in that subtopic. Include every course into only one of the subcategories."},
                {"role": "user", "content": f"Department = {department}, Course information = {course_dict}"}
            ]
        )

        #debug : print raw assignment response 
        #print("assignment raw response:\n", assignment_response.choices[0].message.content)

        assignment_text = assignment_response.choices[0].message.content.strip()

        current_subtopic = None
        subtopic_groups = {}
        for line in assignment_text.splitlines():
            if not line.strip():
                continue
            if line.strip().find('###') !=-1:
                current_subtopic = line.strip().rstrip(":")
                subtopic_groups[current_subtopic] = []
            elif current_subtopic and line.strip().find('- ') != -1:
                course_title = line.strip("-• ").strip()
                matched_title = get_closest_title(course_title, course_dict.keys())  # fuzzy match
                if matched_title:
                    subtopic_groups[current_subtopic].append(matched_title)
                else:
                    print(f"[WARN] Could not match course title: '{course_title}'")


        hierarchy_tree = {}
        for sub, titles in subtopic_groups.items():
            sub_dict = {title: course_dict[title] for title in titles}
            hierarchy_tree[sub] = recursive_labeling(sub_dict, depth - 1)

        return hierarchy_tree

    return {department: recursive_labeling(course_dict, num_layers)}


result = hierarchy("CS", cc_data, num_layers=3)

# def print_hierarchy(data, indent=0):
#     for key, value in data.items():
#         prefix = "  " * indent
#         # If the value is a dict, keep descending
#         if isinstance(value, dict):
#             print(f"{prefix}{key}")
#             print_hierarchy(value, indent + 1)
#         else:
#             # Assume it's a course description
#             short_desc = value.split('.')[0] + '.' if isinstance(value, str) else ''
#             print(f"{prefix}- {key}: {short_desc}")

# print_hierarchy(result)

import json

with open("hierarchy_output.json", "w") as f:
    json.dump(result, f, indent=2)