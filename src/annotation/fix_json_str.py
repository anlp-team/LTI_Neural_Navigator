import json
import re
import json_repair
from json_repair import repair_json


def normalize_dirty_str(dirty_str: str) -> str:
    if "```" in dirty_str:
        s = dirty_str.split("```")[1].strip()
        if s.startswith("json"):
            s = s[len("json"):].strip()
    else:
        s = dirty_str[len("text="):][1:-1].strip()
    return (s.replace("\\'", "'")
            .replace("\\n", "\n")
            .strip())


def fix_json_string(json_str):
    json_str = json_str.strip()
    json_str = re.sub(r',\s*]', ']', json_str)
    fixed_json_str = re.sub(r',\s*}', '}', json_str)
    fixed_json_str = fixed_json_str.strip()

    if not fixed_json_str.startswith('['):
        fixed_json_str = '[' + fixed_json_str
    if not fixed_json_str.endswith(']'):
        fixed_json_str += ']'

    try:
        json_obj = json.loads(fixed_json_str)
        return json.dumps(json_obj)
    except json.JSONDecodeError as e:
        print(f"Error fixing JSON: {e}")
        try:
            last_valid_index = fixed_json_str.rfind('}')
            valid_json_part = fixed_json_str[:last_valid_index + 1] + ']'
            json_obj = json.loads(valid_json_part)
            return json.dumps(json_obj)
        except json.JSONDecodeError as e:
            print(f"Failed to fix JSON: {e}")
            return None


def json_repair_api(json_str):
    json_str = normalize_dirty_str(json_str)
    good_json_string = repair_json(json_str, skip_json_loads=True)
    return good_json_string


if __name__ == "__main__":
    test_str = '''text='\n            ```\n            [\n                {\
n                    "question": "What are the course codes and instructors for the three
 Mathematical Studies Algebra I classes?",\n                    "answer": "The course cod
es and instructors for the three Mathematical Studies Algebra I classes are as follows: 1
2.0 Lec MWF 03:00PM 03:50PM GHC 4301, Instructor: Conley A T; 12.0 Lec MWF 01:00PM 01:50P
M WEH 8201, Instructor: Tice A R; and 12.0 Lec MWF 01:00PM 01:50PM WEH 8201, Instructor:
Tice A R."\n                },\n                {\n                    "question": "What
are the course codes for Matrix Algebra with Applications and Matrices and Linear Transfo
rmations?",\n                    "answer": "The course code for Matrix Algebra with Appli
cations is 10.0 Lec MWF 02:00PM 02:50PM DH 2302, Instructor: Koganemaru A T; and the cour
se code for Matrices and Linear Transformations is 11.0 Lec 1 MWF 09:00AM 09:50AM MM A14,
 Instructor: Gheorghiciuc."\n                },\n                {\n                    "
question": "What are the course codes and instructors for the five Linear Algebra classes
?",\n                    "answer": "The course codes and instructors for the five Linear
Algebra classes are as follows: 12.0 Lec MWF 03:00PM 03:50PM GHC 4301, Instructor: Conley
 A T; 12.0 Lec MWF 01:00PM 01:50PM WEH 8201, Instructor: Tice A R; 12.0 Lec MWF 01:00PM 0
1:50PM WEH 8201, Instructor: Tice A R; 12.0 Lec MWF 03:00PM 03:50PM GHC 4301, Instructor:
 Conley A T; and 12.0 Lecture MWF 09:00AM 09:50AM MM A14, Instructor: Gheorghiciuc."\n
             },\n                {\n                    "question": "What are the course
codes for Matrix Algebra with Applications and Matrices and Linear Transformations?",\n
                  "answer": "The course code for Matrix Algebra with Applications is 10.0
 Lec MWF 02:00PM 02:50PM DH 2302, Instructor: Koganemaru A T; and the course code for Mat
rices and Linear Transformations is 11.0 Lec 1 MWF 09:00AM 09:50AM MM A14, Instructor: Gh
eorghiciuc."\n                },\n                {\n                    "question": "Wha
t are the course codes and instructors for the five Linear Algebra classes?",\n
          "answer": "The course codes and instructors for the five Linear Algebra classes
 are as follows: 12.0 Lec MWF 03:00PM 03:50PM GHC 4301, Instructor'''

    # fixed_json_str = fix_json_string(normalize_dirty_str(test_str))
    fixed_json_str = json_repair_api(test_str)
    print(json.loads(fixed_json_str))
