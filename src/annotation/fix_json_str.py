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


def enum_list_to_json(enum_str) -> list:
    pattern = r'\d+\.\s*"question":\s*"([^"]+)",\s*\\n\s*"answer":\s*"([^"]+)"'
    matches = re.findall(pattern, enum_str)

    json_list = [{"question": match[0], "answer": match[1]} for match in matches]

    return json_list


if __name__ == "__main__":
    test_str = '''text=\'1. "question": "What services does CaPS provide for
graduate students?", \n                "answer": "CaPS provides counseling referrals with
in Carnegie Mellon or the Pittsburgh community."\n            2. "question": "How can app
ointments be made at University Health Services (UHS)?", \n                "answer": "App
ointments can be made in person, by telephone at 412-268-2922, or online through the UHS
website."\n            3. "question": "What is covered under the CMU Student Insurance Pl
an?", \n                "answer": "The plan covers most visit fees to see physicians and
advanced practice clinicians & nurse visits."\n            4. "question": "Who provides g
eneral medical care at University Health Services (UHS)?", \n                "answer": "P
hysicians, advanced practice clinicians, and registered nurses provide general medical ca
re at UHS."\n            5. "question": "What services does the Student Health Insurance
Program administer?", \n                "answer": "The program administers a high level o
f coverage in a wide network of healthcare providers and hospitals."\n            6. "que
stion": "How can appointments be made at Campus Wellness?", \n                "answer": "
Appointments can be made by visiting the Campus Wellness website, walk-in, or by telephon
e, 412-268-2157."\n            7. "question": "What is the role of registered dietitians
and health promotion specialists at University Health Services (UHS)?", \n
 "answer": "Registered dietitians and health promotion specialists assist students in add
ressing nutrition, drug and alcohol issues, and other healthy lifestyle concerns."\n
       8. "question": "What is the purpose of the Student Health Insurance Plan?", \n
            "answer": "The plan offers a high level of coverage to provide students with
access to quality medical care when needed."\n            9. "question": "How can student
s review detailed information about university health insurance requirements and fees?",
\n                "answer": "Students should visit the UHS website or their insurance pla
n for detailed information about the university health insurance requirement and fees."\n
            10. "question": "What is the belief of Carnegie Mellon regarding individual a
nd collective well-being?", \n                "answer": "Carnegie Mellon believes that ou
r individual and collective well-being is rooted in healthy connections to each other and
 campus resources."\''''

    clean_str = normalize_dirty_str(test_str)
    print(enum_list_to_json(clean_str))
