import json
import re

log_file = r"C:\Users\Owner\.gemini\antigravity\brain\b18b908d-3027-4c7f-98c1-f870658ea19d\.system_generated\logs\overview.txt"

def extract_req(text):
    m = re.search(r'<USER_REQUEST>\n(.*?)\n</USER_REQUEST>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text[:200].replace('\n', ' ')

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if 'content' in data and data['source'] == 'USER_EXPLICIT':
                req = extract_req(data['content'])
                print(f"USER: {req}")
            elif 'content' in data and data['source'] == 'MODEL':
                print(f"MODEL: {data['content'][:100].replace(chr(10), ' ')}")
            elif 'message' in data and 'text' in data['message']:
                print(f"MODEL_TEXT: {data['message']['text'][:100].replace(chr(10), ' ')}")
        except:
            pass
