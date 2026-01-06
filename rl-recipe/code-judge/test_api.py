import requests


code_str = """
def cal(x):
    return x+5
    
assert cal(5) == 10
"""

submission = {
    "type": "python",
    "solution": code_str,
    "expected_output": "",
}
submissions = []
submissions.append(submission)
data = {
    "type": "batch",
    "submissions": submissions
}

response = requests.post("http://0.0.0.0:8005/judge/long-batch", json=data)
response.raise_for_status()

results = response.json()
print(results)

