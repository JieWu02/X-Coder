import requests
from time import time


def test_cpp(type):
    url = f'http://localhost:8000/{type}'
    data = {
        "type": "cpp",
        "solution": """#include <cstdio>
#include <unistd.h>
int main(){sleep(3);printf("a");return 0;}
""",
        "expected_output": "a"
    }
    response = requests.post(url, json=data)
    print(response.json())
    assert response.status_code == 200
    assert response.json()['run_success']
    assert response.json()['success']

    data = {
        "type": "cpp",
        "solution": """#include <cstdio>
int main(){printf("a");return 0;}
""",
        "expected_output": "b"
    }
    response = requests.post(url, json=data)
    print(response.json())
    assert response.status_code == 200
    assert response.json()['run_success']
    assert not response.json()['success']


def test_python(type):
    url = f'http://localhost:8000/{type}'
    data = {
        "type": "python",
        "solution": "print(input())",
        "input": "a",
        "expected_output": "a"
    }
    response = requests.post(url, json=data)
    print(response.json())
    assert response.status_code == 200
    assert response.json()['success']
    assert response.json()['run_success']

    data = {
        "type": "python",
        "solution": "print(input())",
        "input": "a",
        "expected_output": "b"
    }
    response = requests.post(url, json=data)
    print(response.json())
    assert response.status_code == 200
    assert not response.json()['success']
    assert response.json()['run_success']


def test_python_timeout(type):
    url = f'http://localhost:8000/{type}'
    data = {
        "type": "python",
        "solution": "print(input())",
        "input": "",
        "expected_output": "a"
    }
    response = requests.post(url, json=data)
    print(response.json())
    assert response.status_code == 200
    assert not response.json()['success']
    assert not response.json()['run_success']


def test_batch(type):
    url = f'http://localhost:8000/{type}/batch'
    data = {
        'type': 'batch',
        "submissions": [{
        "type": "cpp",
        "solution": """#include <cstdio>
int main(){printf("a");return 0;}
""",
        "expected_output": "b"
        }, {
        "type": "python",
        "solution": "print(input())",
        "input": "a",
        "expected_output": "b"
        }, {
            "type": "python",
            "solution": "print(input())",
            "input": "a",
            "expected_output": "a"
        }, {
            "type": "cpp",
            "solution": """#include <cstdio>
#include <unistd.h>
int main(){sleep(3);printf("a");return 0;}
""",
            "expected_output": "a"
        }]
    }
    response = requests.post(url, json=data)
    print(response.json())
    assert response.status_code == 200
    results = response.json()['results']

    assert len(results) == 4
    assert not results[0]['success']
    assert results[0]['run_success']
    assert not results[1]['success']
    assert results[1]['run_success']
    assert results[2]['success']
    assert results[2]['run_success']
    assert results[3]['success']
    assert results[3]['run_success']


def test_batch_timeout(type):
    url = f'http://localhost:8000/{type}/batch'
    data = {
        'type': 'batch',
        "submissions": [{
        "type": "python",
        "solution": "print(input())",
        "input": "",
        "expected_output": "b"
        }, {
            "type": "python",
            "solution": "print(input())",
            "input": "a",
            "expected_output": "a"
        }, {
        "type": "python",
        "solution": "print(input())",
        "input": "",
        "expected_output": "b"
        }, {
            "type": "python",
            "solution": "print(input())",
            "input": "a",
            "expected_output": "a"
        },{
        "type": "python",
        "solution": "print(input())",
        "input": "",
        "expected_output": "b"
        }, {
            "type": "python",
            "solution": "print(input())",
            "input": "a",
            "expected_output": "a"
        }]
    }
    response = requests.post(url, json=data)
    print(response.json())
    assert response.status_code == 200
    results = response.json()['results']

    assert len(results) == 6
    assert not results[0]['success']
    assert not results[0]['run_success']
    assert results[1]['success']
    assert results[1]['run_success']
    assert not results[2]['success']
    assert not results[2]['run_success']
    assert results[3]['success']
    assert results[3]['run_success']
    assert not results[4]['success']
    assert not results[4]['run_success']
    assert results[5]['success']
    assert results[5]['run_success']

def test_all(type):
    start = time()
    test_batch_timeout(type)
    end = time()
    print(f'Time taken: {end - start} seconds')

    test_cpp(type)
    test_python(type)
    test_batch(type)
    test_python_timeout(type)


if __name__ == '__main__':
    test_all('run')
    test_all('judge')
