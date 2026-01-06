response = """
for i in range(0, 4):
    if(i+1 == 4):
        max_symbols = 26
    for root in recursion(max_symbols):
        add_recursion_into(grammar)
        i += 1

print(print_linear_grammar(grammar))
"""
import re

answer_pattern = r'<answer>(.*?)</answer>'
matches = list(re.finditer(answer_pattern, response, re.DOTALL))
if not matches:
    print("failed")
# answer = matches[-1].group(1).strip()

# print(answer)

test_list = []
test_list.append()