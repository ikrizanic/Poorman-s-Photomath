from collections import deque

OPERATORS = ["/", "*", "-", "+"]


def main():
    print(parse_and_solve("2*(3-2*9)"))


def parse_and_solve(string):
    try:
        token_list = fill_token_list(string)
        reverse_polish = return_reverse_polish(token_list)
        return solve_reverse_polish(reverse_polish)
    except:
        return None


def fill_token_list(string):
    stack = deque()
    num = ""
    for c in string:
        if c.isdigit():
            num += c
        else:
            if num != "":
                stack.append(num)
                num = ""
            stack.append(c)
    if num != "":
        stack.append(num)
    return stack


def return_reverse_polish(tokens):
    operator_stack = deque()
    output_stack = deque()
    while len(tokens) != 0:
        element = tokens.popleft()
        if element.isnumeric():
            output_stack.append(element)
        elif element in OPERATORS:
            if len(operator_stack) == 0:
                operator_stack.append(element)
                continue
            if element in "+-":
                while operator_stack[-1] in "*/":
                    output_stack.append(operator_stack.popleft())
            operator_stack.append(element)
        elif element == '(':
            operator_stack.append(element)
        elif element == ')':
            while operator_stack[-1] != '(':
                output_stack.append(operator_stack.pop())
            operator_stack.pop()
    while len(operator_stack) != 0:
        output_stack.append(operator_stack.pop())
    return output_stack


def solve_reverse_polish(stack):
    try:
        result = deque()
        while len(stack) != 0:
            el = stack.popleft()
            if el.isnumeric():
                result.append(el)
            else:
                b, a = float(result.pop()), float(result.pop())
                if el == "*":
                    result.append(a * b)
                elif el == "/":
                    result.append(a / b)
                elif el == "+":
                    result.append(a + b)
                else:
                    result.append(a - b)
        return result.pop()
    except:
        return None


if __name__ == '__main__':
    main()
