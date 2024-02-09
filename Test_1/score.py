import time


def calculate_grad(score):
    time.sleep(1)
    if score >= 90: return 'A'
    if score >= 80: return 'B'
    if score >= 70: return 'C'
    if score >= 60: return 'E'
    return 'F'


calculate_grad(88)

print(calculate_grad)