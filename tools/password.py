import itertools

# 所有可能的字母
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'K']

# 定义一个函数，用于检查是否符合规则
def check(combination, clue, letters_correct, positions_correct):
    letters_count = sum([c in combination for c in clue])
    positions_count = sum([combination[i] == clue[i] for i in range(3)])
    return letters_count == letters_correct and positions_count == positions_correct

# 生成所有可能的三个字母的组合
combinations = list(itertools.permutations(letters, 3))

# 过滤出符合所有规则的组合
valid_combinations = []
for c in combinations:
    if (check(c, 'ABC', 1, 1) and
        check(c, 'AEF', 1, 0) and
        check(c, 'CKA', 2, 0) and
        check(c, 'DEB', 0, 0) and
        check(c, 'BDK', 1, 0)):
        valid_combinations.append(c)

# 打印所有有效的组合
for c in valid_combinations:
    print(''.join(c))
