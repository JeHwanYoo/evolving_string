import numpy as np
from numpy.random import default_rng  # 랜덤 뽑기, 중복 없이 뽑기
import random  # 확률적으로 랜덤 뽑기
import sys

answer = sys.argv[1]
low = 65
high = 123

dawkins = np.array(
    [ord(x) for x in answer if x != ' ']
)


space_position = [x[0] for x in enumerate(answer) if x[1] == ' ']


def chromosomes():
    return np.random.randint(
        low=low, high=high, size=len(dawkins)
    )


def init_population(r: int):
    return np.array([
        chromosomes() for _ in range(r)
    ])


def diff(x: np.ndarray):
    return np.subtract(dawkins, x)


def std(dif: np.ndarray, r=4):
    return round(np.std(dif), r)


def fitness(population: np.ndarray):
    return np.array([std(diff(x)) for x in population])


def description(label: str, min_loss: float, master_string: str):
    print('--------------------------------------------')
    print(label, 'min loss = ', min_loss)
    print(label, 'master = ', master_string)
    print('--------------------------------------------')


def ranking_selection(population: np.ndarray, fargs: np.ndarray, n: int):
    return population[fargs][:n]


def sex(s1: np.ndarray, s2: np.ndarray, mw: int = 99):
    p = np.zeros(len(s1), int)
    d1 = diff(s1)
    d2 = diff(s2)
    for i in range(len(s1)):
        [m] = random.choices(range(0, 2), weights=[mw, 1])
        if m == 0:  # 변이가 발생하지 않음
            p[i] = s1[i] if abs(d1[i]) < abs(d2[i]) else s2[i]
        else:  # 변이 발생
            p[i] = np.random.randint(low, high)
    return p


def crossover(selection: np.ndarray, r: int, mw: int = 99):
    high = len(selection)
    offspring = np.empty((0, len(selection[0])), int)
    while r > 0:
        rng = default_rng()
        x, y = rng.choice(high, size=2, replace=False)
        child = sex(selection[x], selection[y], mw)
        offspring = np.vstack((offspring, child))
        r -= 1
    return offspring


def ascii_to_str(gene: np.ndarray):
    a = list(map(chr, gene))
    for s in space_position:
        a.insert(s, ' ')
    return ''.join(a)


gen = 1  # 세대
n = 1000  # 인구수
m = 100  # 선택 수
mw = 99  # 변이 가중치 (mw:1)
population = init_population(n)  # 인구 초기화
while True:
    fit = fitness(population)  # 적합도 계산
    fargs = np.argsort(fit)  # 순위 매기기
    master_index = fargs[0]  # 대표 유전자
    master_string = ascii_to_str(population[master_index])  # 대표 문자열
    description(f'Generation {gen}', fit[fargs[0]], master_string)
    if np.array_equal(population[master_index], dawkins):  # 정답을 찾았는가?
        print(f'Generation {gen} found the answer: {answer}')
        break
    else:
        selection = ranking_selection(population, fargs, m)  # 순위 선택
        population = crossover(selection, n, mw)  # 교차
        gen += 1
