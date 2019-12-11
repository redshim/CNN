# Tensorflow install : 1.14
# matplotlib, scipy, scikit-learn, sklearn

import tensorflow as tf
import numpy as np

# ctrl + shift + F10 : 실행
# ctrl + / : 주석

a = [1,3,5, 'abc']
print(a)
print(a[0],a[3])
print(a[1:3])
print(a[0:len(a)])
print(a[:])
print(a[::2]) #시작,종료,증감
print(a[-1],a[-2]) #역순
print(a[::-1]) #종료,시작,증감(역방향)

print('+'*30)

print(a[0],a[1],a[2],a[3])

print(list(range(0,4,1)))
print(list(range(0,4)))
print(list(range(4)))

for i in range(len(a)):
    print(i, a[i])

for i in a:
    print(i, end=' ')
print()

for i, v in enumerate(a, start=10):
    print(i,v, end=' ')
print()

d = {'name':'hoon', 'age':21}
print(d)
print(d['name'])

# 컴프리헨션
# 집계함수에 전달할 리스트를 만드는 한줄짜리 반복문
# 가독성이 좋아짐
for i in range(5):
    if i % 2:
        i

b = [i for i in range(5)]
print(b)
print(sum([i for i in range(5)]))

# 문제
# 홀수합계
c = [31, 17, 26, 41, 12]

for i in c:
    if i % 2 ==0:
        print(i)

print([i for i in c if i % 2])
print(sum([i for i in c if i % 2]))

print('+'*30)

e = np.arange(10)
print(e)
print(e.dtype, e.shape)
print(type(e))

#전달한 Shape을 갖고 감
print(e + 1)        # broadcast
print(e + e)        # vector 연산 shape 어긋나면 문제
print(np.sin(e))    # universal function



print('+'*30)

f = 1
print(e)
print(e+1)
print(e+f)



g = np.array([[1,3,5,7],
              [2,6,0,4],
              [2,3,7,9]
              ])
print(g)

h = g.reshape(-1)
print(h)

i  = [True, False] * 6
print(i)

print(h[i])
print(h[h>5])
#반복문을 돌면서 계산해야하는 것들을 Index로 계산

j = np.arange(12)
print(j)
print(h[j])
print(h[j[:6]])
print(h[[3,5,1]])

# np.random.shuffle(h)
# print(h)

np.random.shuffle(j)
print(j)
print(h[j])

