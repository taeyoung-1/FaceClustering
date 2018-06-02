arr_1 = []
for i in range(100):
    arr_1.append(i)

arr_2 = []
for i in range(100,200):
    arr_2.append(i)

k = [j for j, _ in zip(arr_1,arr_2) if j%2 is 0]
print(k)