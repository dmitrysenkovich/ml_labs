import random

data = open('ex1data1.txt', 'w')
data_size = 1000
min_population = 10000
max_population = 100000
max_noise = 100000
k = 30
b = 500
for i in range(data_size):
	x = random.randint(min_population, max_population)
	noise = random.uniform(0, max_noise)
	y = k * x + b + noise
	data.write(str(x) + ' ' + str(y) + '\n')

data.close()
