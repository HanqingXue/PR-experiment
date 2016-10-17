f = open('ClusterSamples.csv')
count = 1
data = {}
L = []
T = []
for item in f:
	data[item[0:-1]] = count
	t = item[0:-1].split(',')
	L.append(t)
	count += 1
f.close()

'''
print type(L[0])
f2 = open('out.txt')
name = []

def strip(s=str):
	return s.lstrip()

for elm in f2:
	if len(elm) > 10:
		name.append(elm[1:-2])
		t = elm[1:-2].split(',')
		T.append(t)

L2 = []
for item in T:
	t = map(strip, item)
	L2.append(t)

o = open('map.txt', 'w')
for item in L2:
	out = str(item).replace('[', '')
	out = out.replace(' ', '')
	out = out.replace('\'', '')
	o.write(out)
'''
m = open('map.txt')
for item in m:
	print item
	print data[item]
	break






