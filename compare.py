import csv
reader1 = csv.reader(open('bernoulliNB_alexnet.csv','rb'))
reader2 = csv.reader(open('svm_result.csv','rb'))
header1 = reader1.next()
header2 = reader2.next()
compare = dict()
num = 0
for picname,label in reader1:
	compare[picname] = label
for picname,label in reader2:
	if compare[picname] == label:
		num = num + 1
print num
