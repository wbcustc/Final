import read_data
import json
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer
import numpy as np

caption_path = '../CS5785-final-data/captions.json'
attri_path = '../CS5785-final-data/attributes_list_copy.txt'
st = LancasterStemmer()

def compare_attri(attributes, comment):
	result = np.zeros(102)
	for comment in comment:
		comment = comment.strip()
		comment = comment.split(" ")
		for t in comment:
			t = st.stem(t)
			for i in range(0, len(attributes)):
				for j in attributes[i]:
					for p in wn.synsets(j):
						if t == p:
							result[i] = 1
	return result
										

def calculate_attri10k(path):
	attributes_list = read_data.read_attributes_list(attri_path)
	for i in range(0, len(attributes_list)):
		attributes_list[i] = attributes_list[i].strip('\'')
		attributes_list[i] = attributes_list[i].split(' ')
		#for j in range(0, len(attributes_list[i])):
		#	attributes_list[i][j] = 
        caption = open(path)
        attri_10k = []
        for line in caption:
		print line
                s = json.loads(line)
                for key in s.keys():
                        comments = s[key]
                        attri = compare_attri(attributes_list, comments)
			print attri
                        attri_10k.append(attri)
	return attri_10k

if __name__ == '__main__':
	result = calculate_attri10k(caption_path)
