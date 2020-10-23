from flask import Flask, render_template, request
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from IPython.display import display
import string
import nltk
import re
import os
import numpy as np
import csv
import pandas as pd
app = Flask(__name__)


def preprocessing(csv):
	#Read csv dari list kalimat
	readCSV = open(csv, 'r', encoding='ISO-8859-1')
	#Menjadikkan list dari kalimat menjadi array
	youtube = list(readCSV)
	print("CASE FOLDING")
	#Case Folding
	youtubelower = []
	for line in youtube:
		a = line.lower()
		youtubelower.append(a)

	youtubenumber = []
	for line in youtubelower:
		result = re.sub(r'\d+', '', line)
		youtubenumber.append(result)

	print("STEMMING")
	#Stemming
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	youtubestemmed = map(lambda x: stemmer.stem(x),youtubenumber)
	youtube_no_punc = map(lambda x: x.translate(str.maketrans('','', string.punctuation)), youtubestemmed)
	youtube_no_punc = list(youtube_no_punc)

	#Stopwords dan Tokenizing
	stopwords = open('stoplist.txt', 'r').read()
	youtubetrain = []
	youtubefinal = []
	df = []

	print("STOPWORD")
	for line in youtube_no_punc:
		word_token = word_tokenize(line)
		word_token = [word for word in word_token if not word in stopwords and not word[0].isdigit()]
		youtubefinal.append(word_token)
		df.append(" ".join(word_token))

	return df

def naive_bayes(df):
	trainPositif = pd.read_csv("DataLatihPositifFIX.csv", usecols = ["nomor","word", "frekuensi", "ln"])
	trainNegatif = pd.read_csv("DataLatihNegativeFIX.csv", usecols = ["nomor","word", "frekuensi", "ln"])
	#Klasifikasi Naive Bayes Positif
	print("NAIVE BAYES")
	print("NB POSITIF - - - - - - -")
	nomorPositif = -1
	countAdaPositif = 0
	countAllPositif = 0
	unknownPositif = 0
	TambahKataLamaPositif = 0
	rumusPositif = 0
	probPositif = []
	JumlahPositif = 0;
	

	#Menghitung jumlah frekuensi positif
	for n in trainPositif['frekuensi']:
		JumlahPositif +=n

	for words in df:
		wordAll = word_tokenize(words)
		print (wordAll)
		for word in wordAll:
			countAllPositif += 1
			print ("count all",countAllPositif)
			
			for searchPositif in trainPositif['word']:
				nomorPositif += 1
				if word == searchPositif:
					print (searchPositif)
					print ("Fr :",trainPositif["frekuensi"][nomorPositif])
					print ("Ln :",trainPositif["ln"][nomorPositif])
					TambahKataLamaPositif += trainPositif["ln"][nomorPositif]
					countAdaPositif +=1
					print ("count Ada",countAdaPositif)
				unknownPositif = countAllPositif - countAdaPositif
			print ("kata baru ",unknownPositif)
			nomorPositif = -1
		print ("TambahKataLamaPositif :",TambahKataLamaPositif)
		rumusPositif = TambahKataLamaPositif + (unknownPositif*(np.log(1/JumlahPositif)))
		print ("hasil Klasifikasi :",rumusPositif)
		print ("\n")
		probPositif.append([words,rumusPositif])
		TambahKataLamaPositif = 0
		unknownPositif = 0
		countAllPositif = 0
		countAdaPositif = 0
		
	#Klasifikasi Naive Bayes Negatif
	print("NB NEGATIF - - - - - - -")
	nomorNegatif = -1
	countAdaNegatif = 0
	countAllNegatif = 0
	unknownNegatif = 0
	TambahKataLamaNegatif = 0
	rumusNegatif = 0
	probNegatif = []
	JumlahNegatif = 0;

	#Menghitung jumlah frekuensi positif
	for n in trainNegatif['frekuensi']:
		JumlahNegatif +=n


	for wordsNegatif in df:
		wordAll = word_tokenize(wordsNegatif)
		print (wordAll)
		for word in wordAll:
			countAllNegatif += 1
			print ("count all",countAllNegatif)
			
			#data negatif
			for searchNegatif in trainNegatif['word']:
				nomorNegatif += 1
				if word == searchNegatif:
					print (searchNegatif)
					print ("Fr :",trainNegatif["frekuensi"][nomorNegatif])
					print ("Ln :",trainNegatif["ln"][nomorNegatif])
					TambahKataLamaNegatif += trainNegatif["ln"][nomorNegatif] 
					countAdaNegatif +=1
					print ("count Ada",countAdaNegatif)
				unknownNegatif = countAllNegatif - countAdaNegatif
			print ("kata baru ",unknownNegatif)
			nomorNegatif = -1
		print ("TambahKataLamaNegatif :",TambahKataLamaNegatif)
		rumusNegatif = TambahKataLamaNegatif + (unknownNegatif*(np.log(1/17871)))
		print ("hasil Klasifikasi :",rumusNegatif)
		print ("\n")
		probNegatif.append([wordsNegatif,rumusNegatif])
		TambahKataLamaNegatif = 0
		unknownNegatif = 0
		countAllNegatif = 0
		countAdaNegatif = 0

	classification_result = []
	positif_total = 0
	negatif_total = 0
	with open ('Klasifikasi.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("teks","klasifikasi"))
		for n, PostifValue in enumerate(probPositif):
			print ("positif ",PostifValue)
			print ("Negatif ", probNegatif [n][1])
			if PostifValue[1] > probNegatif [n][1]:
				classification_result.append([probPositif[n][0],"POSITIF"])
				writer.writerow([probPositif[n][0],"POSITIF"])
				positif_total+=1
			elif PostifValue[1] == 0 and probNegatif ["nbNegatif"][n] == 0:
				print ("NULL")
			else:
				classification_result.append([probNegatif[n][0],"NEGATIF"])
				writer.writerow([probPositif[n][0],"NEGATIF"])
				negatif_total+=1
	
	return classification_result, positif_total, negatif_total

def training(file):
	dataTraining = pd.read_csv("Klasifikasi.csv", usecols = ["teks","klasifikasi"])
	TrainPositif1 = pd.read_csv("DataLatihPositifFIX.csv", usecols = ["nomor","word", "frekuensi", "ln"])
	TrainNegatif1 = pd.read_csv("DataLatihNegativeFIX.csv", usecols = ["nomor","word", "frekuensi", "ln"])
	FrBaru = 0
	number = 0 
	dataPositif = []
	dataNegatif = []
	word_token_positive_2 = []
	word_token_negative_2 = []
	youtubetrainP = []
	youtubetrainN = []
	countDataP = -1
	countDataN = -1
	DataTambahanPositif = []
	DataTambahanNegatif = []
	contDataPositif = 0
	contDataNegatif = 0
	anyDataPos=0
	anyDataNeg=0
	print(file)
	#Memisahkan Data Train Positif dan Negatif
	for num in file:		
		number = int(num)
		if dataTraining['klasifikasi'][number] == "POSITIF":
			dataPositif.append(dataTraining['teks'][number])
			anyDataPos=1
		else :
			dataNegatif.append(dataTraining['teks'][number])
			anyDataNeg=1


	if anyDataPos == 1:
		#Train Data Positif
		for line in dataPositif:
			word_token_positive_1 = word_tokenize(line)
			word_token_positive_2.append(word_token_positive_1)

		#Count Frekuensi
		for l in word_token_positive_2:
			youtubetrainP+= l
			final_youtubeP={v: youtubetrainP.count(v) for v in set(youtubetrainP)}

		
		#Memasukan Data Latih Ke array
		for data in TrainPositif1['word']:
			countDataP += 1
			DataTambahanPositif.append([data,TrainPositif1['frekuensi'][countDataP]])

		DataValueP = []
		datafrP = []

		#Remove data frekuensi lama
		for key, value in final_youtubeP.items():
			for w,f in DataTambahanPositif:
				if w == key:
					DataValueP.append([w,f])
					DataTambahanPositif.remove([w,f])
					
		n = 0;
		#Adding data frekuensi baru	(Data Sudah Ada)			
		for key, value in final_youtubeP.items():
			for w,f in DataValueP:
				if w == key:
					n +=1
					DataTambahanPositif.append([w,(f+value)])
			if n == 0:
				datafrP.append([key,value])
				print("data",key)
				print(n)
			n = 0

		#Adding data frekuensi baru	(Data Belum Ada)			
		for w,f in datafrP:
			if f > 1:
				DataTambahanPositif.append([w,f])
					
		#Menghitung jumlah frekuensi
		for n1, n2 in DataTambahanPositif:
			contDataPositif +=n2
			print(n1)
			print(n2)
			print("Total",contDataPositif)

		#Menghitung Data Latih Positif dan masuke ke file csv
		nomorUrut = 0
		with open ('DataLatihPositifFIX.csv','w',newline="") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(("nomor","word","frekuensi","ln"))
			for n1, n2 in DataTambahanPositif:
				Prob = (n2 + 1)/contDataPositif
				result = np.log(Prob)
				nomorUrut +=1
				writer.writerow([nomorUrut,n1,n2,result])
	else:
		print("Data Positif Tidak Ada")

	if anyDataNeg == 1:
		########################################
		#Train Data Negatif
		for line in dataNegatif:
			word_token_negatif_1 = word_tokenize(line)
			word_token_negative_2.append(word_token_negatif_1)

		#Count Frekuensi
		for l in word_token_negative_2:
			youtubetrainN+= l
			final_youtubeN={v: youtubetrainN.count(v) for v in set(youtubetrainN)}
		
		#Memasukan Data Latih Ke array
		for data in TrainNegatif1['word']:
			countDataN += 1
			DataTambahanNegatif.append([data,TrainNegatif1['frekuensi'][countDataN]])

		DataValueN = []
		datafrN = []

		#Remove data frekuensi lama
		for key, value in final_youtubeN.items():
			for w,f in DataTambahanNegatif:
				if w == key:
					DataValueN.append([w,f])
					DataTambahanNegatif.remove([w,f])
					
		n = 0;
		#Adding data frekuensi baru	(Data Sudah Ada)			
		for key, value in final_youtubeN.items():
			for w,f in DataValueN:
				if w == key:
					n +=1
					DataTambahanNegatif.append([w,(f+value)])
			if n == 0:
				datafrN.append([key,value])
				print("data",key)
				print(n)
			n = 0

		#Adding data frekuensi baru	(Data Belum Ada)			
		for w,f in datafrN:
			if f > 1:
				DataTambahanNegatif.append([w,f])
					
		#Menghitung jumlah frekuensi
		for n1, n2 in DataTambahanNegatif:
			contDataNegatif +=n2
			print(n1)
			print(n2)
			print("Total",contDataNegatif)

		#Menghitung Data Latih Positif dan masuke ke file csv
		nomorUrut = 0
		with open ('DataLatihNegativeFIX.csv','w',newline="") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(("nomor","word","frekuensi","ln"))
			for n1, n2 in DataTambahanNegatif:
				Prob = (n2 + 1)/contDataNegatif
				result = np.log(Prob)
				nomorUrut +=1
				writer.writerow([nomorUrut,n1,n2,result])
	else:
		print("Data Negatif Tidak Ada")

	return number

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST' :
		filename = request.form.get('csvfile')
		if filename != None:
			preprocessing_result = preprocessing(filename)
			classification_result, positif_total, negatif_total = naive_bayes(preprocessing_result)
			return render_template('index.html',classification_result=enumerate(classification_result),positif=positif_total,negatif=negatif_total)
		else:
			selected_users = request.form.getlist("validasiData")
			print("DATA MASUK ",selected_users)
			training(selected_users)
			return render_template('index.html')
	else :
		return render_template('index.html')
	

if __name__ == '__main__':
	app.run(debug=True)