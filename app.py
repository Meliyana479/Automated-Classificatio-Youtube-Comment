from flask import Flask, render_template, request
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from googleapiclient.discovery import build
from scipy.stats import chi2_contingency
from nltk.tokenize import word_tokenize
from scipy.stats import kstest, shapiro
from scipy.stats.stats import pearsonr
from IPython.display import display
import statsmodels.api as sm
from scipy.stats import chi2
import logging
import logging.config
import pandas as pd
import numpy as np
import statistics
import string
import tablib
import nltk
import time
import re
import os
import csv

app = Flask(__name__)


def preprocessing(csvFileData):
	#Read csv dari list kalimat
	readCSV = open(csvFileData, 'r', encoding='ISO-8859-1')
	#Menjadikkan list dari kalimat menjadi array
	youtube = list(readCSV)
	

	print("CASE FOLDING")
	#Case Folding
	#Mengubah sentimen menjadi huruf kecil
	youtubelower = []
	for line in youtube:
		line = re.sub(r'[^\w\s]',' ', line)
		line = " ".join(line.split())
		a = line.lower()
		youtubelower.append(a)

	#Menghilangkan angka pada sentiment
	youtubenumber = []
	for line in youtubelower:
		result = re.sub(r'\d+',' ', line)
		youtubenumber.append(result)


	print("STEMMING")
	#Stemming
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	youtubestemmed = map(lambda x: stemmer.stem(x),youtubenumber)
	youtube_no_punc = map(lambda x: x.translate(str.maketrans(' ',' ', string.punctuation)), youtubestemmed)
	youtube_no_punc = list(youtube_no_punc)

	#Stopwords dan Tokenizing
	youtubetrain = []
	youtubefinal = []
	Under3 = []
	df = []

	datafile =  open('stop.txt', 'a+',)
	for line in youtube_no_punc:
		word_token = word_tokenize(line)
		for w in word_token:
			qtyWord = len(w)
			if qtyWord <=3:
				Under3.append(w)
				datafile.write("%s\r\n" % (w))

	datafile.close()
	stopwords = open('stop.txt', 'r').read()

	print("STOPWORD")
	for line in youtube_no_punc:
		word_token = word_tokenize(line)
		word_token = [word for word in word_token if not word in stopwords and not word[0].isdigit()]
		youtubefinal.append(word_token)
		df.append(" ".join(word_token))

	with open ('d:/Website/dataUjiBaru/ResultPreprocessing.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		for dataW in df:
			writer.writerow([dataW," "])

	return df

def naive_bayes(df):
	trainPositif = pd.read_csv("d:/Website/dataLatih/FrekuensiPositiveTrainComment.csv", usecols = ["nomor","word", "frekuensi", "ln"])
	trainNegatif = pd.read_csv("d:/Website/dataLatih/FrekuensiNegativeTrainComment.csv", usecols = ["nomor","word", "frekuensi", "ln"])

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
	

	#Menghitung jumlah frekuensi data latih positif
	for n in trainPositif['frekuensi']:
		JumlahPositif +=n
		JumlahPositif += 1
	print("Data Fr Positif ", JumlahPositif)

	#Klasifikasi dilakukan pada masing2 kalimat
	for words in df:
		wordAll = word_tokenize(words)
		
		for word in wordAll:
			#Menghitung jumlah keselurahan kata pada sebuah kalimat
			countAllPositif += 1
			
			
			#Mencari kata pada data latih untuk dihitung 
			for searchPositif in trainPositif['word']:
				nomorPositif += 1
				if word == searchPositif:
					
					#Menghitung (Menambah) nilai ln probabilitas pada kata yang terdapat pada data latih positif
					TambahKataLamaPositif += trainPositif["ln"][nomorPositif]
					

					#Menghitung jumlah kata yang ada pada data latih
					countAdaPositif +=1
					

				#Jumlah kata yang tidak ada pada data latih (kata baru)
				unknownPositif = countAllPositif - countAdaPositif
			nomorPositif = -1
		

		#Rumus Menghitung Ln probabilitas Naive Bayes untuk klasifikasi sentimen positif
		rumusPositif = TambahKataLamaPositif + (unknownPositif*(np.log(1/JumlahPositif)))
		
		
		#Memasukan ke array
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

	#Menghitung jumlah frekuensi data latih negatif
	for n in trainNegatif['frekuensi']:
		JumlahNegatif +=n
		JumlahNegatif += 1
	print("Data Fr Negatif ", JumlahNegatif)

	#Klasifikasi dilakukan pada masing2 kalimat
	for wordsNegatif in df:
		wordAll = word_tokenize(wordsNegatif)
		#print (wordAll)
		for word in wordAll:

			#Menghitung jumlah keselurahan kata pada sebuah kalimat
			countAllNegatif += 1
		
			
			#Mencari kata pada data latih untuk dihitung 
			for searchNegatif in trainNegatif['word']:
				nomorNegatif += 1
				if word == searchNegatif:
		

					#Menghitung (Menambah) nilai ln probabilitas pada kata yang terdapat pada data latih negatif
					TambahKataLamaNegatif += trainNegatif["ln"][nomorNegatif]

					#Menghitung jumlah kata yang ada pada data latih
					countAdaNegatif +=1


				#Jumlah kata yang tidak ada pada data latih (kata baru)
				unknownNegatif = countAllNegatif - countAdaNegatif

			nomorNegatif = -1


		#Rumus Menghitung Ln probabilitas Naive Bayes untuk klasifikasi sentimen negatif
		rumusNegatif = TambahKataLamaNegatif + (unknownNegatif*(np.log(1/JumlahNegatif)))


		#Memasukan ke array
		probNegatif.append([wordsNegatif,rumusNegatif])
		TambahKataLamaNegatif = 0
		unknownNegatif = 0
		countAllNegatif = 0
		countAdaNegatif = 0

	classification_result = []
	positif_total = 0
	negatif_total = 0

	#Melakukan Proses Klasifikasi
	#Mengecek nilai ln probabilitas sentimen positif dan negatif
	#nilai ln probabilitas terkecil adalah hasil klasifikasi
	with open ('d:/Website/dataUjiBaru/ResultKlasifikasi.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("teks","lnPositif","lnNegatif","klasifikasi"))
		for n, PostifValue in enumerate(probPositif):
			if PostifValue[1] > probNegatif [n][1]:
				cvtP = '{:02.3f}'.format(PostifValue[1])
				cvtN = '{:02.3f}'.format(probNegatif [n][1])
				classification_result.append([probPositif[n][0],cvtP,cvtN,"POSITIVE"])
				writer.writerow([probPositif[n][0],cvtP,cvtN,"POSITIVE"])
				positif_total+=1
			elif PostifValue[1] < probNegatif [n][1]:
				cvtP = '{:02.3f}'.format(PostifValue[1])
				cvtN = '{:02.3f}'.format(probNegatif [n][1])
				classification_result.append([probNegatif[n][0],cvtP,cvtN,"NEGATIVE"])
				writer.writerow([probNegatif[n][0],cvtP,cvtN,"NEGATIVE"])
				negatif_total+=1		
			else:
				print("NULL")

	
	return classification_result, positif_total, negatif_total

def training(file):
	dataTraining = pd.read_csv("d:/Website/dataUjiBaru/ResultKlasifikasi.csv", usecols = ["teks","klasifikasi"])
	TrainPositif1 = pd.read_csv("d:/Website/dataLatih/FrekuensiPositiveTrainComment.csv", usecols = ["nomor","word", "frekuensi", "ln"])
	TrainNegatif1 = pd.read_csv("d:/Website/dataLatih/FrekuensiNegativeTrainComment.csv", usecols = ["nomor","word", "frekuensi", "ln"])
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

	#Memisahkan Data latih sesuai dengan label hasil klasifikasi (Sentimen positif/negatif)
	newFilePositif = open('d:/Website/dataLatih/DataLatihPositiveTest.csv', 'a+',newline='')
	newFileN = open('d:/Website/dataLatih/DataLatihNegativeTest.csv', 'a+',newline='')
	newFileWriterP = csv.writer(newFilePositif)
	newFileWriterN = csv.writer(newFileN)
	for num in file:		
		number = int(num)
		if dataTraining['klasifikasi'][number] == "POSITIVE":
			dataPositif.append(dataTraining['teks'][number])
			newFileWriterP.writerow([dataTraining['teks'][number],""])
			anyDataPos=1
		else :
			dataNegatif.append(dataTraining['teks'][number])
			newFileWriterN.writerow([dataTraining['teks'][number],""])
			anyDataNeg=1
	
	
	newFilePositif.close()
	newFileN.close()

	#Akan dieksekusi jika ada data pada array
	#Validasi data positif
	if anyDataPos == 1:
		#Tokenizing
		for line in dataPositif:
			word_token_positive_1 = word_tokenize(line)
			word_token_positive_2.append(word_token_positive_1)

		#Count Frekuensi kata pada data sentimen yang di validasi (Positif)/TDM
		for l in word_token_positive_2:
			youtubetrainP+= l
			final_youtubeP={v: youtubetrainP.count(v) for v in set(youtubetrainP)}

		
		#Memasukan Data Latih Ke array (Positif)
		for data in TrainPositif1['word']:
			countDataP += 1
			DataTambahanPositif.append([data,TrainPositif1['frekuensi'][countDataP]])

		DataValueP = []
		datafrP = []

		#Remove data frekuensi lama (pada data latih positif)
		#Untuk digantikan dengan data frekuensi baru
		for key, value in final_youtubeP.items():
			for w,f in DataTambahanPositif:
				if w == key:
					#Data Latih Baru Yang ada pada data latih lama (Untuk di replace data baru)
					DataValueP.append([w,f])
					DataTambahanPositif.remove([w,f])
					
		n = 0;
		#Adding data frekuensi baru	(Data Yang Sudah Ada)			
		for key, value in final_youtubeP.items():
			for w,f in DataValueP:
				if w == key:
					n +=1
					DataTambahanPositif.append([w,(f+value)])
			#Jika data n = 0 artinya kata baru tidak terdapat pada data latih lama
			#Disimpan pada array baru (kata baru yang tidak ada di data lama)
			if n == 0:
				datafrP.append([key,value])
			n = 0

		#Adding data frekuensi baru	(Data Belum Ada di Data Latih Positif)
		#Ekstraksi Fitur		
		for w,f in datafrP:
			if f > 3:
				print("kata ",w)
				print("fr ",f)
				DataTambahanPositif.append([w,f])
					
		#Menghitung jumlah frekuensi, hasil dari penambahan data baru.
		#Disimpan dalam array dulu
		for n1, n2 in DataTambahanPositif:
			contDataPositif += n2
			contDataPositif += 1

		#Menghitung (ln probabilitas) Data Latih Positif dan masukan ke file csv
		nomorUrut = 0
		with open ('d:/Website/dataLatih/FrekuensiPositiveTrainComment.csv','w',newline="") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(("nomor","word","frekuensi","ln"))
			for n1, n2 in DataTambahanPositif:
				Prob = (n2 + 1)/contDataPositif
				result = np.log(Prob)
				result2 = '{:01.3f}'.format(result)
				nomorUrut +=1

				writer.writerow([nomorUrut,n1,n2,result2])
	#Jika tidak ada data pada array positif
	else:
		print("Data Positif Tidak Ada")


	#Akan dieksekusi jika ada data pada array
	#Validasi data negatif
	if anyDataNeg == 1:
		########################################
		#Tokenizing
		for line in dataNegatif:
			word_token_negatif_1 = word_tokenize(line)
			word_token_negative_2.append(word_token_negatif_1)

		#Count Frekuensi kata pada data sentimen yang di validasi (Negatif)
		for l in word_token_negative_2:
			youtubetrainN+= l
			final_youtubeN={v: youtubetrainN.count(v) for v in set(youtubetrainN)}
		
		#Memasukan Data Latih Ke array (Negatif)
		for data in TrainNegatif1['word']:
			countDataN += 1
			DataTambahanNegatif.append([data,TrainNegatif1['frekuensi'][countDataN]])

		DataValueN = []
		datafrN = []

		#Remove data frekuensi lama (pada data latih negatif)
		#Untuk digantikan dengan data frekuensi baru
		for key, value in final_youtubeN.items():
			for w,f in DataTambahanNegatif:
				if w == key:
					DataValueN.append([w,f])
					DataTambahanNegatif.remove([w,f])
					
		n = 0;
		#Adding data frekuensi baru	(Data Yang Sudah Ada)			
		for key, value in final_youtubeN.items():
			for w,f in DataValueN:
				if w == key:
					n +=1
					DataTambahanNegatif.append([w,(f+value)])
			#Jika data n = 0 artinya kata baru tidak terdapat pada data latih lama
			#Disimpan pada array baru (kata baru yang tidak ada di data lama)
			if n == 0:
				datafrN.append([key,value])
			n = 0

		#Adding data frekuensi baru	(Data Belum Ada di Data Latih Negatif)
		#Ekstraksi Fitur		
		for w,f in datafrN:
			if f > 3:
				DataTambahanNegatif.append([w,f])
					
		#Menghitung jumlah frekuensi, hasil dari penambahan data baru.
		#Disimpan dalam array dulu
		for n1, n2 in DataTambahanNegatif:
			contDataNegatif +=n2
			contDataNegatif += 1

		#Menghitung (ln probabilitas) Data Latih Negatif dan masukan ke file csv
		nomorUrut = 0
		with open ('d:/Website/dataLatih/FrekuensiNegativeTrainComment.csv','w',newline="") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(("nomor","word","frekuensi","ln"))
			for n1, n2 in DataTambahanNegatif:
				Prob = (n2 + 1)/contDataNegatif
				result = np.log(Prob)
				result2 = result2 = '{:01.3f}'.format(result)
				nomorUrut +=1

				writer.writerow([nomorUrut,n1,n2,result2])
	else:
		print("Data Negatif Tidak Ada")

	return number

def getDataYoutube(idVideo,qtyComment):
	print("QTY",qtyComment)
	api_key = 'AIzaSyBPI4Dod8Z5Snc4PKg3u2CYB7AyERaxUXY'
	youtube = build('youtube', 'v3', developerKey=api_key)
	
	#URL untuk mengambil id video
	name_regex = re.compile(r'v=(.*)')
	mo = name_regex.search(idVideo)
	id_vidio = mo.group(1)
	print(id_vidio)
	
	CountComment = 0
	CountView = 0
	CountDislike = 0
	CountLike = 0
	subs = 0
	saveDataComment = []

	
	#Tidak mengambil data secara keseluruhan
	if qtyComment != '0':
		#Endpoint ambil komentar
		req1 = youtube.commentThreads().list(part='snippet', videoId=id_vidio, textFormat='plainText',maxResults=qtyComment)
		res1 = req1.execute()

		#Menyimpan data ke csv
		with open ('d:/Website/dataUjiBaru/GetDataComment.csv','w',newline="") as csv_file:
			writer = csv.writer(csv_file)
			for item1 in res1['items']:
				comment = item1['snippet']['topLevelComment']['snippet']['textOriginal']				
				convertComment=comment.encode('unicode-escape').decode('utf-8')
				writer.writerow([convertComment,""])
				saveDataComment.append(convertComment)
	else:
		#Mengambil data secara keseluruhan
		with open ('d:/Website/dataUjiBaru/GetDataComment.csv','w',newline="") as csv_file:
			writer = csv.writer(csv_file)

			comments = []
			results = youtube.commentThreads().list(part='snippet', videoId=id_vidio, textFormat='plainText').execute()
			while results:
				for item in results['items']:
					comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
					convertComment=comment.encode('unicode-escape').decode('utf-8')
					comments.append(convertComment)
					writer.writerow([convertComment,""])
		 
				# Check if another page exists
				if 'nextPageToken' in results:
					token = results['nextPageToken']
					results = youtube.commentThreads().list(part='snippet', videoId=id_vidio, textFormat='plainText', pageToken=token).execute()
				else:
					break

	
	#Endpoint untuk mengambil data jumlah koment, view, like dan dislike
	req2 = youtube.videos().list(part='statistics', id=id_vidio)
	res2 = req2.execute()
	
	#Endpoint untuk mengambil id channel
	reqId = youtube.videos().list(part='snippet', id=id_vidio)	
	resId = reqId.execute()
	
	for item2 in res2['items']:
		CountComment = item2['statistics']['commentCount']
		CountView = item2['statistics']['viewCount']
		CountDislike = item2['statistics']['dislikeCount']
		CountLike = item2['statistics']['likeCount']


	for itemId in resId['items']:
		getIdChannel = itemId['snippet']['channelId']

	#Endpoint untuk mengambil data jumlah subscriber
	req3 = youtube.channels().list(part='statistics', id=getIdChannel)
	res3 = req3.execute()
	print("Id Channel",getIdChannel)
	for item3 in res3['items']:
		subs = item3['statistics']['subscriberCount']



	return CountComment, CountView, CountDislike, CountLike, subs, id_vidio, getIdChannel

def AddDataSample(CountView,CountDislike,CountLike,positif_total,negatif_total,subs,id_vidio,getIdChannel):
	NewDataStatus = -1
	Exist = 0
	NotExist = 0
	channelCekList = pd.read_csv("FileDataHub.csv", usecols = ["channel","video","positif","negatif","view","like","dislike","subs"])

	#Mengecek data sudah data atau tidak
	for channelCek in channelCekList['channel']:
		print("channel list",channelCek)
		print("channel search",getIdChannel)
		if channelCek != getIdChannel:
			NotExist = 1
			print("Not Exist",NotExist)
		else:
			Exist = 1
			print("Exist",Exist)
	
	if Exist != 1:
		#Menambah data sampel chi square
		newData = open('FileDataHub.csv', 'a+',newline='')

		newFileWriterP = csv.writer(newData)
		newFileWriterP.writerow([getIdChannel,id_vidio,positif_total,negatif_total,CountView,CountLike,CountDislike,subs])
		NewDataStatus = 1
	else :
		NewDataStatus = 0

	newData.close()


	return NewDataStatus

def addDataChiSquare():
	SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
	SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
	MedianView = []
	MedianLike = []
	MedianDislike = []
	MedianPositiveCom = []
	MedianNegativeCom = []
	MedianSubs = []
	KlasifikasiView = []
	KlasifikasiViewShow = []
	KlasifikasiLike = []
	KlasifikasiDislike = []
	KlasifikasiPositif = []
	KlasifikasiNegatif = []
	KlasifikasiSubs = []
	KlasifikasiSubsShow = []
	
	
	Exist = 0
	NotExist = 0
	channelCekList = pd.read_csv("FileDataHub.csv", usecols = ["channel","video","positif","negatif","view","like","dislike","subs"])


	ChiSquareData = pd.read_csv("FileDataHub.csv", usecols = ["channel","video","positif","negatif","view","like","dislike","subs"])
		
	#Menghitung Median View
	for MedView in ChiSquareData['view']:
		MedianView.append(MedView)
		
	#Menghitung Median positif
	for MedPositif in ChiSquareData['positif']:
		MedianPositiveCom.append(MedPositif)
		
	#Menghitung Median negatif
	for MedNegatif in ChiSquareData['negatif']:
		MedianNegativeCom.append(MedNegatif)
			
	#Menghitung Median like
	for MedLike in ChiSquareData['like']:
		MedianLike.append(MedLike)

	#Menghitung Median dislike
	for MedDislike in ChiSquareData['dislike']:
		MedianDislike.append(MedDislike)

	#Menghitung Median subs
	for MedSubs in ChiSquareData['subs']:
		MedianSubs.append(MedSubs)


	MedianViewResult = statistics.median(MedianView)
	MedianPositifResult = statistics.median(MedianPositiveCom)
	MedianNegatifResult = statistics.median(MedianNegativeCom)
	MedianLikeResult = statistics.median(MedianLike)
	MedianDislikeResult = statistics.median(MedianDislike)
	MedianSubsResult = statistics.median(MedianSubs)

	print("Median View", MedianViewResult)
	print("Median Positif", MedianPositifResult)
	print("Median Negatif", MedianNegatifResult)
	print("Median Like", MedianLikeResult)
	print("Median Dislike", MedianDislikeResult)
	print("Median Subs", MedianSubsResult)

	#Klasifikasi View (Bagi 2 kelas)
	for view in ChiSquareData['view']:
		if view < MedianViewResult:
			KlasifikasiView.append(1)
			KlasifikasiViewShow.append([view,"RENDAH"])
		else:
			KlasifikasiView.append(2)
			KlasifikasiViewShow.append([view,"TINGGI"])

	#Klasifikasi Like (Bagi 2 kelas)
	for like in ChiSquareData['like']:
		if like < MedianLikeResult:
			KlasifikasiLike.append(1)
		else:
			KlasifikasiLike.append(2)

	#Klasifikasi Dislike (Bagi 2 kelas)
	for dislike in ChiSquareData['dislike']:
		if dislike < MedianDislikeResult:
			KlasifikasiDislike.append(1)
		else:
			KlasifikasiDislike.append(2)

	#Klasifikasi Positif (Bagi 2 kelas)
	for positif in ChiSquareData['positif']:
		if positif < MedianPositifResult:
			KlasifikasiPositif.append(1)
		else:
			KlasifikasiPositif.append(2)

	#Klasifikasi Negatif (Bagi 2 kelas)
	for negatif in ChiSquareData['negatif']:
		if negatif < MedianNegatifResult:
			KlasifikasiNegatif.append(1)
		else:
			KlasifikasiNegatif.append(2)

	#Klasifikasi Subs (Bagi 2 kelas)
	for subs in ChiSquareData['subs']:
		if subs < MedianSubsResult:
			KlasifikasiSubs.append(1)
			KlasifikasiSubsShow.append([subs,"RENDAH"])
		else:
			KlasifikasiSubs.append(2)
			KlasifikasiSubsShow.append([subs,"TINGGI"])

			

	#View And Subs
	#Memasukan data hasil klasifikasi ke dalam csv
	subQty = 0
	with open ('d:/Website/dataChiSquare/view_subs.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("x","y"))
		for v in KlasifikasiView:
			writer.writerow([v,KlasifikasiSubs[subQty]])
			subQty += 1

	
	#Data For View in Web
	subQty = 0
	with open ('d:/Website/dataChiSquare/view_subs_view.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("view","class","subs","class"))
		for s,v in KlasifikasiViewShow:
			writer.writerow([s,v,KlasifikasiSubsShow[subQty][0],KlasifikasiSubsShow[subQty][1]])
			subQty += 1

	#Like And Subs
	#Memasukan data hasil klasifikasi ke dalam csv
	subQty = 0
	with open ('d:/Website/dataChiSquare/likes_subs.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("x","y"))
		for v in KlasifikasiLike:
			writer.writerow([v,KlasifikasiSubs[subQty]])
			subQty += 1

	#Dislike And Subs
	#Memasukan data hasil klasifikasi ke dalam csv
	subQty = 0
	with open ('d:/Website/dataChiSquare/dislike_subs.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("x","y"))
		for v in KlasifikasiDislike:
			writer.writerow([v,KlasifikasiSubs[subQty]])
			subQty += 1

	#Positif And Subs
	#Memasukan data hasil klasifikasi ke dalam csv
	subQty = 0
	with open ('d:/Website/dataChiSquare/positive_subs.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("x","y"))
		for v in KlasifikasiPositif:
			writer.writerow([v,KlasifikasiSubs[subQty]])
			subQty += 1

	#Negatif And Subs
	#Memasukan data hasil klasifikasi ke dalam csv
	subQty = 0
	with open ('d:/Website/dataChiSquare/negatif_subs.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("x","y"))
		for v in KlasifikasiNegatif:
			writer.writerow([v,KlasifikasiSubs[subQty]])
			subQty += 1

	resultChisquareViewSubs = 0
	resultChisquareLikeSubs = 0
	resultChisquareDislikeSubs = 0
	resultChisquarePositifSubs = 0
	resultChisquareNegatifSubs = 0
	atasVS = []
	bawahVS = []

	atasLS = []
	bawahLS = []

	atasDS = []
	bawahDS = []

	atasPS = []
	bawahPS = []

	atasNS = []
	bawahNS = []

	allTableVS = []
	allTableLS = []
	allTableDS = []
	allTablePS = []
	allTableNS = []


	DataChiSquareX_VS = []
	DataChiSquareY_VS = []

	DataChiSquareX_LS = []
	DataChiSquareY_LS = []

	DataChiSquareX_DS = []
	DataChiSquareY_DS = []

	DataChiSquareX_PS = []
	DataChiSquareY_PS = []

	DataChiSquareX_NS = []
	DataChiSquareY_NS = []
	noDataVS = 0
	noDataLS = 0
	noDataDS = 0
	noDataPS = 0
	noDataNS = 0

	rrVS = 0
	rtVS = 0
	trVS = 0
	ttVS = 0

	rrLS = 0
	rtLS = 0
	trLS = 0
	ttLS = 0

	rrDS = 0
	rtDS = 0
	trDS = 0
	ttDS = 0

	rrPS = 0
	rtPS = 0
	trPS = 0
	ttPS = 0

	rrNS = 0
	rtNS = 0
	trNS = 0
	ttNS = 0

	#CHI SQUARE VIEW AND SUBS
	#Memasukan ke crosstabs
	DataChiSquareVS = pd.read_csv('d:/Website/dataChiSquare/view_subs.csv', usecols = ["x","y"])
	for dataX in DataChiSquareVS["x"]:
		DataChiSquareX_VS.append(dataX)

	for dataY in DataChiSquareVS["y"]:
		DataChiSquareY_VS.append(dataY)

	for x in DataChiSquareX_VS:
			
		#tinggi (x) dan rendah (subs)
		if DataChiSquareY_VS[noDataVS] == 1 and x == 2:
			trVS += 1
				
		#tinggi (x) dan tinggi (subs)
		elif DataChiSquareY_VS[noDataVS] == 2 and x == 2:
			ttVS += 1

		#rendah (x) dan rendah (subs)
		elif DataChiSquareY_VS[noDataVS] == 1 and x == 1:
			rrVS += 1
				
		#rendah (x) dan tinggi (subs)
		elif DataChiSquareY_VS[noDataVS] == 2 and x == 1:
			rtVS += 1

		else :
			print("NULL")
		noDataVS +=1
			
	Ashow = rrVS
	Bshow = rtVS
	Cshow = trVS
	Dshow = ttVS

	atasVS.append(rrVS)
	atasVS.append(rtVS)
	bawahVS.append(trVS)
	bawahVS.append(ttVS)
	allTableVS.append([atasVS,bawahVS])
	print(allTableVS)

		
	#perhitungan chi2 dengan library
	statVS1, pVS, dofVS, expectedVS = chi2_contingency(allTableVS)
	statVS ='{:02.3f}'.format(statVS1)
	print("statVS ",statVS)
	# interpret test-statistic
	probVS = 0.95
	criticalVS = chi2.ppf(probVS, dofVS)
		
	alphaVS = 1.0 - probVS
			
	if pVS <= alphaVS:
		resultChisquareViewSubs = 1
		messageV = "Accept H1".translate(SUB)

	else:
		resultChisquareViewSubs = 0
		messageV = "Accept H0".translate(SUB)



	print("VIEW AND SUBS DONE")

	#CHI SQUARE LIKE AND SUBS
	#Memasukan ke crosstabs
	DataChiSquareLS = pd.read_csv('d:/Website/dataChiSquare/likes_subs.csv', usecols = ["x","y"])
	for dataX in DataChiSquareLS["x"]:
		DataChiSquareX_LS.append(dataX)

	for dataY in DataChiSquareLS["y"]:
		DataChiSquareY_LS.append(dataY)

	for x in DataChiSquareX_LS:
			
		#tinggi (x) dan rendah (subs)
		if DataChiSquareY_LS[noDataLS] == 1 and x == 2:
			trLS += 1
				
		#tinggi (x) dan tinggi (subs)
		elif DataChiSquareY_LS[noDataLS] == 2 and x == 2:
			ttLS += 1

		#rendah (x) dan rendah (subs)
		elif DataChiSquareY_LS[noDataLS] == 1 and x == 1:
			rrLS += 1
				
		#rendah (x) dan tinggi (subs)
		elif DataChiSquareY_LS[noDataLS] == 2 and x == 1:
			rtLS += 1

		else :
			print("NULL")
		noDataLS +=1
			

	atasLS.append(rrLS)
	atasLS.append(rtLS)
	bawahLS.append(trLS)
	bawahLS.append(ttLS)
	allTableLS.append([atasLS,bawahLS])
		
		
	#perhitungan chi2 dengan library		
	statLS1, pLS, dofLS, expectedLS = chi2_contingency(allTableLS)
	statLS ='{:02.3f}'.format(statLS1)
	# interpret test-statistic
	probLS = 0.95
	criticalLS = chi2.ppf(probLS, dofLS)
		
	alphaLS = 1.0 - probLS
			
	if pLS <= alphaLS:
		resultChisquareLikeSubs = 1
		messageL = "Accept H1".translate(SUB)

	else:
		resultChisquareLikeSubs = 0
		messageL = "Accept H0".translate(SUB)

	print("LIKE AND SUBS DONE")

	#CHI SQUARE DISLIKE AND SUBS
	#Memasukan ke crosstabs
	DataChiSquareDS = pd.read_csv('d:/Website/dataChiSquare/dislike_subs.csv', usecols = ["x","y"])
	for dataX in DataChiSquareDS["x"]:
		DataChiSquareX_DS.append(dataX)

	for dataY in DataChiSquareDS["y"]:
		DataChiSquareY_DS.append(dataY)

	for x in DataChiSquareX_DS:
		
		#tinggi (x) dan rendah (subs)
		if DataChiSquareY_DS[noDataDS] == 1 and x == 2:
			trDS += 1
				
		#tinggi (x) dan tinggi (subs)
		elif DataChiSquareY_DS[noDataDS] == 2 and x == 2:
			ttDS += 1

		#rendah (x) dan rendah (subs)
		elif DataChiSquareY_DS[noDataDS] == 1 and x == 1:
			rrDS += 1
				
		#rendah (x) dan tinggi (subs)
		elif DataChiSquareY_DS[noDataDS] == 2 and x == 1:
			rtDS += 1

		else :
			print("NULL")
		noDataDS +=1
			

	atasDS.append(rrDS)
	atasDS.append(rtDS)
	bawahDS.append(trDS)
	bawahDS.append(ttDS)
	allTableDS.append([atasDS,bawahDS])
		
	#perhitungan chi2 dengan library		
	statDS1, pDS, dofDS, expectedDS = chi2_contingency(allTableDS)
	statDS ='{:02.3f}'.format(statDS1)
	# interpret test-statistic
	probDS = 0.95
	criticalDS = chi2.ppf(probDS, dofDS)
		
	alphaDS = 1.0 - probDS
			
	if pDS <= alphaDS:
		resultChisquareDislikeSubs = 1
		messageD = "Accept H1".translate(SUB)

	else:
		resultChisquareDislikeSubs = 0
		messageD = "Accept H0".translate(SUB)

	print("DISLIKE AND SUBS DONE")

	#CHI SQUARE POSITIF AND SUBS
	#Memasukan ke crosstabs
	DataChiSquarePS = pd.read_csv('d:/Website/dataChiSquare/positive_subs.csv', usecols = ["x","y"])
	for dataX in DataChiSquarePS["x"]:
		DataChiSquareX_PS.append(dataX)

	for dataY in DataChiSquarePS["y"]:
		DataChiSquareY_PS.append(dataY)

	for x in DataChiSquareX_PS:
			
		#tinggi (x) dan rendah (subs)
		if DataChiSquareY_PS[noDataPS] == 1 and x == 2:
			trPS += 1
				
		#tinggi (x) dan tinggi (subs)
		elif DataChiSquareY_PS[noDataPS] == 2 and x == 2:
			ttPS += 1

		#rendah (x) dan rendah (subs)
		elif DataChiSquareY_PS[noDataPS] == 1 and x == 1:
			rrPS += 1
				
		#rendah (x) dan tinggi (subs)
		elif DataChiSquareY_PS[noDataPS] == 2 and x == 1:
			rtPS += 1

		else :
			print("NULL")
		noDataPS +=1
			

	atasPS.append(rrPS)
	atasPS.append(rtPS)
	bawahPS.append(trPS)
	bawahPS.append(ttPS)
	allTablePS.append([atasPS,bawahPS])
		
	#perhitungan chi2 dengan library
	statPS1, pPS, dofPS, expectedPS = chi2_contingency(allTablePS)
	statPS ='{:02.3f}'.format(statPS1)
	# interpret test-statistic
	probPS = 0.95
	criticalPS = chi2.ppf(probPS, dofPS)
		
	alphaPS = 1.0 - probPS
			
	if pPS <= alphaPS:
		resultChisquarePositifSubs = 1
		messageP = "Accept H1".translate(SUB)
		print("pPS",'{:02.3f}'.format(pPS))

	else:
		resultChisquarePositifSubs = 0
		messageP = "Accept H0".translate(SUB)

	print("POSITIF AND SUBS DONE")

	#CHI SQUARE NEGATIF AND SUBS
	#Memasukan ke crosstabs
	DataChiSquareNS = pd.read_csv('d:/Website/dataChiSquare/negatif_subs.csv', usecols = ["x","y"])
	for dataX in DataChiSquareNS["x"]:
		DataChiSquareX_NS.append(dataX)

	for dataY in DataChiSquareNS["y"]:
		DataChiSquareY_NS.append(dataY)

	for x in DataChiSquareX_NS:
			
		#tinggi (x) dan rendah (subs)
		if DataChiSquareY_NS[noDataNS] == 1 and x == 2:
			trNS += 1
				
		#tinggi (x) dan tinggi (subs)
		elif DataChiSquareY_NS[noDataNS] == 2 and x == 2:
			ttNS += 1

		#rendah (x) dan rendah (subs)
		elif DataChiSquareY_NS[noDataNS] == 1 and x == 1:
			rrNS += 1
				
		#rendah (x) dan tinggi (subs)
		elif DataChiSquareY_NS[noDataNS] == 2 and x == 1:
			rtNS += 1

		else :
			print("NULL")
		noDataNS +=1
			

	atasNS.append(rrNS)
	atasNS.append(rtNS)
	bawahNS.append(trNS)
	bawahNS.append(ttNS)
	allTableNS.append([atasNS,bawahNS])
		
	#perhitungan chi2 dengan library
	statNS1, pNS, dofNS, expectedNS = chi2_contingency(allTableNS)
	statNS ='{:02.3f}'.format(statNS1)
	# interpret test-statistic
	probNS = 0.95
	criticalNS = chi2.ppf(probNS, dofNS)
		
	alphaNS = 1.0 - probNS
		
	if pNS <= alphaNS:
		resultChisquareNegatifSubs = 1
		messageN = "Accept H1".translate(SUB)
		print("pNS",pNS)

	else:
		resultChisquareNegatifSubs = 0
		messageN = "Accept H0".translate(SUB)
		print("pNS",'{:02.3f}'.format(pNS))

	print("NEGATIF AND SUBS DONE")
	print("SAMPLE SUCCESSFUL TO ADD")

	#Memasukan data chi2 ke csv
	with open ('d:/Website/dataChiSquare/resultChiSquare.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("view_subs","like_subs","dislike_subs","positif_subs","negatif_subs"))
		writer.writerow(['{:02.3f}'.format(statVS1),'{:02.3f}'.format(statLS1),'{:02.3f}'.format(statDS1),'{:02.3f}'.format(statPS1),'{:02.3f}'.format(statNS1)])
	
	
	return messageV,messageL,messageD,messageP,messageN,resultChisquareViewSubs,resultChisquareLikeSubs,resultChisquareDislikeSubs,resultChisquarePositifSubs,resultChisquareNegatifSubs,statVS,statLS,statDS,statPS,statNS,pVS,pLS,pDS,pPS,pNS,Ashow,Bshow,Cshow,Dshow

def UjiNormalitas():
	channelCekList = pd.read_csv("FileDataHub.csv", usecols = ["positif","negatif","view","like","dislike","subs"])

	code = 0

	#Uji Normalitas variabel positve
	statP, pP = shapiro(channelCekList['positif'])
	print('Statistics Positif =%.3f, p Positif =%.3f' % (statP, pP))
	convertP ='{:02.3f}'.format(pP)
	
	#Uji Normalitas variabel negatif
	statN, pN = shapiro(channelCekList['negatif'])
	print('Statistics Negatif =%.3f, p Negatif =%.3f' % (statN, pN))
	convertN ='{:02.3f}'.format(pN)


	#Uji Normalitas variabel view
	statV, pV = shapiro(channelCekList['view'])
	print('Statistics View =%.3f, p View =%.3f' % (statV, pV))
	convertV ='{:02.3f}'.format(pV)

	#Uji Normalitas variabel like
	statL, pL = shapiro(channelCekList['like'])
	print('Statistics Like =%.3f, p Like =%.3f' % (statL, pL))
	convertL ='{:02.3f}'.format(pL)
	

	#Uji Normalitas variabel dislike
	statD, pD = shapiro(channelCekList['dislike'])
	print('Statistics Dislike =%.3f, p Dislike =%.3f' % (statD, pD))
	convertD ='{:02.3f}'.format(pD)
	

	#Uji Normalitas variabel subscriber
	statS, pS = shapiro(channelCekList['subs'])
	print('Statistics Subs =%.3f, p Subs =%.3f' % (statS, pS))
	convertS ='{:02.3f}'.format(pS)

	
	if pP == 1 and pS == 1 and pN == 1 and pV == 1 and pL == 1 and pD == 1:
		code = 1
	else:
		code = 0


	print("code",code)

	with open ('d:/Website/dataNormalitas/resultUjiNormalitas.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("ujiP","ujiN","ujiV","ujiL","ujiD","ujiS"))
		writer.writerow([convertP,convertN,convertV,convertL,convertD,convertS])



	return convertP,convertN,convertV,convertL,convertD,convertS,code

def addDataRegresiLinear():
	channelCekList = pd.read_csv("FileDataHub.csv", usecols = ["positif","negatif","view","like","dislike","subs"])

	XV = channelCekList['view']
	XL = channelCekList['like']
	XD = channelCekList['dislike']
	XP = channelCekList['positif']
	XN = channelCekList['negatif']
	Y = channelCekList['subs']
	

	XV = sm.add_constant(XV) # adding a constant
	XL = sm.add_constant(XL) # adding a constant
	XD = sm.add_constant(XD) # adding a constant
	XP = sm.add_constant(XP) # adding a constant
	XN = sm.add_constant(XN) # adding a constant
	

	modelVS = sm.OLS(Y,XV)
	modelLS = sm.OLS(Y,XL)
	modelDS = sm.OLS(Y,XD)
	modelPS = sm.OLS(Y,XP)
	modelNS = sm.OLS(Y,XN)
	resultsVS = modelVS.fit()
	resultsLS = modelLS.fit()
	resultsDS = modelDS.fit()
	resultsPS = modelPS.fit()
	resultsNS = modelNS.fit()
	

	nVS = resultsVS.pvalues
	nLS = resultsLS.pvalues
	nDS = resultsDS.pvalues
	nPS = resultsPS.pvalues
	nNS = resultsNS.pvalues
	

	cVS ='{:02.3f}'.format(nVS[1])
	cLS ='{:02.3f}'.format(nLS[1])
	cDS ='{:02.3f}'.format(nDS[1])
	cPS ='{:02.3f}'.format(nPS[1])
	cNS ='{:02.3f}'.format(nNS[1])

	with open ('d:/Website/dataRegresi/resultsRegresiLinear.csv','w',newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(("ujiP_S","ujiN_S","ujiV_S","ujiL_S","ujiD_S"))
		writer.writerow([cVS,cLS,cDS,cPS,cNS])

	return cVS,cLS,cDS,cPS,cNS




@app.route('/', methods=['GET', 'POST'])	
def index():


	logger = logging.getLogger(__name__)
	#View Data in web
	dataset = tablib.Dataset()
	with open(os.path.join(os.path.dirname(__file__),'FileDataHub.csv')) as f:
		dataset.csv = f.read()

	dataset1 = tablib.Dataset()
	with open(os.path.join(os.path.dirname(__file__),'d:/Website/dataLatih/FrekuensiPositiveTrainComment.csv')) as f:
		dataset1.csv = f.read()

	dataset2 = tablib.Dataset()
	with open(os.path.join(os.path.dirname(__file__),'d:/Website/dataLatih/FrekuensiNegativeTrainComment.csv')) as f:
		dataset2.csv = f.read()

	dataset3 = tablib.Dataset()
	with open(os.path.join(os.path.dirname(__file__),'d:/Website/dataChiSquare/view_subs_view.csv')) as f:
		dataset3.csv = f.read()

	data = dataset.html
	data1 = dataset1.html
	data2 = dataset2.html
	data3 = dataset3.html
	messageV = ""
	messageL = ""
	messageD = ""
	messageP = ""
	messageN = ""
	codeP = 0
	codeV = 0
	codeL = 0
	codeD = 0
	codeN = 0


	chi2Result = pd.read_csv("d:/Website/dataChiSquare/resultChiSquare.csv", usecols = ["view_subs","like_subs","dislike_subs","positif_subs","negatif_subs"])
	ujiNormalResult = pd.read_csv("D:/Website/dataNormalitas/resultUjiNormalitas.csv", usecols = ["ujiP","ujiN","ujiV","ujiL","ujiD","ujiS"])
	sVS ='{:02.3f}'.format(chi2Result['view_subs'][0])
	sLS ='{:02.3f}'.format(chi2Result['like_subs'][0])
	sDS ='{:02.3f}'.format(chi2Result['dislike_subs'][0])
	sNS ='{:02.3f}'.format(chi2Result['negatif_subs'][0])
	sPS ='{:02.3f}'.format(chi2Result['positif_subs'][0])


	if request.method == 'POST' :
		DataPositif = open("d:/Website/dataLatih/DataLatihPositiveTest.csv","r+")
		reader_file_positif = csv.reader(DataPositif)
		QtyPositif = len(list(reader_file_positif))
		
		DataNegatif = open("d:/Website/dataLatih/DataLatihNegativeTest.csv","r+")
		reader_file_negatif = csv.reader(DataNegatif)
		QtyNegatif = len(list(reader_file_negatif))

		text_file = open("stop.txt", "r")
		lines = text_file.readlines()
		QtyStop = (len(lines))

		filename = request.form.get('csvfile')
		idVideo = request.form.get('idVideo')
		qtyComment = request.form.get('qtyComment')
		qtyCommentAll = request.form.getlist("qtyCommentAll")
		AddSample = request.form.getlist("AddSample")
		fileIs = "d:/Website/dataUjiBaru/GetDataComment.csv"
		
		# Get All Comment if checkbox active.
		print('qtyCommentAll',qtyCommentAll)
		if not qtyCommentAll:
			qtyComment = qtyComment
			print("q",qtyComment)
		else:
			qtyComment = '0'
			print("q",qtyComment)
			

		#Naive Bayes And Chi Square Processing
		if idVideo != None:
			try:

				print("GET DATA YOUTUBE")
				startGetData = time.time()
				CountComment, CountView, CountDislike, CountLike, subs, id_vidio, getIdChannel = getDataYoutube(idVideo,qtyComment)
				endGetData = time.time()
				ComputationGetData = endGetData - startGetData
				print("GET DATA TIME",ComputationGetData)

				if subs != 0:
				
					try:
						print("PRE PROCESSING")
						startPrePro = time.time()
						preprocessing_result = preprocessing(fileIs)
						endPrePro = time.time()
						ComputationPrePro = endPrePro - startPrePro
						print("PREPROCESSING TIME",ComputationPrePro)

					except Exception as e:
						logger.error('First Failed to upload to ftp: '+ str(e))
						message = "Failed to Preprocessing Text"
						return render_template('index.html',message=message,QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),QtyStop=QtyStop)


					try:
						print("NBAYES")
						startNb = time.time()
						classification_result, positif_total, negatif_total = naive_bayes(preprocessing_result)
						endNb = time.time()
						ComputationNb = endNb - startNb
						print("NAIVE BAYES TIME",ComputationNb)

					except Exception as e:
						logger.error('second Failed to upload to ftp: '+ str(e))
						message = "Failed to Naive Bayes Proccessing"
						return render_template('index.html',message=message,QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),QtyStop=QtyStop)

					
					try:
						if not AddSample:
							print("Not Add to Data Sampel Chi Square")
							return render_template('index.html',classification_result=enumerate(classification_result),positif=positif_total,negatif=negatif_total,QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),QtyStop=QtyStop,CountComment=CountComment, CountView=CountView, CountDislike=CountDislike, CountLike=CountLike, subs=subs,data1=data1,data2=data2)
			
						else:
							NewDataStatus = AddDataSample(CountView,CountDislike,CountLike,positif_total,negatif_total,subs,id_vidio,getIdChannel)

							
							if NewDataStatus == 1:
								convertP,convertN,convertV,convertL,convertD,convertS,code = UjiNormalitas()
								
								
								if code == 0:
									print("codeee ",code)
									message = "Data tidak berdistribusi normal"
									messageV,messageL,messageD,messageP,messageN,resultChisquareViewSubs,resultChisquareLikeSubs,resultChisquareDislikeSubs,resultChisquarePositifSubs,resultChisquareNegatifSubs,statVS,statLS,statDS,statPS,statNS,pVS,pLS,pDS,pPS,pNS,Ashow,Bshow,Cshow,Dshow= addDataChiSquare()
									
									print("Akhir")

									
									return render_template('index.html',classification_result=enumerate(classification_result),positif=positif_total,negatif=negatif_total,QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),	QtyStop=QtyStop,CountComment=CountComment, CountView=CountView, CountDislike=CountDislike, CountLike=CountLike, subs=subs, sVS=statVS, sLS=statLS, sDS=statDS, sPS=statPS, sNS=statNS, pVS=pVS, pLS=pLS, pDS=pDS, pPS=pPS, pNS=pNS,data=data,data1=data1,data2=data2,messageV=messageV,messageL=messageL,messageD=messageD,messageP=messageP,messageN=messageN,message=message,convertP=convertP,convertN=convertN,convertV=convertV,convertL=convertL,convertD=convertD,convertS=convertS,code=code,data3=data3)
								else:
									cVS,cLS,cDS,cPS,cNS = addDataRegresiLinear()
									return render_template('index.html',classification_result=enumerate(classification_result),positif=positif_total,negatif=negatif_total,QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),QtyStop=QtyStop,CountComment=CountComment, CountView=CountView, CountDislike=CountDislike, CountLike=CountLike, subs=subs,cVS=cVS,cLS=cLS,cDS=cDS,cPS=cPS,cNS=cNS,convertP=convertP,convertN=convertN,convertV=convertV,convertL=convertL,convertD=convertD,convertS=convertS,data1=data1,data2=data2)
									
			
							else:
								print("DATA TIDAK MASUK")
								message = "Data Sampel Exist in Database1"
								return render_template('index.html',classification_result=enumerate(classification_result),positif=positif_total,negatif=negatif_total,QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),QtyStop=QtyStop,CountComment=CountComment, CountView=CountView, CountDislike=CountDislike, CountLike=CountLike, subs=subs,convertP=ujiNormalResult['ujiP'][0],convertN=ujiNormalResult['ujiN'][0],convertV=ujiNormalResult['ujiV'][0],convertL=ujiNormalResult['ujiL'][0],convertD=ujiNormalResult['ujiD'][0],convertS=ujiNormalResult['ujiS'][0],data1=data1,data2=data2)
							
					except Exception as e:
						logger.error('third Failed to upload to ftp: '+ str(e))
						message = "Data Sampel Exist in Database2"
						return render_template('index.html',message=message,QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),QtyStop=QtyStop)

				else:
					message = "Your Id Channel no valid"
					return render_template('index.html',message=message,QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),QtyStop=QtyStop)

			except Exception as e:
				logger.error('fourth Failed to upload to ftp: '+ str(e))
				message = "Your Id Video no valid"
				return render_template('index.html',message=message,QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),QtyStop=QtyStop)


		#Validate Processing
		else:
			print("VALIDATE")
			selected_users = request.form.getlist("validasiData")
			startValid = time.time()
			training(selected_users)
			endValid = time.time()
			ComputationValid = endValid - startValid
			print("VALIDASI TIME",ComputationValid)

			return render_template('index.html',QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),QtyStop=QtyStop)
	
	else :
		DataPositif = open("d:/Website/dataLatih/DataLatihPositiveTest.csv","r+")
		reader_file_positif = csv.reader(DataPositif)
		QtyPositif = len(list(reader_file_positif))
		
		DataNegatif = open("d:/Website/dataLatih/DataLatihNegativeTest.csv","r+")
		reader_file_negatif = csv.reader(DataNegatif)
		QtyNegatif = len(list(reader_file_negatif))

		text_file = open("stop.txt", "r")
		lines = text_file.readlines()
		QtyStop = (len(lines))


		return render_template('index.html',QtyPositif=(QtyPositif),QtyNegatif=(QtyNegatif),QtyStop=QtyStop)

	

if __name__ == '__main__':
	app.run(debug=True)
