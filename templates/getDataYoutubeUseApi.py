from googleapiclient.discovery import build
import csv
import sys
api_key = 'AIzaSyBPI4Dod8Z5Snc4PKg3u2CYB7AyERaxUXY'
youtube = build('youtube', 'v3', developerKey=api_key)


req2 = youtube.videos().list(part='statistics', id='WCooNX8-jQo')
req3 = youtube.channels().list(part='statistics', id='UCDCBzQLR7UrHxDUU1pHHorw')
res2 = req2.execute()
res3 = req3.execute()
w = 0
subs = 0
save = []


for item2 in res2['items']:
    w = item2['statistics']['commentCount']
    x = item2['statistics']['viewCount']
    y = item2['statistics']['dislikeCount']
    z = item2['statistics']['likeCount']
    print('Comment count : ',w)
    print('View count : ',x)
    print('Dislike count : ',y)
    print('Like count : ',z)

for item3 in res3['items']:
    subs = item3['statistics']['subscriberCount']
    print('Subscribers count : ',subs)

req1 = youtube.commentThreads().list(part='snippet', videoId='WCooNX8-jQo', textFormat='plainText', maxResults=90)
res1 = req1.execute()

with open ('dataText.csv','w',newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(("text",""))
    for item1 in res1['items']:
        comment = item1['snippet']['topLevelComment']['snippet']['textOriginal']
        convertComment=comment.encode('unicode-escape').decode('utf-8')
        #print(convertComment)
        writer.writerow([convertComment,""])
