import boto3
import pandas as pd

s3 = boto3.resource('s3')

s3bucket = s3.Bucket('dataxteamprojectfacebookphotos')

ENFJ = list(s3bucket.objects.filter(Prefix='Facebook/ENFJ'))
#obj = ENFJ[0].key

ENFJlist = list()
for i in ENFJ:
    ENFJlist.append(i.key)
          
ENFJphotos = pd.DataFrame(ENFJlist,columns=['fileName'])          
          
bucket='dataxteamprojectfacebookphotos'

client=boto3.client('rekognition')

Labels = []

for i in range(1000):
    response = client.detect_labels(Image={'S3Object':{'Bucket':bucket,'Name':ENFJphotos.fileName[i]}},MinConfidence=50,MaxLabels=50)
    Labels.append(response['Labels'])
    if i%50 == 0:
        print(str(i) + " photos complete") 

xPhotos = ENFJlist[0:1000]

ENFJtags = pd.DataFrame(xPhotos,columns=['fileName'])
ENFJtags['Labels'] = Labels


faceDetails = []
for i in range(50):
    response1 = client.detect_faces(Image={'S3Object':{'Bucket':bucket,'Name':ENFJphotos.fileName[i]}},Attributes=['ALL'])
    faceDetails.append(response1['FaceDetails'])

ENFJtags['FaceDetails'] = faceDetails 

ENFJtags.to_csv('50ENFJphotos.csv')     
     
     

#photolist = list()
#for i in s3bucket.objects.all():
#    photolist.append(i.key)

#pilot.loc[0,'Labels'][0]['Name']
#list(pilot.loc[0,'FaceDetails'][0].keys())

#temp = []
#for i in pilot.loc[:,'Labels']:
#    for z in i:
#        temp.append(z['Name'])
    


