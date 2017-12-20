import boto3
import pandas as pd

s3 = boto3.resource('s3')
s3bucket = s3.Bucket('dataxteamprojectfacebookphotos')
bucket='dataxteamprojectfacebookphotos'
client=boto3.client('rekognition')



MBTI = list(s3bucket.objects.filter(Prefix='Facebook/'+str('ENFP')))

MBTIlist = list()
for i in MBTI:
    MBTIlist.append(i.key)
          
MBTIphotos = pd.DataFrame(MBTIlist,columns=['fileName'])          
          


Labels = []
faceDetails = []

for i in range(1000):
    response = client.detect_labels(Image={'S3Object':{'Bucket':bucket,'Name':MBTIphotos.fileName[i]}},MinConfidence=50,MaxLabels=50)
    Labels.append(response['Labels'])
    response1 = client.detect_faces(Image={'S3Object':{'Bucket':bucket,'Name':MBTIphotos.fileName[i]}},Attributes=['ALL'])
    faceDetails.append(response1['FaceDetails'])
    print(str(i) + " photos complete") 

xPhotos = MBTIlist[0:1000]

MBTItags = pd.DataFrame(xPhotos,columns=['fileName'])
MBTItags['Labels'] = Labels
MBTItags['FaceDetails'] = faceDetails 

MBTItags.to_csv('ENFP-1000.csv')     