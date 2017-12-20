#Final version
import boto3
import pandas as pd
from botocore.exceptions import ClientError

s3 = boto3.resource('s3')
s3bucket = s3.Bucket('dataxteamprojectfacebookphotos')
bucket='dataxteamprojectfacebookphotos'
client=boto3.client('rekognition')


def callRekognition(MBTItype):
    MBTI = list(s3bucket.objects.filter(Prefix='Facebook/'+str(MBTItype)))
    
    MBTIlist = list()
   
    for i in MBTI:
        MBTIlist.append(i.key)
    
    MBTIphotos = pd.DataFrame(MBTIlist,columns=['fileName'])
 
    imageNames = []
    Labels = []
    faceDetails = []
    
    for i in range(0,1000):
        try:
            response = client.detect_labels(Image={'S3Object':{'Bucket':bucket,'Name':MBTIphotos.fileName[i]}},MinConfidence=50,MaxLabels=50)
            response1 = client.detect_faces(Image={'S3Object':{'Bucket':bucket,'Name':MBTIphotos.fileName[i]}},Attributes=['ALL'])
            Labels.append(response['Labels'])
            faceDetails.append(response1['FaceDetails'])
            imageNames.append(MBTIlist[i])
            print(str(i) + " photos complete")
        except (ClientError,ValueError):
            continue
    
    MBTItags = pd.DataFrame(imageNames,columns=['fileName'])
    MBTItags['Labels'] = Labels
    MBTItags['FaceDetails'] = faceDetails
    MBTItags.to_csv(str(MBTItype)+'-1000.csv')
    

#callRekognition('ENFP')
#callRekognition('ENTJ')
#callRekognition('ENTP')
#callRekognition('ESFJ')
#callRekognition('INFJ')
#callRekognition('INFP')
#callRekognition('INTJ')

callRekognition('ESFP')
callRekognition('ESTJ')
callRekognition('ESTP')
callRekognition('INTP')
callRekognition('ISFJ')
callRekognition('ISFP')
callRekognition('ISTJ')

#ENFP 979 is error
