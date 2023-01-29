import moviepy.editor as mp
import cv2
from evaluation import prediction
from audioevaluation import predictionaudio
from mtcnn import MTCNN
import os
import time
import numpy as np

########################################################################################################################
#Extract Audio

my_clip = mp.VideoFileClip(r"/Users/quentinaudy/PycharmProjects/ferproject/Full_process/test_helene.mov")
my_clip.audio.write_audiofile(r"/Users/quentinaudy/PycharmProjects/ferproject/Full_process/test_audio.wav")

#PATHS

global_path = "/Users/quentinaudy/PycharmProjects/ferproject/Full_process"
clip_path = "/Users/quentinaudy/PycharmProjects/ferproject/Full_process/test_helene.mov"

########################################################################################################################

#Capture the face in the video

vidcap = cv2.VideoCapture(clip_path)
success, image = vidcap.read()
count = 0

detector = MTCNN()

while success:
    if count%7 == 0 and count !=0 and count !=7 and count != 14 and count!= 21 :
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detectedFace = detector.detect_faces(image)[0]["box"]
        print(detectedFace)
        #image = cv2.rectangle(image,(detectedFace[0],detectedFace[1]),(detectedFace[0]+detectedFace[2],detectedFace[1]+detectedFace[3]),(0,255,0),10)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        center = [(2*detectedFace[0]+detectedFace[2])/2,(2*detectedFace[1]+detectedFace[3])/2]
        maxwidth = max(detectedFace[1],detectedFace[3])


        cropped_image = gray[int(center[1]-maxwidth/2):int(center[1]+maxwidth/2),int(center[0]-maxwidth/2):int(center[0]+maxwidth/2)]

        resized_image = cv2.resize(cropped_image,(48,48),interpolation = cv2.INTER_AREA)

        cv2.imwrite("/Users/quentinaudy/PycharmProjects/ferproject//Full_process/MTCNN/test_image{}.jpg".format(count), resized_image)

        print('Read a new frame: ', success)
    count += 1
    success, image = vidcap.read()

########################################################################################################################
# vidcap = cv2.VideoCapture(clip_path)
# middle_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
# success, image = vidcap.read()
# count = 0
# success = True
# while success:
#     success, image = vidcap.read()
#     if count == middle_frame:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite("/Users/quentinaudy/PycharmProjects/ferproject//Full_process/test_image.jpg", gray)
#     count += 1

########################################################################################################################
image_path = global_path + "/MTCNN"
audio_path = global_path + "/test_audio.wav"

full_list = np.array([0., 0., 0., 0., 0., 0., 0.])
count = 0

time.sleep(3)
for image in os.listdir(image_path):
    if image != ".DS_Store":
        #Predict the emotion in the image
        print(image_path+"/"+image)
        frame_prediction = prediction(image_path+"/"+image)
        #print(image_prediction)
        full_list += frame_prediction
        count += 1

full_list = full_list/count

#probability_image = max(full_list)
#Emotion_idx = full_list.index(probability_image)

Emotion_idx = np.argmax(full_list)
probability_image = full_list[Emotion_idx]

Emotion_list = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

image_prediction = Emotion_list[Emotion_idx]
print("Prédiction pour l'image: " + image_prediction)
print("Probabilité de la prédiction pour l'image: " + str(probability_image*100))

########################################################################################################################
#Audio Recognition

audio_prediction, probability_audio = predictionaudio(audio_path)
#audio_prediction = "fearful"
print("Prédiction pour l'audio: " + audio_prediction)
print("Probabilité de la prédiciton pour l'audio: " + str(probability_audio))

########################################################################################################################
#Score Fusion

#if image_prediction == audio_prediction:
#    global_prediction = image_prediction
#elif audio_prediction == "angry" and image_prediction != "angry":
#    global_prediction = "angry"
#elif audio_prediction == "fear" and image_prediction != "fear":
#    global_prediction = "fear"
#elif audio_prediction == "surprised" and image_prediction != "surprised":
#    global_prediction = "surprised"
#elif audio_prediction == "disgust" and image_prediction != "disgust":
#    global_prediction = "disgust"
#elif audio_prediction == "sad" and image_prediction != "sad":
#    global_prediction = "sad"
#elif audio_prediction == "happy" and image_prediction != "happy":
#    global_prediction = image_prediction
#elif audio_prediction == "neutral" and image_prediction != "neutral":
#    global_prediction = image_prediction

if image_prediction == audio_prediction:
    global_prediction = image_prediction
elif probability_audio>(probability_image*100):
    global_prediction = audio_prediction
elif probability_audio<(probability_image*100):
    global_prediction = image_prediction

print("The emotion is " + global_prediction)


