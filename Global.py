import moviepy.editor as mp
import cv2
from evaluation import prediction
from audioevaluation import predictionaudio

########################################################################################################################
#Extract Audio

my_clip = mp.VideoFileClip(r"/Users/quentinaudy/PycharmProjects/ferproject/Full_process/test.mp4")
my_clip.audio.write_audiofile(r"/Users/quentinaudy/PycharmProjects/ferproject/Full_process/test_audio.wav")

#PATHS

global_path = "/Users/quentinaudy/PycharmProjects/ferproject/Full_process"
clip_path = "/Users/quentinaudy/PycharmProjects/ferproject/Full_process/test.mp4"

########################################################################################################################
#Capture the face in the video

vidcap = cv2.VideoCapture(clip_path)
middle_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
success, image = vidcap.read()
count = 0
success = True
while success:
    success, image = vidcap.read()
    if count == middle_frame:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("/Users/quentinaudy/PycharmProjects/ferproject//Full_process/test_image.jpg", gray)
    count += 1

image_path = global_path + "/test_image.jpg"
audio_path = global_path + "/test_audio.wav"

#Predict the emotion in the image

image_prediction = prediction(image_path)
print(image_prediction)

########################################################################################################################
#Audio Recognition

audio_prediction, probability = predictionaudio(audio_path)
#audio_prediction = "fearful"
print(audio_prediction)

########################################################################################################################
#Score Fusion

if image_prediction == audio_prediction:
    global_prediction = image_prediction
elif audio_prediction == "angry" and image_prediction != "angry":
    global_prediction = "angry"
elif audio_prediction == "fear" and image_prediction != "fear":
    global_prediction = "fear"
elif audio_prediction == "surprised" and image_prediction != "surprised":
    global_prediction = "surprised"
elif audio_prediction == "disgust" and image_prediction != "disgust":
    global_prediction = "disgust"
elif audio_prediction == "sad" and image_prediction != "sad":
    global_prediction = "sad"
elif audio_prediction == "happy" and image_prediction != "happy":
    global_prediction = image_prediction
elif audio_prediction == "neutral" and image_prediction != "neutral":
    global_prediction = image_prediction


print("The emotion is " + global_prediction)


