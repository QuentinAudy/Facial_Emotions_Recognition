from mtcnn import MTCNN
import cv2
import os
import shutil


img = cv2.imread('test.jpg')
detector = MTCNN()

detectedFace = detector.detect_faces(img)[0]
print(detectedFace)




def saveVideoFramesCropped(videoFilePath):

    print(videoFilePath)

    basename = os.path.basename(videoFilePath)[0:-4]
    stringIndexes = basename.split('-')
    indexes = list(map(int,stringIndexes))

    listOfEmotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']

    pathFolderWhereToSaveFrames = "VideoFrames/"+listOfEmotions[indexes[2]-1]

    vidcap = cv2.VideoCapture(videoFilePath)
    success, image = vidcap.read()
    count = 0

    detector = MTCNN()

    while success:
        if count%15 == 0 and count !=0 and count !=15 :
            if not os.path.exists(pathFolderWhereToSaveFrames + '/' + basename + "f{}.jpg".format(count)) :
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                detectedFace = detector.detect_faces(image)[0]["box"]
                print(detectedFace)
                #image = cv2.rectangle(image,(detectedFace[0],detectedFace[1]),(detectedFace[0]+detectedFace[2],detectedFace[1]+detectedFace[3]),(0,255,0),10)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                center = [(2*detectedFace[0]+detectedFace[2])/2,(2*detectedFace[1]+detectedFace[3])/2]
                maxwidth = max(detectedFace[1],detectedFace[3])


                cropped_image = gray[int(center[1]-maxwidth/2):int(center[1]+maxwidth/2),int(center[0]-maxwidth/2):int(center[0]+maxwidth/2)]

                resized_image = cv2.resize(cropped_image,(48,48),interpolation = cv2.INTER_AREA)

                cv2.imwrite(pathFolderWhereToSaveFrames + '/' + basename + "f{}.jpg".format(count), resized_image)

                print('Read a new frame: ', success)
        count += 1
        success, image = vidcap.read()



def main():

    # for i in range(10,25):
    #     directory = 'C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Ravdess database/Video_Speech_Actor_{}/Actor_{}'.format(i,i)
    #     for filename in os.listdir(directory) :
    #         print(filename)
    #         shutil.move(directory+'/'+str(filename),'C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Ravdess database')

    directory = 'C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Ravdess database'

    for filename in os.listdir(directory):
        saveVideoFramesCropped(directory+'/'+filename)
    return

if __name__ == '__main__':
    main()