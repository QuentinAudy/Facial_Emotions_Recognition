# Facial_Emotions_Recognition

	Here is a Readme file where you can find the purpose of each python file, and how to use them. 


Main.py: This code is the first one to be created. The purpose was to train different models that recognize the face in an image. 
We have the imagePreorocessing function, that takes the different images, and perform data augmentation. It also divides them into batches. 
The baselineModel function is a simple ANN model tried at first. The CNNsimpleModel is the CNN model we created from scratch. 
The defineCNNModelVGGPretrained function is for the VGG16 model, and for the VGGFace (we can choose the model by commenting and uncommenting the lines). 
The visualizeTheTrainingPerformances function is a simple function used to display the final histograms at the end of the training. 
And finally, the main function takes the dataset (we can here choose the original one, or an augmented by fake images one), 
and apply all the previous functions. At the end, we have a saved model, and the histogram of the training. 



evaluation.py: This code is the one used for testing our image emotion recognition model. This prediction function takes a test image as input, 
preprocess it in order to make it fit the neural network, load the trained model and gives a prediction. There are two possibilities for this code. 
If we want to predict only one picture without using the global code, we must uncomment lines 21, 22 (path of the tested image), 
and line 51 (call of the prediction function). If we use the Global.py or Global_V2.py file, we must comment these lines 
(because these two files use the prediction function).



MTCNN.py: This code is used to take a video, to crop all the frames and to detect the faces in them. As an output, it gives us a collection of pictures 
with only faces, taken from the original video. We implemented this code in order to augment our dataset, but also after to have the multimodality.



realtime.py: This code is the one used to recognize the emotion of our face, thanks to the webcam of the computer in real time. We just have to run 
the code, no parameter has to be changed. This code uses our saved optimized model. 



helpers.py: This code contains some functions used to transform audio files into spectrograms. We don’t have to touch it, it is just a function library 
used by the other files.



audioconversion.py: This code is used to convert all the audio files into spectrograms. It uses helpers, and contains different function depending on 
which transform we want to use (MFCC or Mel Spectrogram).



MELCNN.py: This code is used to take all the spectrograms, to load the basic model, and to train everything in order to have a final model with a 
65% accuracy. This one works like the Main.py, but for the audio recognition. 



audioevalution.py: This code has the same purpose as the evaluation.py, but for the audio part. We take an audio file as an input, 
and the function extracts the spectrogram, reshapes it, and uses our trained model to give a prediction. There are two possibilities for this code. 
If we want to predict only one audio without using the global code, we must uncomment lines 25 (path of the tested audio), 
and line 67 (call of the prediction function). If we use the Global.py or Global_V2.py file, we must comment these lines 
(because these two files use the prediction function).



Global.py: This code is the first version of the multimodality. We just have to run this one, and everything is done automatically using the other files.
We just have to put our test video in the Full_process folder, to change the paths in the beginning of the code, and that’s it ! 
The file extracts the audio, selects one frame in the middle of the video, and uses evaluation.py and audioevaluation.py to predict the emotion. 
Finally, with a rule of choice created by ourself, we obtain a global prediction. 



Global_V2.py: This version is the final one, more complex and using all the tools we implemented. This one is similar to the previous one, 
but instead of taking the middle picture of the video, it uses MTCNN to create a collection of pictures, cropped around the head. 
It predicts the emotion for each picture, and keeps the most frequent one. We have the video prediction, and a probability. For the audio part, 
it remains the same, but we also return the probability. Finally, the choice for the final prediction is not a rule of choice, 
but a comparison of the probabilities.


And for the folders we have these ones:




archive: The first FER2013 dataset that we had, but it didn’t work well so it is unused. 


Audioset: This is the Ravdess audio dataset used to train our audio model. Inside, we have the Full folder, 
where we have all the data already sorted (train, test and all the emotion folders). We also have the folder Statement 1 where 
we just have the first sentence, and the folder Statement 2 with the second sentence. 



FER: Original FER2013 dataset, used to train our image emotion recognition model. We have a training folder, 
a testing folder but also an evaluation folder where I put some images to test the final model. 



FER_Extend and FER_Extend_2: This is the original FER2013 dataset, but extended with our data augmentation (using Dall.e and OpenAI).



Full_process: This is the folder used for the multimodality. Inside, we must put our test video. If we run Global.py or Global_V2.py, we obtain the audio file that will be stored in this same folder. In MTCNN folder, we have all the images cropped from the original video, that will be used after for all the predictions. 




Finally, we have some files, that are the model we saved:



200Epoch64BatchAugmented.hdf5: This is our final audio model, with data augmentation.

basic-model-arch.hdf5: This is the first audio model, we trained on it.

vggface_test.h5: This is our final model, used for image emotion recognition
