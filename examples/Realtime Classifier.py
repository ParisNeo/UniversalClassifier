"""TextPinner
Author : Saifeddine ALOUI
Description : This uses 


You can imagine doing this for a hue lighting system or other command based app. Be creative
"""
from UniversalClassifier import UniversalClassifier
from PIL import Image
import cv2
import numpy as np

# use one of these or build your own classes and play with them. Make sure the threshold you use is higher than the minimum similarity level between the image and the class texts.
# You can try using None, spot the threshold value then put it.

# Apples classifier. just execute the code and show some apples and it will classify them
# tp = UniversalClassifier(["red apple", "green apple", "yellow apple"], 0.6)

# Male female classifier. Just let people stay in front of the camera and the algorithm will recognize their gender
# tp = UniversalClassifier(["a man", "a woman", "a young girl", "a young boy"], 0.45)

# Emotions classifier
# tp = UniversalClassifier(["a happy face", "a sad face", "a surprised face", "an angry face"], 0.50)

# Aples count classifier (for this to work, please make sure you show an empty background then start putting apples 1,2, or 3 and it should work)
# tp = UniversalClassifier(["single apple", "two apples", "three apples"], 0.70)

# Complex caracteristics description, try show images to the camera and you'll see.
skin_colors = ["white", "black", "yellow", "broun"]
genders = ["male","female"]
ages = ["a child","an adult","an old"]
beards = ["bearded", "beardless"]
hairs = ["black hair", "blonde hair", "white hair", "red hair"]
tp = UniversalClassifier([f"{age} {beard if (gender=='male' and age!='child') else ''} {skin_color} {gender} with {hair}" for beard in beards for skin_color in skin_colors for gender in genders for age in ages for hair in hairs], None)

# Now you come up with your own classifications.

# open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read image
    success, cv_image = cap.read()
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    output_text, index, similarity=tp.process(image) # try other images red_apple, green_apple, yellow_apple

    cv_image = cv2.resize(cv_image, (1920, 1080))
    # If index <0 then the text meaning is too far from any of the anchor texts. You can still use np.argmin(dists) to find the nearest meaning.
    # or just change the maximum_distance parameter in your TextPinner when constructing TextPinner
    if index>=0:
        # Finally let's give which text is the right one
        cv2.putText(cv_image, output_text + f" (sim={100*np.max(similarity):0.2f}%)", (10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        # Text is too far from any of the anchor texts
        cv2.putText(cv_image, "Unknown", (10,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    
    # Show the image
    cv2.imshow('Video', cv_image)
    # Wait for key stroke for 5 ms
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break


