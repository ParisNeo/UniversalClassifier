"""TextPinner
Author : Saifeddine ALOUI
Description : This code uses Universal classifier to classify apples by color
"""
from UniversalClassifier import UniversalClassifier
from PIL import Image
uc = UniversalClassifier(["red apple", "green apple", "yellow apple"])
output_text, index, similarity=uc.process(Image.open("images/red_apple.jpg")) # try other images red_apple, green_apple, yellow_apple

# If index <0 then the text meaning is too far from any of the anchor texts. You can still use np.argmin(dists) to find the nearest meaning.
# or just change the maximum_distance parameter in your TextPinner when constructing TextPinner

if index>=0:
    # Finally let's give which text is the right one
    print(f"The anchor text you are searching for is:\n {output_text}\n")
else:
    # Text is too far from any of the anchor texts
    print(f"Your text meaning is very far from the anchors meaning. Please try again")

for txt, sim in zip(uc.class_names, similarity):
    print(f"text : {txt}\t\t prob \t\t {sim:0.2f}")
