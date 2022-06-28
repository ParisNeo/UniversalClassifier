# UniversalClassifier
A universal tool to classify images based on Contrastive Learning Image/Text pre-training (CLIP for short). Use natural language to classify images with no borders. One algorithm to rule them all. CLIP is an Open-AI neural network that Universal classifier uses at its core to acheive unbounded image classification with high accuracy.

# How it works
What if we can find a 512 dimensions space where we can represent the meaning of words and in the same time the meaning of the content of images? This would make us able to find the nearest description of an image by projecting both the image and the texts in this space. Then find the dimilarity inside this space and make a decision what text is the nearest in meaning to the content of the image.

Well that's what Open AI's CLIP model can be used for. This is a very powerful Idea as we show here, we can exploit this to build a boundless universal classifier to classify images in any context.

If you are interested in more detains please read Open AI's paper about CLIP : [paper](https://arxiv.org/abs/2103.00020)

# Install
To install UniversalClassifier, just type:
```bash
pip install UniversalClassifier
```
It is advised to install cudatoolkit if you have a cuda enabled GPU.
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 
```

You build a UniversalClassifier instance as well as Image from pollow
```python
from UniversalClassifier import UniversalClassifier
from PIL import Image
```

Create an instance of UniversalClassifier. There are one mandatory parameter which is the list of class names, and an optional parameter which is the minimum similarity between the image and the classes. This allows the AI to detect if the image the user is entering is too far from any of the classes. By default the value is None (don't check for minimal distance). A value of 0.5 has proven to be a good distance for the tests we have done but this can be changed depending on the anchors you are using. Feel free to use another value :
```Python
uc = UniversalClassifier(["raise right hand", "raise left hand", "nod", "shake hands", "look left", "look right"], minimum_similarity_level=0.5)
```

Now we are ready to classify some images. Here we use PIL images. You can also use opencv images and convert them to PIL images using Image.fromarray(image) (don't forget to change the channels ordre from BGR t RGB first using : image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)).

To classify from a file:
```Python
image = Image.open("images/red_apple.jpg")
output_text, index, similarity=uc.process(image)
```

To classify from an opencv image (cv_image) :
```Python
image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
output_text, index, similarity=uc.process(image) # try other images red_apple, green_apple, yellow_apple
```
- The index tells you which text of your anchors list is most likely to have the same meaning as the text_command. If it is -1, this means that the meaning of the text is too far from any of the anchors. If maximum_distance is None then there is no maximum distance test and the AI will return the anchor with nearest meaning.
- output_text is literally the anchor text that has the nearest meaning to the one of text_command.
- similarity is a numpy array containing the similarity of this text with each of the anchor texts. Useful to get an idea about the certainty of the algorithm about its decision.