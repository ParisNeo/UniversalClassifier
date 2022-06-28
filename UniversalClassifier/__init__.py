"""TextPinner
Author      : Saifeddine ALOUI
Licence     : MIT
Description : 
This class can be used to classify any image with unlimited class numbers
"""
import torch
# if you don't have clip, just download it 
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
import clip 
from PIL import Image
import numpy as np


class UniversalClassifier():
    def __init__(self, class_names:list, minimum_similarity_level:float=None):
        """Builds the TextPinner

        Args:
            classes (list[str]) : The list of class names
            minimum_similarity_level (float) : The minimum acceptable similarity between the image and the classes (to avoid classifying images that are very far from all classes)
        """
        self.class_names = class_names
        self.minimum_similarity_level = minimum_similarity_level
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.class_names_tokenized = clip.tokenize(self.class_names).to(self.device)
        with torch.no_grad():
            # Now let's encode the class names
            self.class_names_embedding = self.clip.encode_text(self.class_names_tokenized) # Anchor texts
            self.class_names_embedding /= self.class_names_embedding.norm(dim=-1, keepdim=True)

    def process(self, image:Image):
        """Processes text ang gives the text entended to

        Args:
            command_text (str): The command text to pin to one of the texts list

        Returns:
            str, int, list: _description_
        """
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip.encode_image(image)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ self.class_names_embedding.T).softmax(dim=-1).detach().numpy()[0,:]
        max = similarity.max()
        if self.minimum_similarity_level is not None:
            if max<self.minimum_similarity_level:
                return "",-1, similarity
        text_index = similarity.argmax()
        return self.class_names[text_index], text_index, similarity
        """
        
        dists = [np.square((self.class_names_embedding[i,:]-image_features[0,:])).mean() for i in range(len(self.class_names))]
        # Convert distances to probabilities
        mn = np.min(dists)
        # If there is a maximum accepted distance then verify!
        if self.maximum_distance is not None:
            if mn>self.maximum_distance:
                # None of the anchors has a meaning near the one proposed by the command_text
                return "", -1, [0 for d in dists], dists

        mx = np.max(dists)
        range_ = mx-mn
        # Compute a probability value
        prob = np.array([1-(d-mn)/range_ for d in dists])
        prob = prob/prob.sum()
        # Get the index of the nearest anchor
        text_index = np.argmin(dists)
        # Return everything
        return self.class_names[text_index], text_index, prob, dists
        
        """

