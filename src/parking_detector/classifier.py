import pickle
import numpy as np
from skimage.transform import resize
from pathlib import Path
from typing import List, Tuple

class SpaceClassifier:
    """Classifier for determining if a parking space is empty or occupied."""
    
    def __init__(self, model_path: str):
        """Initialize the classifier.
        
        Args:
            model_path: Path to the saved classifier model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model
        try:
            self.model = pickle.load(open(model_path, "rb"))
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
        # Constants
        self.TARGET_SIZE = (15, 15)  # Size expected by the model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image for classification.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            np.ndarray: Preprocessed image features
        """
        # Resize image to target size
        resized = resize(image, (*self.TARGET_SIZE, 3))
        
        # Flatten the image
        return resized.flatten().reshape(1, -1)
    
    def predict_single(self, image: np.ndarray) -> Tuple[bool, float]:
        """Predict if a single parking space is empty.
        
        Args:
            image: Input image of parking space (BGR format)
            
        Returns:
            Tuple[bool, float]: (is_empty, confidence)
        """
        # Preprocess the image
        features = self.preprocess_image(image)
        
        # Get prediction
        prediction = self.model.predict(features)[0]
        
        # Get confidence using decision function
        # Convert to probability-like score between 0 and 1
        decision_scores = self.model.decision_function(features)
        confidence = 1 / (1 + np.exp(-np.abs(decision_scores[0])))  # Sigmoid
        
        # Convert prediction to boolean (0 = empty, 1 = occupied)
        is_empty = prediction == 0
        
        return is_empty, confidence
    
    def predict_batch(self, 
                     images: List[np.ndarray]
                     ) -> Tuple[List[bool], List[float]]:
        """Predict multiple parking spaces at once.
        
        Args:
            images: List of parking space images
            
        Returns:
            Tuple[List[bool], List[float]]: (is_empty_list, confidence_list)
        """
        if not images:
            return [], []
            
        # Preprocess all images
        features = np.vstack([
            self.preprocess_image(img) for img in images
        ])
        
        # Get predictions
        predictions = self.model.predict(features)
        
        # Get confidence scores using decision function
        decision_scores = self.model.decision_function(features)
        confidences = 1 / (1 + np.exp(-np.abs(decision_scores)))  # Sigmoid
        
        # Convert predictions to boolean list
        is_empty_list = [pred == 0 for pred in predictions]
        
        return is_empty_list, confidences.tolist()