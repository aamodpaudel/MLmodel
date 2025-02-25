import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, models
from PIL import Image

class DogEmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.resnet18(pretrained=False)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

class DogEmotionPredictor:
    def __init__(self, model_path):
  
        self.model = DogEmotionCNN(num_classes=4)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
      
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
      
        self.classes = ['Angry', 'Happy', 'Sad', 'Relaxed']

    def predict_and_visualize(self, image_path):
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
       
        probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy()
        
        
        plt.figure(figsize=(12, 6))
        
       
        plt.subplot(1, 2, 1)
        self._show_image(image)
        plt.title("Input Image")
        
        
        plt.subplot(1, 2, 2)
        self._plot_probabilities(probabilities)
        
        plt.tight_layout()
        plt.show()

        return {cls: prob for cls, prob in zip(self.classes, probabilities)}

    def _show_image(self, image):
        
       
        image = np.array(image).astype('float32') / 255
        plt.imshow(image)
        plt.axis('off')

    def _plot_probabilities(self, probs):
        
        colors = ['#ff6666', '#66b366', '#6666ff', '#ffcc66']
        bars = plt.barh(self.classes, probs, color=colors)
        plt.xlabel('Probability')
        plt.title('Emotion Probabilities')
        plt.xlim(0, 1)
        
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.02, 
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.1%}',
                    va='center')

if __name__ == "__main__":
   
    predictor = DogEmotionPredictor(
        r"C:\Users\aamod\MLmodel\trained_model\dog_emotion_model.pth"
    )
    
   
    try:
        results = predictor.predict_and_visualize(
            r"C:\Users\aamod\MLmodel\predictions\test_image.jpg"
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")