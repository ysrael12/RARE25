import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

class SwinTinyClassificationModel:
    def __init__(self, weights: str, num_classes: int = 2, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Re-cria a arquitetura Swin-Tiny
        self.model = models.swin_t(weights=None)
        
        # Ajusta o classificador para 2 classes
        in_f = self.model.head.in_features
        self.model.head = nn.Linear(in_f, num_classes)
        
        # Carrega o estado do seu modelo treinado
        self.model.load_state_dict(torch.load(weights, map_location=self.device), strict=True)
        self.model.to(self.device).eval()

        # Usa as mesmas normalizações que você definiu no seu código
        try:
            swin_w = models.Swin_T_Weights.DEFAULT
            MEAN, STD = swin_w.meta["mean"], swin_w.meta["std"]
        except Exception:
            MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            
        # Define as transformações de validação/inferência
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def predict(self, images: list[np.ndarray]):
        """
        Aceita uma lista de imagens numpy (HWC)
        e retorna a probabilidade da classe 1.
        """
        pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
        probs = []
        for img in pil_images:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(img_tensor)
                # O seu modelo retorna logits de 2 classes, então usamos softmax
                # para obter a probabilidade da classe 1.
                prob = torch.softmax(logits, dim=1)[:, 1].squeeze().cpu().item()
            probs.append(prob)
        return probs
