import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
import torchvision.models as models

class InceptionV3FeatureExtractor(nn.Module):
    """Extrator de características para cálculo do FID - CORRIGIDO"""
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        
        # Carregar InceptionV3 pré-treinado CORRETAMENTE
        # Para versões mais recentes do torchvision, aux_logits tem comportamento diferente
        try:
            # Tentar carregar sem especificar aux_logits primeiro
            inception = models.inception_v3(pretrained=True)
        except TypeError as e:
            # Se falhar, tentar com aux_logits=False
            if "aux_logits" in str(e):
                inception = models.inception_v3(pretrained=True, aux_logits=False)
            else:
                raise e
        
        # Configurar para extrair características da última camada antes da classificação
        # O InceptionV3 tem saídas auxiliares, precisamos pegar a saída principal
        self.features = nn.Sequential(
            *list(inception.children())[:-1]  # Remover última camada (classificação)
        )
        
        # Alternativa mais robusta: usar apenas as camadas convolucionais
        # self.features = nn.Sequential(
        #     inception.Conv2d_1a_3x3,
        #     inception.Conv2d_2a_3x3,
        #     inception.Conv2d_2b_3x3,
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     inception.Conv2d_3b_1x1,
        #     inception.Conv2d_4a_3x3,
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     inception.Mixed_5b,
        #     inception.Mixed_5c,
        #     inception.Mixed_5d,
        #     inception.Mixed_6a,
        #     inception.Mixed_6b,
        #     inception.Mixed_6c,
        #     inception.Mixed_6d,
        #     inception.Mixed_6e,
        #     inception.Mixed_7a,
        #     inception.Mixed_7b,
        #     inception.Mixed_7c,
        # )
        
        # Congelar parâmetros
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Configurar modo de avaliação
        self.eval()
        
    def forward(self, x):
        """Extrai características de um batch de imagens"""
        # Redimensionar imagens para 299x299 (tamanho esperado pelo Inception)
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Expandir para 3 canais (RGB) se for grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            try:
                features = self.features(x)
            except Exception as e:
                # Fallback: usar camadas específicas
                print(f"Erro ao extrair features: {e}")
                print("Usando fallback...")
                features = self._extract_features_fallback(x)
            
            # Pooling adaptativo para vetorizar
            features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
            features = features.view(features.size(0), -1)
        
        return features.cpu().numpy()
    
    def _extract_features_fallback(self, x):
        """Fallback para extração de features"""
        # Implementação simplificada para caso a extração padrão falhe
        from torchvision.models.inception import Inception3
        
        # Criar modelo manualmente se necessário
        model = Inception3(aux_logits=False, init_weights=False)
        
        # Carregar pesos pré-treinados
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
                progress=False
            )
            model.load_state_dict(state_dict)
        except:
            pass
        
        # Usar apenas até a camada antes da classificação
        features = []
        
        def hook(module, input, output):
            features.append(output)
        
        # Registrar hook na última camada antes do fc
        handle = model.fc.register_forward_hook(hook)
        
        with torch.no_grad():
            _ = model(x)
        
        handle.remove()
        
        return features[0] if features else x