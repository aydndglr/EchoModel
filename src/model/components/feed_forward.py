
# src/model/components/feed_forward.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.model_config import ModelConfig
from typing import Callable, Union

class FeedForward(nn.Module):
    """
    Transformer bloğundaki Konum Bazlı İleri Besleme Ağı (Position-wise Feed-Forward Network).
    İki doğrusal katmandan ve aralarında bir aktivasyon fonksiyonundan oluşur.
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.d_model = model_config.d_model
        self.d_ff = model_config.d_ff # Genellikle d_model * 4
        self.dropout_p = model_config.dropout
        self.activation_function_name = model_config.activation_function

        # Giriş doğrusal katmanı (d_model -> d_ff)
        # ModelConfig'te 'include_bias' ayarı yoksa varsayılan olarak True kullan
        bias_for_projections = getattr(model_config, 'include_bias', True)
        self.fc1 = nn.Linear(self.d_model, self.d_ff, bias=bias_for_projections)
        
        # Çıkış doğrusal katmanı (d_ff -> d_model)
        self.fc2 = nn.Linear(self.d_ff, self.d_model, bias=bias_for_projections)
        
        self.dropout = nn.Dropout(self.dropout_p)

        # Aktivasyon fonksiyonunu dinamik olarak seç
        self.activation = self._get_activation_function(self.activation_function_name)

    def _get_activation_function(self, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Belirtilen isimdeki aktivasyon fonksiyonunu döndürür.
        """
        if name == "relu":
            return F.relu
        elif name == "gelu":
            return F.gelu # PyTorch'un GELU implementasyonu
        elif name == "silu":
            return F.silu
        else:
            raise ValueError(f"Desteklenmeyen aktivasyon fonksiyonu: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Giriş tensörü (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: İleri besleme ağı çıktısı (batch_size, seq_len, d_model).
        """
        # Katman 1: Doğrusal dönüşüm
        x = self.fc1(x)
        
        # Aktivasyon
        x = self.activation(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Katman 2: Doğrusal dönüşüm
        x = self.fc2(x)
        
        # Son dropout (genellikle residual bağlantıdan önce)
        x = self.dropout(x) # OLMO'da bu dropout, FFN çıktısına uygulanır ve sonra residual bağlantıya eklenir.

        return x

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("FeedForward katmanı testi başlatılıyor...")

    # Test için gerekli config'leri doğrudan içeri aktarıyoruz.
    from src.config.base_config import BaseConfig # BaseConfig'i de aktaralım, gerekli olabilir
    
    base_cfg = BaseConfig() # Varsayılan değerlerle
    model_cfg = ModelConfig() # Varsayılan değerlerle

    # Parametreler
    batch_size = 2
    seq_len = 32
    d_model_test = model_cfg.d_model # 512
    d_ff_test = model_cfg.d_ff # Otomatik olarak 512 * 4 = 2048 olmalı

    # Rastgele giriş tensörü
    dummy_input = torch.randn(batch_size, seq_len, d_model_test)
    print(f"Giriş tensörü boyutu: {dummy_input.shape}")

    # FeedForward katmanı oluştur
    feed_forward_layer = FeedForward(model_config=model_cfg)

    # İleri besleme testi
    output = feed_forward_layer(dummy_input)
    print(f"Çıktı boyutu: {output.shape} (Beklenen: {batch_size, seq_len, d_model_test})")
    assert output.shape == (batch_size, seq_len, d_model_test), "FeedForward çıktı boyutu yanlış!"

    print("FeedForward katmanı testi başarılı. ✅")