# src/config/model_config.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class ModelConfig:
    """
    EchoModel için model mimarisine özel yapılandırma ayarları.
    Bu ayarlar, Transformer modelinin katman boyutları ve sayıları gibi detaylarını tanımlar.
    """
    
    # Model Boyutları
    d_model: int = 512 # Gömme boyutu ve Transformer katmanlarındaki gizli boyut
                    # OLMO'da bu `d_model` veya `model_dim` olarak geçer.
    n_layers: int = 6 # Transformer Decoder katman sayısı (OLMO: `n_layers`)
    n_heads: int = 8 # Çoklu Kafa Dikkat mekanizmasındaki kafa sayısı (OLMO: `n_attention_heads`)
    
    # İleri Besleme Ağı (Feed-Forward Network) Boyutu
    # Genellikle d_model'in 4 katı olarak belirlenir.
    d_ff: Optional[int] = None # İleri besleme ağının gizli boyutu (OLMO: `mlp_hidden_size`)
    
    # Dropout Oranları
    dropout: float = 0.1 # Genel dropout oranı (embeddings, attention, feed-forward) (OLMO: `attention_dropout_p`, `resid_p`, `mlp_p` gibi ayrı ayrı yönetilir)
    embedding_dropout: Optional[float] = None # Gömme katmanı için özel dropout (varsayılan: genel dropout)
    
    # Aktivasyon Fonksiyonu
    activation_function: str = "gelu" # Transformer katmanlarındaki aktivasyon fonksiyonu (örn. "gelu", "relu")
                                    # OLMO varsayılan olarak GELU kullanır.

    # Layer Normalizasyon Ayarları
    # OLMO'da Layer Norm'un konumu (pre-norm veya post-norm) ve normalizasyon türü (RMSNorm) gibi detaylar.
    use_rms_norm: bool = True # Layer Normalizasyon yerine RMSNorm kullanılıp kullanılmayacağı (OLMO RMSNorm kullanır)
    pre_normalization: bool = True # Normalizasyonun dikkat/FFN bloğundan önce mi (pre-norm) sonra mı (post-norm) uygulanacağı (OLMO pre-norm kullanır)
    
    # Modelin başlatılması ile ilgili ayarlar (detaylı init için)
    # OLMO, kendi ağırlık başlatma stratejilerini kullanır (örn. 'kaiming_normal_').
    # Bu kısmı model.py içinde veya util/initialization.py gibi ayrı bir dosyada işleyeceğiz.
    # Burada sadece bir placeholder olarak tutalım:
    initializer_range: float = 0.02 # Ağırlıkların başlatılacağı aralık
    
    def __post_init__(self):
        """Dataclass başlatıldıktan sonra kontroller ve otomatik ayarlamalar."""
        # Eğer d_ff belirtilmemişse, varsayılan olarak d_model * 4 yap.
        if self.d_ff is None:
            self.d_ff = self.d_model * 4
        
        # d_model'in n_heads'e tam bölünebilirliğini kontrol et.
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) n_heads ({self.n_heads})'e tam bölünmeli.")
        
        # Eğer embedding_dropout özel olarak belirtilmemişse, genel dropout'u kullan.
        if self.embedding_dropout is None:
            self.embedding_dropout = self.dropout
        
        # Aktivasyon fonksiyonu kontrolü
        if self.activation_function not in ["relu", "gelu", "silu"]:
            raise ValueError(f"Desteklenmeyen aktivasyon fonksiyonu: {self.activation_function}. 'relu', 'gelu' veya 'silu' olmalı.")