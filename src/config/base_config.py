# src/config/base_config.py

from dataclasses import dataclass, field
import os
from typing import Optional, List, Dict, Any

@dataclass
class BaseConfig:
    """
    EchoModel için temel yapılandırma ayarları.
    Bu ayarlar, model mimarisi, eğitim süreci ve veri yükleme için genel varsayılanları tanımlar.
    """

    # --- Genel Ayarlar ---
    seed: int = 42 # Rastgelelik tohumu, yeniden üretilebilir sonuçlar için
    device: str = "cuda" # 'cuda' (GPU için) veya 'cpu' (CPU için)
    log_level: str = "INFO" # Loglama seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    output_dir: str = "runs" # Model checkpoint'lerinin ve logların kaydedileceği ana dizin

    # --- Model Ortak Ayarları ---
    vocab_size: int = 50257 # Tokenizer'ın kelime haznesi boyutu (varsayılan: GPT-2'ye yakın)
    max_seq_len: int = 512 # Modelin işleyebileceği maksimum giriş dizisi uzunluğu
    
    # --- Eğitim Ayarları ---
    train_batch_size: int = 4 # Eğitim batçesi boyutu
    eval_batch_size: int = 4 # Değerlendirme batçesi boyutu
    gradient_accumulation_steps: int = 1 # Gradyan biriktirme adımları (daha büyük sanal batçe için)
    
    # --- Optimizasyon Ayarları ---
    learning_rate: float = 1e-4 # Başlangıç öğrenme oranı
    weight_decay: float = 0.01 # Ağırlık bozunması (L2 regularizasyonu)
    adam_beta1: float = 0.9 # Adam optimizer'ın beta1 parametresi
    adam_beta2: float = 0.95 # Adam optimizer'ın beta2 parametresi
    adam_epsilon: float = 1e-8 # Adam optimizer'ın epsilon parametresi
    max_grad_norm: float = 1.0 # Gradyan patlamasını engellemek için maksimum gradyan normu

    # --- Planlayıcı (Scheduler) Ayarları ---
    lr_scheduler_type: str = "cosine" # Öğrenme oranı çizelgesi tipi (örn. "linear", "cosine", "constant")
    warmup_steps: int = 1000 # Öğrenme oranının ısınma dönemi adımı
    
    # --- Checkpoint ve Loglama ---
    save_steps: int = 5000 # Kaç adımda bir model checkpoint'i kaydedileceği
    eval_steps: int = 1000 # Kaç adımda bir modelin değerlendirileceği
    log_steps: int = 100 # Kaç adımda bir log basılacağı (eğitim ilerlemesi)
    
    # --- Veri Yükleme Ayarları ---
    num_workers: int = 4 # Veri yükleme için kullanılacak işçi sayısı
    
    # --- Diğer Ayarlar ---
    dtype: str = "float32" # Modelin hesaplama veri tipi ("float32", "float16", "bfloat16")
    
    def __post_init__(self):
        """Dataclass başlatıldıktan sonra kontroller ve eklemeler."""
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Geçersiz cihaz: {self.device}. 'cuda' veya 'cpu' olmalı.")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Geçersiz log seviyesi: {self.log_level}")

        # Otomatik olarak tam batch boyutunu hesapla
        self.effective_batch_size = self.train_batch_size * self.gradient_accumulation_steps

        # Çıktı dizini oluştur
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class ModelConfig:
    """
    Model mimarisine özel yapılandırma ayarları.
    Bu ayarlar, Transformer modelinin katman boyutları ve sayıları gibi detaylarını tanımlar.
    """
    d_model: int = 512 # Gömme boyutu ve Transformer katmanlarındaki gizli boyut
    n_layers: int = 6 # Transformer Decoder katman sayısı
    n_heads: int = 8 # Çoklu Kafa Dikkat mekanizmasındaki kafa sayısı
    d_ff: Optional[int] = None # İleri besleme ağının gizli boyutu (varsayılan olarak d_model * 4)
    dropout: float = 0.1 # Genel dropout oranı (embeddings, attention, feed-forward)
    
    # Gömme katmanları için ek ayarlar
    # (OLMO'da embedding_size model boyutundan farklı olabilir, biz şimdilik aynı tutalım)
    embedding_dropout: Optional[float] = None # Gömme katmanı için özel dropout (varsayılan: genel dropout)

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = self.d_model * 4
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) n_heads ({self.n_heads})'e tam bölünmeli.")
        if self.embedding_dropout is None:
            self.embedding_dropout = self.dropout

# --- Config'leri bir araya getirme (main.py veya trainer.py'de kullanılacak) ---
# Örnek kullanım:
# from src.config.base_config import BaseConfig, ModelConfig
# base_cfg = BaseConfig()
# model_cfg = ModelConfig()