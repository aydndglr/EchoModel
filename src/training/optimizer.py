
# src/training/optimizer.py

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
import logging
from typing import Dict, Any

from src.config.base_config import BaseConfig

log = logging.getLogger(__name__)

def get_optimizer(model: nn.Module, config: BaseConfig) -> AdamW:
    """
    Model parametreleri için AdamW optimizer'ını oluşturur.
    Ağırlık bozunması (weight decay) ile parametre gruplarını ayırır.

    Args:
        model (nn.Module): Optimize edilecek PyTorch modeli.
        config (BaseConfig): Optimizer ayarlarını içeren BaseConfig objesi.

    Returns:
        torch.optim.AdamW: Yapılandırılmış AdamW optimizer.
    """
    # Ağırlık bozunmasını sadece ağırlık (weight) parametrelerine uygula, bias ve LayerNorm ağırlıklarına uygulama.
    # Bu, çoğu modern LLM eğitiminde standart bir uygulamadır.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_f.weight"] # OLMO'da da benzer şekilde
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    log.info(f"AdamW optimizer oluşturuldu. Learning Rate: {config.learning_rate}, Weight Decay: {config.weight_decay}")
    return optimizer


def get_scheduler(optimizer: AdamW, config: BaseConfig, num_training_steps: int) -> LambdaLR:
    """
    Öğrenme oranı çizelgesini (scheduler) oluşturur.
    Cosine veya Linear decay ile warmup periyodu destekler.

    Args:
        optimizer (AdamW): Öğrenme oranı uygulanacak optimizer.
        config (BaseConfig): Planlayıcı ayarlarını içeren BaseConfig objesi.
        num_training_steps (int): Toplam eğitim adımı sayısı.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Yapılandırılmış öğrenme oranı çizelgesi.
    """
    warmup_steps = config.warmup_steps
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Warmup periyodu: Öğrenme oranı 0'dan başlar ve `learning_rate`'e lineer olarak yükselir.
            return float(current_step) / float(max(1, warmup_steps))
        
        # Deçay periyodu: Cosine veya Linear decay
        if config.lr_scheduler_type == "cosine":
            # Cosine decay: learning rate'i düşürür
            # https://huggingface.co/docs/transformers/main_classes/optimizer_and_interation#transformers.optimization.get_cosine_schedule_with_warmup
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif config.lr_scheduler_type == "linear":
            # Linear decay: learning rate'i lineer olarak 0'a düşürür
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 1.0 - progress)
        elif config.lr_scheduler_type == "constant":
            return 1.0 # Sabit öğrenme oranı (warmup sonrası)
        else:
            raise ValueError(f"Desteklenmeyen lr_scheduler_type: {config.lr_scheduler_type}")

    scheduler = LambdaLR(optimizer, lr_lambda)
    log.info(f"Öğrenme oranı çizelgesi oluşturuldu. Tip: {config.lr_scheduler_type}, Warmup Adımları: {warmup_steps}, Toplam Adım: {num_training_steps}")
    return scheduler

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("Optimizer ve Scheduler testi başlatılıyor...")

    import shutil
    from src.utils.logger import setup_logging
    from src.config.base_config import BaseConfig

    test_output_dir = "test_runs_optimizer"
    test_run_name = "optimizer_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # Test için dummy model ve config
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 1)
            self.norm = nn.LayerNorm(10) # LayerNorm parametresi
            self.bias_param = nn.Parameter(torch.zeros(1)) # Ayrı bir bias parametresi

        def forward(self, x):
            return self.linear2(self.linear1(self.norm(x) + self.bias_param))

    dummy_model = DummyModel()
    config = BaseConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type="cosine",
        warmup_steps=100
    )
    total_steps = 1000

    # Optimizer testi
    print("\n--- Optimizer Oluşturma Testi ---")
    optimizer = get_optimizer(dummy_model, config)
    # Parametre gruplarının doğru ayrıldığını kontrol et
    # linear1.weight ve linear2.weight weight_decay'e tabi olmalı
    # norm.weight ve bias_param weight_decay'e tabi olmamalı
    found_decay_group = False
    found_no_decay_group = False
    for group in optimizer.param_groups:
        if group["weight_decay"] == config.weight_decay:
            # Ağırlıkların bu grupta olduğunu kontrol et
            assert any("linear" in n for n in [name for name, p in dummy_model.named_parameters() if p in group["params"]]), "Weight decay grubunda lineer katman ağırlığı yok!"
            found_decay_group = True
        elif group["weight_decay"] == 0.0:
            # Bias ve LayerNorm ağırlıklarının bu grupta olduğunu kontrol et
            assert any("norm.weight" in n or "bias_param" in n for n in [name for name, p in dummy_model.named_parameters() if p in group["params"]]), "No decay grubunda LayerNorm veya bias yok!"
            found_no_decay_group = True
    assert found_decay_group and found_no_decay_group, "Optimizer parametre grupları yanlış ayrıldı!"
    print("Optimizer oluşturma testi başarılı. ✅")

    # Scheduler testi
    print("\n--- Scheduler Oluşturma ve LR Değişimi Testi ---")
    scheduler = get_scheduler(optimizer, config, total_steps)
    
    # Warmup dönemi
    lr_at_step_50 = scheduler.get_last_lr()[0] # İlk LR
    # Mock optimizer'ın LR'sini güncelleyerek ilerleyelim
    optimizer.param_groups[0]['lr'] = config.learning_rate * (50 / 100) # Doğrudan hesapla
    scheduler.step(50) # Adım 50
    lr_after_warmup_50 = scheduler.get_last_lr()[0]
    # Yaklaşık olarak beklenen değerleri kontrol et
    assert abs(lr_after_warmup_50 - (config.learning_rate * (50 / 100))) < 1e-6, "Warmup LR'si yanlış!"
    print(f"  Warmup (step 50) LR: {lr_after_warmup_50:.6f} (Beklenen: {config.learning_rate * (50 / 100):.6f})")

    # Cosine decay dönemi (son adıma yakın)
    optimizer.param_groups[0]['lr'] = config.learning_rate # Tekrar base LR'ye getir (manuel simülasyon)
    scheduler.step(total_steps - 1) # Son adımdan bir önceki adım
    lr_at_end = scheduler.get_last_lr()[0]
    print(f"  Son adıma yakın LR: {lr_at_end:.6f} (Beklenen: ~0.0)") # Cosine decay'de sona doğru 0'a yaklaşır
    assert lr_at_end < config.learning_rate / 100, "Cosine decay LR'si yanlış (çok yüksek)!"
    print("Scheduler oluşturma ve LR değişimi testi başarılı. ✅")

    print("\nOptimizer ve Scheduler tüm testleri tamamlandı. ✅")
    
    # Temizlik
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")