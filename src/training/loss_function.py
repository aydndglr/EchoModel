
# src/training/loss_function.py

import os
import torch
import torch.nn as nn
import logging

log = logging.getLogger(__name__)

class LanguageModelingLoss(nn.Module):
    """
    Dil modelleme görevi için Çapraz Entropi Kaybı fonksiyonunu tanımlar.
    Bu kayıp, modelin bir sonraki tokeni ne kadar doğru tahmin ettiğini ölçer.
    Padding tokenlarını (genellikle -100) otomatik olarak yoksayar.
    """
    def __init__(self, ignore_index: int = -100):
        """
        Args:
            ignore_index (int): Kayıp hesaplamasında yoksayılacak hedef etiket ID'si.
                                Dil modellemede bu genellikle padding tokenlarının ID'sidir.
        """
        super().__init__()
        # PyTorch'un kendi CrossEntropyLoss'unu kullanıyoruz.
        # Bu, logits (ham model çıktısı) ve hedef etiketleri alır.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        log.info(f"LanguageModelingLoss başlatıldı. Yoksayılacak indeks: {ignore_index}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Kayıp hesaplamasını yapar.

        Args:
            logits (torch.Tensor): Modelin çıkış logits'leri (batch_size, seq_len, vocab_size).
            labels (torch.Tensor): Gerçek token ID'leri (batch_size, seq_len).
                                   Padding tokenları `ignore_index` değeriyle işaretlenmelidir.

        Returns:
            torch.Tensor: Hesaplanan ortalama kayıp değeri (skaler).
        """
        # CrossEntropyLoss beklentisi:
        # logits: (N, C) veya (N, C, D1, D2, ...)
        # labels: (N) veya (N, D1, D2, ...) (int veya long)
        
        # Bizim logits (batch_size, seq_len, vocab_size) şeklinde.
        # Bunu (batch_size * seq_len, vocab_size) şekline getirmeliyiz.
        # Labels da (batch_size, seq_len) şeklinde.
        # Bunu (batch_size * seq_len) şekline getirmeliyiz.
        
        # reshape logits to (batch_size * seq_len, vocab_size)
        logits_reshaped = logits.view(-1, logits.size(-1))
        # reshape labels to (batch_size * seq_len)
        labels_reshaped = labels.view(-1)

        loss = self.loss_fn(logits_reshaped, labels_reshaped)
        return loss

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("LanguageModelingLoss testi başlatılıyor...")

    import shutil
    from src.utils.logger import setup_logging
    
    test_output_dir = "test_runs_loss_function"
    test_run_name = "lm_loss_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # Test parametreleri
    batch_size = 2
    seq_len = 5
    vocab_size = 100
    ignore_index = -100

    # Sahte logits ve etiketler oluştur
    # Logits: (batch_size, seq_len, vocab_size)
    # 0. batch: İlk token doğru tahmin edildi, ikinci yanlış, üçüncü doğru, son ikisi padding.
    # 1. batch: Tüm tokenlar doğru tahmin edildi.
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor([
        [10, 20, 30, ignore_index, ignore_index], # İlk batch: 2 padding
        [50, 60, 70, 80, 90]                      # İkinci batch: padding yok
    ], dtype=torch.long)

    # Logits'i doğru tahminleri gösterecek şekilde ayarla
    logits[0, 0, 10] += 10.0 # İlk örnek, 0. token: 10'u doğru tahmin et
    logits[0, 1, 25] += 10.0 # İlk örnek, 1. token: 20 yerine 25'i tahmin et (yanlış)
    logits[0, 2, 30] += 10.0 # İlk örnek, 2. token: 30'u doğru tahmin et

    logits[1, 0, 50] += 10.0
    logits[1, 1, 60] += 10.0
    logits[1, 2, 70] += 10.0
    logits[1, 3, 80] += 10.0
    logits[1, 4, 90] += 10.0

    # Kayıp fonksiyonunu oluştur
    loss_calculator = LanguageModelingLoss(ignore_index=ignore_index)

    print("\n--- Kayıp Hesaplama Testi (Padding ile) ---")
    calculated_loss = loss_calculator(logits, labels)
    
    print(f"Hesaplanan Kayıp: {calculated_loss.item():.4f}")

    # Manuel olarak beklenen kaybı hesaplama (sadece aktif tokenlar için)
    # CrossEntropyLoss, log_softmax + NLLLoss kombinasyonudur.
    # log_softmax(logits) + NLLLoss(labels)
    
    # Sadece aktif (padding olmayan) etiketleri al
    active_mask = (labels != ignore_index)
    active_logits = logits[active_mask] # (aktif_token_sayısı, vocab_size)
    active_labels = labels[active_mask] # (aktif_token_sayısı)
    
    # PyTorch'un kendi CrossEntropyLoss'u ile kontrol
    ref_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    expected_loss = ref_loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    
    print(f"Beklenen Kayıp (PyTorch ref): {expected_loss.item():.4f}")

    assert torch.isclose(calculated_loss, expected_loss, rtol=1e-4, atol=1e-4), "Kayıp hesaplaması yanlış!"
    print("Kayıp hesaplama testi başarılı. ✅")

    # Temizlik
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")

    print("\nLanguageModelingLoss tüm testleri tamamlandı. ✅")