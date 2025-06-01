
# src/utils/metrics.py

import os
import shutil
import torch
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

class MetricCalculator:
    """
    Dil modellerinin performansını değerlendirmek için çeşitli metrikleri hesaplar.
    Başlıca dil modelleme görevi için Perplexity ve Top-k Accuracy içerir.
    """
    def __init__(self):
        log.info("Metrik hesaplayıcı başlatıldı.")

    def calculate_perplexity(self, loss: torch.Tensor) -> float:
        """
        Kayıp (loss) değerinden perplexity'yi hesaplar.
        Perplexity = exp(loss)

        Args:
            loss (torch.Tensor): Ortalama çapraz entropi kaybı.

        Returns:
            float: Perplexity değeri.
        """
        # Kayıp değeri zaten ortalama alındığı varsayılır.
        # Eğer loss çok büyükse, math.exp hata verebilir.
        # Bu durumda float('inf') döndürmek veya bir üst limit belirlemek faydalı olabilir.
        if loss.item() > 100: # Arbitrary large value to prevent overflow
            log.warning(f"Çok yüksek kayıp değeri ({loss.item():.2f}) nedeniyle perplexity sonsuz olarak döndürüldü.")
            return float('inf')
        return math.exp(loss.item())

    def calculate_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Modelin bir sonraki token tahminlerinin doğruluğunu hesaplar.
        En yüksek olasılığa sahip tokenın doğru olup olmadığını kontrol eder.

        Args:
            logits (torch.Tensor): Modelin çıkış logits'leri (batch_size, seq_len, vocab_size).
            labels (torch.Tensor): Gerçek token ID'leri (batch_size, seq_len). Padding tokenları -100 olmalı.

        Returns:
            float: Doğruluk (accuracy) değeri.
        """
        # Padding tokenlarını yoksaymak için (labels = -100 olanlar)
        # Sadece gerçek tokenlerin tahminlerini değerlendir.
        
        # Logits'ten tahmin edilen token ID'lerini al
        # argmax, son boyutta en yüksek değeri olan indeksi verir.
        predicted_tokens = torch.argmax(logits, dim=-1) # (batch_size, seq_len)
        
        # Etiketlerdeki padding tokenlarını (genellikle -100) yoksay
        active_elements = (labels != -100).sum().item()
        
        # Doğru tahmin edilen token sayısını bul
        # predicted_tokens ve labels'ın eşit olduğu yerleri say, sadece aktif elemanlar için.
        correct_predictions = ((predicted_tokens == labels) & (labels != -100)).sum().item()
        
        if active_elements == 0:
            return 0.0 # Bölme hatasını önle
        
        return correct_predictions / active_elements

    def calculate_topk_accuracy(self, logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
        """
        Modelin bir sonraki token tahminlerinin Top-k doğruluğunu hesaplar.
        Gerçek token, modelin en yüksek k olasılığa sahip tahminleri arasında ise doğru kabul edilir.

        Args:
            logits (torch.Tensor): Modelin çıkış logits'leri (batch_size, seq_len, vocab_size).
            labels (torch.Tensor): Gerçek token ID'leri (batch_size, seq_len). Padding tokenları -100 olmalı.
            k (int): Top-k değeri.

        Returns:
            float: Top-k doğruluk değeri.
        """
        # K değerini vocab_size'dan büyük olmaması için kısıtla
        k = min(k, logits.size(-1))

        # Padding tokenlarını yoksaymak için
        active_mask = (labels != -100)
        
        # Top-k tahminleri al
        # topk, en yüksek k değeri ve indekslerini döndürür. [0] değerleri, [1] indeksleri.
        _, topk_predictions = torch.topk(logits, k=k, dim=-1) # (batch_size, seq_len, k)

        # Labels'ı genişletip Top-k tahminlerle karşılaştır
        # labels.unsqueeze(-1) -> (batch_size, seq_len, 1)
        # topk_predictions -> (batch_size, seq_len, k)
        # Eşitlik kontrolü (batch_size, seq_len, k) boyutunda bir boolean tensör döndürür.
        correct_in_topk = (labels.unsqueeze(-1) == topk_predictions) # (batch_size, seq_len, k)

        # Sadece aktif elemanlar için doğru tahminleri say
        # `correct_in_topk.any(dim=-1)` -> herhangi bir k tahmininden biri doğruysa True (batch_size, seq_len)
        # `active_mask` ile AND işlemi yap
        correct_predictions_in_topk = (correct_in_topk.any(dim=-1) & active_mask).sum().item()
        
        active_elements = active_mask.sum().item()

        if active_elements == 0:
            return 0.0
        
        return correct_predictions_in_topk / active_elements

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("MetricCalculator testi başlatılıyor...")

    # Test için gerekli loglamayı kuralım
    from src.utils.logger import setup_logging
    test_output_dir = "test_runs_metrics"
    test_run_name = "metrics_test_run"
    # Geçici olarak tüm log handler'larını kapat (Windows PermissionError'ı için)
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    metrics_calculator = MetricCalculator()

    # Ortak test parametreleri
    batch_size = 2
    seq_len = 5
    vocab_size = 100
    
    # Gerçek etiketler (labels): -100 padding tokenlarını temsil eder
    # Örnek 1: [5, 10, 15, -100, -100] (Son 2 padding)
    # Örnek 2: [20, 25, 30, 35, 40] (Padding yok)
    labels = torch.tensor([
        [5, 10, 15, -100, -100],
        [20, 25, 30, 35, 40]
    ])

    # Logits: (batch_size, seq_len, vocab_size)
    # Her bir konum için bir sonraki token olasılıkları
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Test Senaryosu 1: Perplexity hesaplama
    print("\n--- Test Senaryosu 1: Perplexity ---")
    # Gerçek kayıp değerini simüle edelim (eğitimde çıkan kayıp gibi)
    dummy_loss_tensor = torch.tensor(2.5)
    perplexity = metrics_calculator.calculate_perplexity(dummy_loss_tensor)
    print(f"Hesaplanan Perplexity (loss=2.5): {perplexity:.2f} (Beklenen: {math.exp(2.5):.2f})")
    assert abs(perplexity - math.exp(2.5)) < 0.01, "Perplexity testi başarısız!"
    
    dummy_high_loss_tensor = torch.tensor(101.0) # Çok yüksek kayıp
    perplexity_high = metrics_calculator.calculate_perplexity(dummy_high_loss_tensor)
    print(f"Hesaplanan Perplexity (loss=101.0): {perplexity_high} (Beklenen: inf)")
    assert perplexity_high == float('inf'), "Yüksek kayıp perplexity inf olmalı!"
    print("Perplexity testi başarılı. ✅")

    # Test Senaryosu 2: Accuracy (Doğruluk) hesaplama
    print("\n--- Test Senaryosu 2: Accuracy ---")
    # Logits'i, bazı doğru ve yanlış tahminler içerecek şekilde düzenleyelim.
    # Örnek 1 için:
    # 5 doğru, 10 yanlış, 15 doğru, -100, -100 (token ID'ler)
    # Yani 15'i doğru tahmin ettiğini varsayalım.
    logits[0, 0, 5] += 10.0 # İlk tokenı doğru tahmin ettir
    logits[0, 1, 12] += 10.0 # İkinci tokenı yanlış tahmin ettir (doğrusu 10)
    logits[0, 2, 15] += 10.0 # Üçüncü tokenı doğru tahmin ettir

    # Örnek 2 için:
    # 20 doğru, 25 doğru, 30 doğru, 35 yanlış, 40 doğru
    logits[1, 0, 20] += 10.0
    logits[1, 1, 25] += 10.0
    logits[1, 2, 30] += 10.0
    logits[1, 3, 37] += 10.0 # Yanlış tahmin (doğrusu 35)
    logits[1, 4, 40] += 10.0

    accuracy = metrics_calculator.calculate_accuracy(logits, labels)
    # Toplam aktif token: 3 (ilk örnekte) + 5 (ikinci örnekte) = 8
    # Doğru tahminler: 2 (ilk örnek) + 4 (ikinci örnek) = 6
    expected_accuracy = 6 / 8 # 0.75
    print(f"Hesaplanan Accuracy: {accuracy:.2f} (Beklenen: {expected_accuracy:.2f})")
    assert abs(accuracy - expected_accuracy) < 0.01, "Accuracy testi başarısız!"
    print("Accuracy testi başarılı. ✅")

    # Test Senaryosu 3: Top-k Accuracy hesaplama
    print("\n--- Test Senaryosu 3: Top-k Accuracy (k=3) ---")
    # Logits'i, bazı doğru tahminlerin Top-k içinde olacağı şekilde düzenleyelim.
    # Örnek 1, token 10: En iyi 3 tahminden biri 10 olsun.
    # Örnek 1, token 15: En iyi 3 tahminden biri 15 olsun.
    # Örnek 2, token 35: En iyi 3 tahminden biri 35 olsun.
    
    # Labels ve dummy logits'i sıfırlayalım (önceki testten etkilenmesin)
    labels_topk = torch.tensor([
        [5, 10, 15, -100, -100],
        [20, 25, 30, 35, 40]
    ])
    logits_topk = torch.randn(batch_size, seq_len, vocab_size)

    # Örnek 1: [5, 10, 15, -100, -100]
    logits_topk[0, 0, 5] = 100.0 # Token 5 kesin doğru
    logits_topk[0, 1, 10] = 90.0 # Token 10 doğru
    logits_topk[0, 1, 11] = 91.0 # Token 11 yanlış
    logits_topk[0, 1, 12] = 89.0 # Token 12 yanlış. 10, 11, 12 top-3 içinde olacak.
    logits_topk[0, 2, 15] = 95.0 # Token 15 doğru

    # Örnek 2: [20, 25, 30, 35, 40]
    logits_topk[1, 0, 20] = 100.0
    logits_topk[1, 1, 25] = 100.0
    logits_topk[1, 2, 30] = 100.0
    logits_topk[1, 3, 35] = 80.0 # Token 35 doğru
    logits_topk[1, 3, 36] = 81.0 # Token 36 yanlış
    logits_topk[1, 3, 37] = 82.0 # Token 37 yanlış. 35, 36, 37 top-3 içinde olacak.
    logits_topk[1, 4, 40] = 100.0

    topk_accuracy = metrics_calculator.calculate_topk_accuracy(logits_topk, labels_topk, k=3)
    # Toplam aktif token: 8
    # Doğru Top-3 tahminler:
    # (0,0): 5 -> doğru (1)
    # (0,1): 10 -> top3 içinde (1) (çünkü 10, 11, 12 arasından 11, 10, 12 olabilir)
    # (0,2): 15 -> doğru (1)
    # (1,0): 20 -> doğru (1)
    # (1,1): 25 -> doğru (1)
    # (1,2): 30 -> doğru (1)
    # (1,3): 35 -> top3 içinde (1)
    # (1,4): 40 -> doğru (1)
    # Toplam doğru: 8 (hepsi Top-3 içinde)
    expected_topk_accuracy = 8 / 8 # 1.0
    print(f"Hesaplanan Top-3 Accuracy: {topk_accuracy:.2f} (Beklenen: {expected_topk_accuracy:.2f})")
    assert abs(topk_accuracy - expected_topk_accuracy) < 0.01, "Top-k Accuracy testi başarısız!"
    print("Top-k Accuracy testi başarılı. ✅")

    print("\nMetricCalculator testleri tamamlandı. ✅")
    
    # Test sonrası oluşturulan test dizinlerini temizle
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)