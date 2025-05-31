
# src/utils/logger.py

import logging
import os
import sys
from datetime import datetime

def setup_logging(log_level: str = "INFO", output_dir: str = "runs", run_name: str = "default_run"):
    """
    Proje genelinde loglama sistemini kurar.
    Loglar hem konsola hem de bir dosyaya yazılır.

    Args:
        log_level (str): Loglama seviyesi ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        output_dir (str): Ana çıktı dizini (loglar ve checkpoint'ler için).
        run_name (str): Mevcut eğitim çalıştırmasının adı. Log dosyası bu isimle oluşturulur.
    """
    # Loglama seviyesini dönüştür
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Geçersiz log seviyesi: {log_level}")

    # Log dosyasının kaydedileceği dizini oluştur
    log_dir = os.path.join(output_dir, run_name, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Log dosyasının adı (tarih ve saat ile benzersiz)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file_path = os.path.join(log_dir, f"{run_name}_{timestamp}.log")

    # Temel loglama yapılandırması
    # Mevcut root logger'ı temizle (eğer varsa)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'), # Dosyaya loglama
            logging.StreamHandler(sys.stdout) # Konsola loglama
        ]
    )
    # Varsayılan olarak root logger'ı döndürüyoruz.
    # Diğer modüllerde `log = logging.getLogger(__name__)` ile kendi logger'larını alabilirler.
    
    # Kütüphane loglarının seviyesini düşür (çok fazla gürültü yapmamaları için)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)
    logging.getLogger('tokenizers').setLevel(logging.WARNING)
    logging.getLogger('filelock').setLevel(logging.WARNING)


# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("Logger testi başlatılıyor...")

    # Test için örnek ayarlar
    test_output_dir = "test_runs"
    test_run_name = "logger_test_run"

    # Loglama sistemini kur
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    
    # Birkaç test mesajı gönder
    logger = logging.getLogger(__name__) # Test scripti için logger al
    
    logger.debug("Bu bir DEBUG mesajıdır (INFO seviyesinde görünmemeli).")
    logger.info("Bu bir INFO mesajıdır.")
    logger.warning("Bu bir WARNING mesajıdır.")
    logger.error("Bu bir ERROR mesajıdır.")
    logger.critical("Bu bir CRITICAL mesajıdır.")

    print(f"\nLoglar '{os.path.join(test_output_dir, test_run_name, 'logs')}' dizininde oluşturuldu.")
    print("Lütfen log dosyasını kontrol edin ve konsol çıktısına bakın. ✅")