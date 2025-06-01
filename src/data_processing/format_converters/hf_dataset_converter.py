# src/data_processing/format_converters/hf_dataset_converter.py

import logging
import os
from typing import List, Dict, Any, Generator, Optional, Union
from datasets import load_dataset, Dataset # Hugging Face datasets kütüphanesini içeri aktarıyoruz
from pathlib import Path

log = logging.getLogger(__name__)

def convert_hf_dataset_to_text(
    dataset_name: str,
    dataset_config_name: Optional[str] = None,
    split: str = "train",
    text_column: str = "text"
) -> Generator[Dict[str, str], None, None]:
    """
    Hugging Face datasets kütüphanesinden bir veri setini yükler ve her bir örneği
    standart metin formatımıza dönüştürür.

    Args:
        dataset_name (str): Hugging Face Hub'daki veri setinin adı (örn. "wikitext", "c4").
        dataset_config_name (Optional[str]): Veri setinin yapılandırma adı (örn. "wikitext-103-raw-v1").
        split (str): Yüklenecek veri setinin bölümü (örn. "train", "validation", "test").
        text_column (str): Metin içeriğini içeren sütunun adı.

    Yields:
        Dict[str, str]: "text" anahtarıyla metin içeriğini içeren sözlük.
    """
    log.info(f"Hugging Face veri seti yükleniyor: {dataset_name} ({dataset_config_name if dataset_config_name else 'default'}) - Bölüm: {split}")
    
    try:
        # Veri setini yükle
        dataset = load_dataset(dataset_name, dataset_config_name, split=split)
        log.info(f"Veri seti yüklendi. Toplam örnek: {len(dataset)}")
    except Exception as e:
        log.error(f"Hugging Face veri seti '{dataset_name}' yüklenirken hata oluştu: {e}", exc_info=True)
        return # Generator'dan çık

    num_processed = 0
    for example in dataset:
        if text_column in example and example[text_column] is not None:
            text_content = str(example[text_column]).strip()
            if text_content:
                num_processed += 1
                yield {"text": text_content}
        else:
            log.warning(f"Örnekte '{text_column}' sütunu bulunamadı veya boş. Örnek atlanıyor: {example.keys()}")
    
    log.info(f"Hugging Face veri setinden {num_processed} örnek dönüştürüldü.")


# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("HFDatasetConverter testi başlatılıyor...")

    # Loglama sistemini kur (test için gerekli)
    import shutil
    from src.utils.logger import setup_logging
    test_output_dir = "test_runs_hf_converter"
    test_run_name = "hf_converter_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # Test Senaryosu 1: Küçük bir HF veri setini dönüştür (örneğin "glue", "mrpc" gibi)
    # requirements.txt'ye 'datasets' kütüphanesini eklemeyi unutmayın!
    print("\n--- HF Dataset Dönüştürme Testi (glue/mrpc) ---")
    try:
        # 'glue' veri seti 'mrpc' yapılandırması
        converted_samples = list(convert_hf_dataset_to_text(
            dataset_name="glue",
            dataset_config_name="mrpc",
            split="validation",
            text_column="sentence1" # 'sentence1' veya 'sentence2' olabilir
        ))
        
        print(f"Dönüştürülen örnek sayısı: {len(converted_samples)}")
        assert len(converted_samples) > 0, "Hiç örnek dönüştürülemedi!"
        assert "text" in converted_samples[0], "Dönüştürülen örnekte 'text' anahtarı yok!"
        print(f"  İlk örnek metin: '{converted_samples[0]['text'][:100]}...'")
        print("HF Dataset dönüştürme testi başarılı. ✅")

    except Exception as e:
        print(f"HF Dataset dönüştürme testi BAŞARISIZ: {e} ❌")
        log.error(f"HF Dataset dönüştürme testi BAŞARISIZ: {e}", exc_info=True)
    finally:
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)
            log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")

    print("\nHFDatasetConverter tüm testleri tamamlandı. ✅")