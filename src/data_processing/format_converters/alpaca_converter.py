
# src/data_processing/format_converters/alpaca_converter.py

import json
import logging
import os
from typing import List, Dict, Any, Generator
from pathlib import Path

log = logging.getLogger(__name__)

# Alpaca formatında metin birleştirmek için bir şablon
# Genellikle Llama, Alpaca gibi modellerde bu format kullanılır.
# Örn: "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
# Veya daha basiti: "Instruction: {instruction}\nInput: {input}\nOutput: {output}"
# Eğer input yoksa: "Instruction: {instruction}\nOutput: {output}"

ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{response}"
)

ALPACA_PROMPT_NO_INPUT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)

def convert_alpaca_to_text(filepath: Path) -> Generator[Dict[str, str], None, None]:
    """
    Alpaca JSONL formatındaki bir dosyayı okur ve her bir girdiyi
    modelin işleyebileceği tek bir metin dizesine dönüştürür.

    Args:
        filepath (Path): Alpaca JSONL dosyasının yolu.

    Yields:
        Dict[str, str]: "text" anahtarıyla birleştirilmiş metin içeriğini içeren sözlük.
    """
    log.info(f"Alpaca formatındaki '{filepath}' dosyası dönüştürülüyor...")
    num_processed = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                instruction = data.get("instruction", "").strip()
                input_str = data.get("input", "").strip()
                response = data.get("output", "").strip() # 'response' yerine 'output' kullanır Alpaca

                if not instruction and not response:
                    log.warning(f"Boş veya geçersiz Alpaca girdisi atlanıyor: {line.strip()}")
                    continue

                if input_str:
                    full_text = ALPACA_PROMPT_TEMPLATE.format(
                        instruction=instruction,
                        input=input_str,
                        response=response
                    )
                else:
                    full_text = ALPACA_PROMPT_NO_INPUT_TEMPLATE.format(
                        instruction=instruction,
                        response=response
                    )
                
                num_processed += 1
                yield {"text": full_text}

            except json.JSONDecodeError as e:
                log.error(f"JSON ayrıştırma hatası: {e} - Satır atlanıyor: {line.strip()[:100]}...")
            except Exception as e:
                log.error(f"Alpaca girdisi dönüştürülürken beklenmeyen hata: {e} - Satır atlanıyor: {line.strip()[:100]}...")

    log.info(f"'{filepath}' dosyasından {num_processed} Alpaca girdisi dönüştürüldü.")


# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("AlpacaConverter testi başlatılıyor...")

    # Loglama sistemini kur (test için gerekli)
    import shutil
    from src.utils.logger import setup_logging
    test_output_dir = "test_runs_alpaca_converter"
    test_run_name = "alpaca_converter_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # Test için dummy Alpaca JSONL dosyası oluştur
    dummy_alpaca_file_path = Path(test_output_dir) / "dummy_alpaca_data.jsonl"
    with open(dummy_alpaca_file_path, "w", encoding="utf-8") as f:
        json.dump({
            "instruction": "E-posta taslağı hazırla",
            "input": "Konu: Toplantı; Alıcı: Ekip; İçerik: Proje güncellemesi.",
            "output": "Merhaba ekip, bu e-posta proje güncellemesini içermektedir."
        }, f)
        f.write("\n")
        json.dump({
            "instruction": "Kısa bir hikaye yaz",
            "input": "",
            "output": "Bir zamanlar uzak bir ülkede, küçük bir robot yaşarmış."
        }, f)
        f.write("\n")
        json.dump({
            "instruction": "Boş örnek",
            "input": "",
            "output": ""
        }, f)
        f.write("\n")
        f.write("{\"instruction\": \"Geçersiz JSON\",,,\n") # Hatalı JSON satırı

    # Dönüştürme işlemini test et
    print("\n--- Alpaca Dönüştürme Testi ---")
    converted_texts = list(convert_alpaca_to_text(dummy_alpaca_file_path))

    print(f"Dönüştürülen doküman sayısı: {len(converted_texts)}")
    assert len(converted_texts) == 2, "Dönüştürülen doküman sayısı yanlış!"
    
    # İlk dokümanı kontrol et
    expected_text_1_partial = "### Instruction:\nE-posta taslağı hazırla\n\n### Input:\nKonu: Toplantı; Alıcı: Ekip; İçerik: Proje güncellemesi."
    assert expected_text_1_partial in converted_texts[0]["text"], "İlk Alpaca metni yanlış dönüştürüldü!"
    print(f"  İlk dönüştürülen metin: '{converted_texts[0]['text'][:100]}...'")

    # İkinci dokümanı kontrol et (input olmayan)
    expected_text_2_partial = "### Instruction:\nKısa bir hikaye yaz\n\n### Response:\nBir zamanlar uzak bir ülkede, küçük bir robot yaşarmış."
    assert expected_text_2_partial in converted_texts[1]["text"], "İkinci Alpaca metni yanlış dönüştürüldü!"
    print(f"  İkinci dönüştürülen metin: '{converted_texts[1]['text'][:100]}...'")

    print("Alpaca dönüştürme testi başarılı. ✅")

    # Temizlik
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")

    print("\nAlpacaConverter tüm testleri tamamlandı. ✅")