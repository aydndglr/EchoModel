
# src/data_processing/dataset_preprocessor.py

import os
import re
import json
import logging
import shutil
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import hashlib # Deduplikasyon için

log = logging.getLogger(__name__)

class DatasetPreprocessor:
    """
    Ham veri setlerini temizleme, filtreleme ve standartlaştırılmış bir formata dönüştürme sınıfı.
    Çeşitli veri kaynaklarından (metin dosyaları, JSON, vb.) gelen verileri işleyebilir.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): Veri işleme ayarlarını içeren bir sözlük.
                                     Örn: {"min_len": 100, "max_len": 10000, "remove_html": True, "deduplicate": True}
        """
        self.config = config
        log.info(f"DatasetPreprocessor başlatıldı. Konfigürasyon: {self.config}")

        self.min_len = config.get("min_len", 50)
        self.max_len = config.get("max_len", 20000)
        self.remove_html = config.get("remove_html", True)
        self.normalize_whitespace = config.get("normalize_whitespace", True)
        self.remove_empty_lines = config.get("remove_empty_lines", True)
        self.deduplicate = config.get("deduplicate", True)
        self.deduplication_threshold = config.get("deduplication_threshold", 0.8) # Jaccard benzerliği için
        self.language_filter = config.get("language_filter", None) # 'en' gibi
        
        self.processed_texts = [] # İşlenmiş metinleri geçici olarak tutar
        self.seen_hashes = set() # Deduplikasyon için hash'leri tutar

    def _clean_text(self, text: str) -> str:
        """Metni temel temizleme işlemlerinden geçirir."""
        if self.remove_html:
            text = re.sub(r"<[^>]+>", "", text) # HTML etiketlerini kaldır
        
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip() # Birden fazla boşluğu tek boşluğa indir, baştaki/sondaki boşlukları kaldır
        
        if self.remove_empty_lines:
            text = os.linesep.join([s for s in text.split(os.linesep) if s.strip()]) # Boş satırları kaldır

        return text

    def _filter_text(self, text: str) -> bool:
        """Metni uzunluk ve dil filtrelerine göre kontrol eder."""
        if not (self.min_len <= len(text) <= self.max_len):
            return False
        
        # Dil filtresi (basit bir örnek, daha gelişmiş dil tespiti gerekebilir)
        # OLMO'da daha karmaşık dil tespiti (örn. fastText) kullanılır. [cite: `uploaded:OLMo-main/olmo/data/util.py`]
        # Biz şimdilik basit bir regex veya kütüphane kullanmadan placeholder bırakalım.
        # if self.language_filter:
        #    # Burada dil tespiti kütüphanesi kullanılabilir (örn. langdetect)
        #    # if detect(text) != self.language_filter: return False
        #    pass # Placeholder
            
        return True

    def _deduplicate_text(self, text: str) -> bool:
        """Metni hash tabanlı deduplikasyon ile kontrol eder."""
        if not self.deduplicate:
            return True
        
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in self.seen_hashes:
            return False
        
        self.seen_hashes.add(text_hash)
        return True

    def process_document(self, document_content: str) -> Optional[str]:
        """
        Tek bir dokümanı işler (temizler, filtreler, deduplike eder).
        """
        cleaned_text = self._clean_text(document_content)
        if not self._filter_text(cleaned_text):
            return None
        if not self._deduplicate_text(cleaned_text):
            return None
        return cleaned_text

    def process_file(self, input_filepath: Path, output_filepath: Path, converter_func: Optional[callable] = None):
        """
        Belirli bir ham veri dosyasını işler ve işlenmiş veriyi kaydeder.

        Args:
            input_filepath (Path): İşlenecek ham dosyanın yolu.
            output_filepath (Path): İşlenmiş verinin kaydedileceği dosyanın yolu (örn. .parquet).
            converter_func (Optional[callable]): Dosya formatını ayrıştırmak için kullanılacak fonksiyon.
                                                 (örn. format_converters modülünden bir fonksiyon)
        """
        log.info(f"'{input_filepath}' dosyası işleniyor...")
        processed_records = []

        if converter_func:
            # Converter fonksiyonu, ham dosyayı okuyup metin listesi/jeneratörü döndürmeli
            documents = converter_func(input_filepath)
        elif input_filepath.suffix == ".txt":
            with open(input_filepath, 'r', encoding='utf-8') as f:
                documents = f.readlines() # Her satırı bir doküman olarak al
        elif input_filepath.suffix == ".jsonl":
            with open(input_filepath, 'r', encoding='utf-8') as f:
                documents = [json.loads(line).get("text", "") for line in f if line.strip()]
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı veya converter_func eksik: {input_filepath.suffix}")

        for doc in documents:
            if isinstance(doc, dict) and "text" in doc: # Yapılandırılmış dokümanlar için
                text_content = doc["text"]
            elif isinstance(doc, str):
                text_content = doc
            else:
                log.warning(f"Bilinmeyen doküman formatı atlanıyor: {type(doc)}")
                continue

            processed_text = self.process_document(text_content)
            if processed_text:
                processed_records.append({"text": processed_text})
        
        # İşlenmiş veriyi kaydet (örn. Parquet formatında)
        # Pandas ve pyarrow gerektirir. requirements.txt'ye eklemeliyiz.
        try:
            import pandas as pd
            df = pd.DataFrame(processed_records)
            df.to_parquet(output_filepath, index=False)
            log.info(f"İşlenmiş veri '{output_filepath}' konumuna kaydedildi. Toplam doküman: {len(df)}")
        except ImportError:
            log.warning("Pandas veya PyArrow yüklü değil. İşlenmiş veri Parquet olarak kaydedilemedi.")
            # Alternatif olarak JSONL olarak kaydedebiliriz
            with open(output_filepath.with_suffix('.jsonl'), 'w', encoding='utf-8') as f:
                for record in processed_records:
                    f.write(json.dumps(record) + '\n')
            log.info(f"İşlenmiş veri '{output_filepath.with_suffix('.jsonl')}' konumuna JSONL olarak kaydedildi. Toplam doküman: {len(processed_records)}")


    def process_dataset(self, input_dir: str, output_dir: str, converter_map: Optional[Dict[str, callable]] = None):
        """
        Bir dizindeki tüm ham veri dosyalarını işler ve sonuçları kaydeder.

        Args:
            input_dir (str): Ham veri dosyalarının bulunduğu dizin.
            output_dir (str): İşlenmiş verilerin kaydedileceği dizin.
            converter_map (Optional[Dict[str, callable]]): Dosya uzantılarına göre dönüştürücü fonksiyonları.
                                                          Örn: {".xml.bz2": wikipedia_parser.parse_xml_bz2}
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        log.info(f"'{input_dir}' dizinindeki veri seti işleniyor...")
        
        for filepath in input_path.rglob("*"): # Alt dizinlerdeki tüm dosyaları gez
            if filepath.is_file():
                converter_func = None
                if converter_map:
                    for suffix, func in converter_map.items():
                        if str(filepath).endswith(suffix):
                            converter_func = func
                            break
                
                # Çıktı dosya adını belirle (input_filepath'in adını koru)
                output_file_name = filepath.stem # Uzantısız dosya adı
                if output_file_name.endswith(".xml"): # .xml.bz2 gibi durumlar için
                    output_file_name = Path(output_file_name).stem
                
                # Parquet veya JSONL olarak kaydedeceğimiz için uzantıyı ayarla
                output_filepath = output_path / f"{output_file_name}.parquet" 
                
                try:
                    self.process_file(filepath, output_filepath, converter_func)
                except Exception as e:
                    log.error(f"'{filepath}' işlenirken hata: {e}", exc_info=True)
        
        log.info(f"Veri seti işleme tamamlandı. İşlenmiş veriler '{output_dir}' konumunda.")

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("DatasetPreprocessor testi başlatılıyor...")

    # Loglama sistemini kur (test için gerekli)
    from src.utils.logger import setup_logging
    test_output_dir = "test_runs_data_processing"
    test_run_name = "preprocessor_test_run"
    
    # Mevcut test dizinlerini temizle
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    # Geçici olarak tüm log handler'larını kapat (Windows PermissionError'ı için)
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # Test için ham veri dizini ve dosyaları oluştur
    raw_data_dir = os.path.join(test_output_dir, "raw_data")
    processed_data_dir = os.path.join(test_output_dir, "processed_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    # Test TXT dosyası
    with open(os.path.join(raw_data_dir, "doc1.txt"), "w", encoding="utf-8") as f:
        f.write("Bu bir test dokümanıdır.   Çok fazla boşluk var.  \n")
        f.write("İkinci satır.  \n\n")
        f.write("  Üçüncü satır.  <p>HTML içeriği</p>\n")
        f.write("Kısa.\n") # Filtrelenecek
        f.write("Bu bir tekrar eden cümledir. Bu bir tekrar eden cümledir.\n") # Deduplike edilecek

    # Test JSONL dosyası (Alpaca benzeri)
    with open(os.path.join(raw_data_dir, "alpaca_data.jsonl"), "w", encoding="utf-8") as f:
        json.dump({"instruction": "Merhaba", "input": "", "text": "Bu bir JSONL dokümanıdır. İlk satır.\nİkinci satır."}, f)
        f.write("\n")
        json.dump({"instruction": "Tekrar", "input": "", "text": "Bu bir JSONL dokümanıdır. İlk satır.\nİkinci satır."}, f) # Deduplike edilecek
        f.write("\n")
        json.dump({"instruction": "Kısa", "input": "", "text": "Kısa jsonl."}, f) # Filtrelenecek
        f.write("\n")

    # Preprocessor konfigürasyonu
    preprocessor_config = {
        "min_len": 20,
        "max_len": 1000,
        "remove_html": True,
        "normalize_whitespace": True,
        "remove_empty_lines": True,
        "deduplicate": True
    }

    preprocessor = DatasetPreprocessor(config=preprocessor_config)

    print("\n--- Veri Seti İşleme Testi ---")
    try:
        # converter_map'i boş bırakalım, dosya uzantılarına göre otomatik işlesin
        preprocessor.process_dataset(raw_data_dir, processed_data_dir)
        
        # İşlenmiş dosyaların oluştuğunu kontrol et
        processed_files = list(Path(processed_data_dir).glob("*"))
        print(f"Oluşturulan işlenmiş dosyalar: {[f.name for f in processed_files]}")
        assert len(processed_files) > 0, "Hiç işlenmiş dosya oluşturulmadı!"
        
        # Parquet veya JSONL dosyasını oku ve içeriği kontrol et
        processed_txt_file = Path(processed_data_dir) / "doc1.parquet" # veya .jsonl
        if not processed_txt_file.exists():
            processed_txt_file = Path(processed_data_dir) / "doc1.jsonl"
        
        if processed_txt_file.exists():
            if processed_txt_file.suffix == ".parquet":
                import pandas as pd
                df = pd.read_parquet(processed_txt_file)
                # Beklenen: "Kısa." ve tekrar eden cümle deduplike edildiği için 2 doküman olmalı
                assert len(df) == 2, f"Doc1.txt'den beklenen doküman sayısı yanlış! Beklenen 2, bulunan {len(df)}"
                assert "HTML içeriği" not in df.iloc[0]["text"], "HTML kaldırılmadı!"
                assert "  " not in df.iloc[0]["text"], "Boşluklar normalleştirilmedi!"
                assert len(df.iloc[0]["text"]) >= 20, "Minimum uzunluk filtresi çalışmadı!"
                print(f"  İşlenmiş doc1.txt içeriği (ilk): '{df.iloc[0]['text']}'")
                print(f"  İşlenmiş doc1.txt içeriği (ikinci): '{df.iloc[1]['text']}'")
            elif processed_txt_file.suffix == ".jsonl":
                with open(processed_txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    assert len(lines) == 2, f"Doc1.txt'den beklenen doküman sayısı yanlış! Beklenen 2, bulunan {len(lines)}"
                    # JSONL içeriği kontrolü
                    doc_content = json.loads(lines[0])["text"]
                    assert "HTML içeriği" not in doc_content, "HTML kaldırılmadı!"
                    assert "  " not in doc_content, "Boşluklar normalleştirilmedi!"
                    assert len(doc_content) >= 20, "Minimum uzunluk filtresi çalışmadı!"
                    print(f"  İşlenmiş doc1.txt içeriği (ilk): '{json.loads(lines[0])['text']}'")
                    print(f"  İşlenmiş doc1.txt içeriği (ikinci): '{json.loads(lines[1])['text']}'")

        processed_jsonl_file = Path(processed_data_dir) / "alpaca_data.parquet" # veya .jsonl
        if not processed_jsonl_file.exists():
            processed_jsonl_file = Path(processed_data_dir) / "alpaca_data.jsonl"

        if processed_jsonl_file.exists():
            if processed_jsonl_file.suffix == ".parquet":
                import pandas as pd
                df_jsonl = pd.read_parquet(processed_jsonl_file)
                # Beklenen: Kısa jsonl filtresi ve deduplikasyon sonrası 1 doküman kalmalı
                assert len(df_jsonl) == 1, f"Alpaca_data.jsonl'den beklenen doküman sayısı yanlış! Beklenen 1, bulunan {len(df_jsonl)}"
                print(f"  İşlenmiş alpaca_data.jsonl içeriği: '{df_jsonl.iloc[0]['text']}'")
            elif processed_jsonl_file.suffix == ".jsonl":
                with open(processed_jsonl_file, 'r', encoding='utf-8') as f:
                    lines_jsonl = f.readlines()
                    assert len(lines_jsonl) == 1, f"Alpaca_data.jsonl'den beklenen doküman sayısı yanlış! Beklenen 1, bulunan {len(lines_jsonl)}"
                    print(f"  İşlenmiş alpaca_data.jsonl içeriği: '{json.loads(lines_jsonl[0])['text']}'")


        print("Veri seti işleme testi başarılı. ✅")

    except Exception as e:
        print(f"Veri seti işleme testi BAŞARISIZ: {e} ❌")
    finally:
        # Oluşturulan test dizinini temizle
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)
            print(f"Temizlik yapıldı: '{test_output_dir}' silindi.")

    print("\nDatasetPreprocessor tüm testleri tamamlandı. ✅")