
# src/scripts/preprocess_data.py

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
import sys
from typing import Dict, Any, Callable

# Kendi modüllerimizi içeri aktarıyoruz
from src.utils.logger import setup_logging
from src.data_processing.dataset_preprocessor import DatasetPreprocessor
from src.data_processing.format_converters.alpaca_converter import convert_alpaca_to_text
from src.data_processing.format_converters.hf_dataset_converter import convert_hf_dataset_to_text
from src.data_processing.format_converters.generic_parser import parse_text_file, parse_json_file, parse_xml_file

log = logging.getLogger(__name__)

def main(args):
    # Loglama sistemini kur
    # Output dizinini burada belirleyelim veya argümanlardan alalım
    project_output_root = Path(args.output_dir) if args.output_dir else Path("runs")
    run_name = f"data_prep_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    setup_logging(log_level=args.log_level, output_dir=str(project_output_root), run_name=run_name)
    
    log.info("Veri ön işleme betiği başlatıldı.")
    log.info(f"Girdi dizini: {args.input_dir}")
    log.info(f"Çıktı dizini: {args.processed_output_dir}")
    log.info(f"Ön işleme ayarları: {args.preprocessor_config}")

    # Ön işleme yapılandırmasını oluştur
    preprocessor_config = json.loads(args.preprocessor_config) if isinstance(args.preprocessor_config, str) else args.preprocessor_config
    
    # DatasetPreprocessor'ı başlat
    preprocessor = DatasetPreprocessor(config=preprocessor_config)

    # Desteklenen dönüştürücüler haritası
    # Bu harita, DatasetPreprocessor'a hangi dosya uzantısı için hangi fonksiyonun kullanılacağını söyler.
    converter_map: Dict[str, Callable[[Path, Any], Any]] = {
        ".jsonl": parse_json_file, # Hem Alpaca hem de genel JSONL için kullanılabilir
        ".json": parse_json_file,
        ".txt": parse_text_file,
        ".gz": lambda p: parse_text_file(p), # Gzip metin için
        ".bz2": lambda p: parse_xml_file(p, text_tag="text"), # Wikipedia gibi XML.BZ2 için
        ".xml": lambda p: parse_xml_file(p, text_tag="text"), # Genel XML için
        # Özel formatlar için:
        # ".alpaca.jsonl": convert_alpaca_to_text, # Eğer Alpaca'nın kendi özel dönüştürücüsü varsa
        # ".hf_dataset": convert_hf_dataset_to_text # HF datasetleri doğrudan dosya değil, farklı bir yükleme mekanizması gerekir.
                                                    # HF datasetleri için `load_dataset` direkt python kodu içinde çağrılmalı.
                                                    # `convert_hf_dataset_to_text` argümanı filepath değil, name/config_name bekliyor.
                                                    # Bu nedenle `DatasetPreprocessor`'ın `process_dataset` metodunun bu converter'ları
                                                    # nasıl kullanacağı biraz daha düşünülmeli.
                                                    # Şimdilik, `process_dataset` sadece Path alabiliyor.
                                                    # Bu converters, `DatasetPreprocessor`'ın `process_file` içinden çağrılmalı.
    }

    # OLMO'nun `prepare_memmap_dataset.py` script'inde data_mix ve data_config'ler kullanılır.
    # Biz de benzer şekilde, giriş dizinindeki dosya uzantılarına göre uygun dönüştürücüleri kullanacağız.

    # Ham veri dizinini işlenmiş veri dizinine dönüştür
    # `DatasetPreprocessor.process_dataset` metodunu kullanıyoruz.
    try:
        preprocessor.process_dataset(args.input_dir, args.processed_output_dir, converter_map=converter_map)
        log.info("Veri ön işleme tamamlandı.")
    except Exception as e:
        log.error(f"Veri ön işleme sırasında hata oluştu: {e}", exc_info=True)
        sys.exit(1) # Hata durumunda çık

    # İsteğe bağlı: İşlenmiş veriyi eğitim/doğrulama/test setlerine ayır
    if args.split_data:
        from src.data_processing.data_splitter import DataSplitter
        splitter = DataSplitter(seed=args.seed)
        
        # İşlenmiş verinin tek bir dosyada olduğunu varsayalım (örn. tüm küçük parçaların birleşimi)
        # Veya process_dataset metodumuz birden çok parçayı kaydettiği için,
        # bu split'i o parçaları toplayıp yapmak gerekir.
        # En basit yol, tüm işlenmiş veriyi tek bir dosyada birleştirmek (şimdilik yapmıyoruz)
        # veya split işlemini direkt raw_data_dir üzerinden yapmak.
        
        # Varsayalım ki process_dataset, tek bir büyük output dosyası oluşturdu
        # Bu senaryoda output_filepath'i manuel olarak geçmemiz gerekecek.
        # Bu, `process_dataset` metodunun nasıl tasarlandığına bağlı.
        # Şu anki `process_dataset`, input_dir'deki her dosyayı ayrı ayrı output_dir'a .parquet olarak kaydediyor.
        # Dolayısıyla, split_data burada her bir processed .parquet dosyasını bölmeye çalışacak.
        # Bu senaryo için `process_dataset`'in sonuna tek bir birleştirme adımı eklemek daha iyi olur.
        # Şimdilik, sadece `data_splitter`'ın nasıl çağrılabileceğini gösterelim.
        
        # Basitlik için, her bir işlem gören dosyanın sonunda ayrıştırma yapalım
        # Bu, main fonksiyonu için daha karmaşık olurdu.
        # Genellikle, tüm ön işleme bittikten sonra, işlenmiş veriler bir araya getirilir
        # ve o büyük birleştirilmiş dosya üzerinde split işlemi yapılır.
        
        # Bu yüzden, `data_splitter.py`'nin `split_dataset`'i tek bir dosya bekler.
        # `preprocess_data.py`'nin çıktısı birden fazla dosya olabilir.
        # Bu durumda ya `preprocess_data.py`'nin sonunda birleştirme yapmalıyız
        # ya da `data_splitter`'a bir dizin verip içindeki tüm dosyaları işlemesini sağlamalıyız.
        # Şimdilik, bu `split_data` argümanını sadece placeholder olarak tutalım ve
        # `DatasetPreprocessor`'ın çıktısı olan dosyaları daha sonra manuel olarak bölmeyi önerelim.
        log.warning("Veri ayırma (split_data) işlevi henüz tam olarak entegre edilmedi.")
        log.warning("İşlenmiş verileri 'data/processed' altına kaydettikten sonra manuel olarak ayrıştırın.")
        # Örneğin: python src/scripts/split_processed_data.py gibi ayrı bir script
        
    log.info("Veri ön işleme betiği tamamlandı.")

# Bu betiği doğrudan çalıştırırken argümanları yakala
if __name__ == "__main__":
    from datetime import datetime # datetime'ı burada da import et

    parser = argparse.ArgumentParser(description="EchoModel için veri ön işleme betiği.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Ham veri dosyalarının bulunduğu dizin (örn. data/raw/my_dataset)."
    )
    parser.add_argument(
        "--processed_output_dir",
        type=str,
        default="data/processed",
        help="İşlenmiş verilerin kaydedileceği dizin (örn. data/processed)."
    )
    parser.add_argument(
        "--preprocessor_config",
        type=str, # JSON string olarak veya YAML dosyası yolu olarak
        default='{"min_len": 50, "max_len": 20000, "remove_html": true, "normalize_whitespace": true, "remove_empty_lines": true, "deduplicate": true}',
        help="DatasetPreprocessor için JSON string olarak yapılandırma ayarları."
    )
    parser.add_argument(
        "--output_dir", # Genel çıktı kök dizini (loglar için)
        type=str,
        default="runs",
        help="Logların ve geçici dosyaların kaydedileceği ana çıktı dizini."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Loglama seviyesi."
    )
    parser.add_argument(
        "--split_data",
        action="store_true", # Argüman varsa True, yoksa False
        help="İşlenmiş veriyi eğitim, doğrulama ve test setlerine ayır."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Veri ayırma için rastgelelik tohumu."
    )

    args = parser.parse_args()
    main(args)

    # Test fonksiyonunu burada çalıştırmayalım, bu bir script.
    # Testleri ayrı bir test dosyasında tutacağız.
    # `tests/data_processing_test.py` gibi bir dosya daha sonra oluşturulabilir.