
# src/scripts/train_tokenizer.py

import argparse
import logging
import os
import shutil
from pathlib import Path
import sys
from typing import List

# Kendi tokenizer modüllerimizi içeri aktarıyoruz
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.char_tokenizer import CharTokenizer
from src.tokenizer.special_tokens import UNK_TOKEN # UNK tokenı için
from src.utils.logger import setup_logging
from src.config.base_config import BaseConfig # Vocab size gibi config'ler için

log = logging.getLogger(__name__)

def main(args):
    # Loglama sistemini kur
    project_output_root = Path(args.output_dir) if args.output_dir else Path("runs")
    run_name = f"tokenizer_train_{args.tokenizer_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    setup_logging(log_level=args.log_level, output_dir=str(project_output_root), run_name=run_name)
    
    log.info("Tokenizer eğitim betiği başlatıldı.")
    log.info(f"Girdi dosyaları/dizini: {args.input_paths}")
    log.info(f"Çıktı dizini: {args.tokenizer_output_dir}")
    log.info(f"Tokenizer tipi: {args.tokenizer_type}")
    log.info(f"Kelime haznesi boyutu: {args.vocab_size}")

    tokenizer_instance = None
    tokenizer_output_path = Path(args.tokenizer_output_dir)
    tokenizer_output_path.mkdir(parents=True, exist_ok=True) # Tokenizer kayıt dizinini oluştur

    if args.tokenizer_type == "bpe":
        # BPE Tokenizer
        tokenizer_instance = BPETokenizer(vocab_size=args.vocab_size, unk_token=UNK_TOKEN)
        
        # Eğitim dosyalarını hazırla
        files_to_train: List[str] = []
        for path_str in args.input_paths:
            current_path = Path(path_str)
            if current_path.is_file():
                files_to_train.append(str(current_path))
            elif current_path.is_dir():
                # Dizin içindeki tüm metin dosyalarını topla (örn. .txt, .jsonl uzantılı)
                for f in current_path.rglob("*"):
                    if f.is_file() and f.suffix in [".txt", ".jsonl", ".parquet"]: # Eğiteceğimiz metin dosyaları
                        files_to_train.append(str(f))
            else:
                log.warning(f"Geçersiz girdi yolu atlanıyor: {path_str}")

        if not files_to_train:
            log.error("Eğitim için hiçbir uygun dosya bulunamadı. Lütfen --input_paths'ı kontrol edin.")
            sys.exit(1)

        log.info(f"BPE eğitimi için bulununan dosyalar: {len(files_to_train)}")
        tokenizer_instance.train(files=files_to_train, save_path=str(tokenizer_output_path))
        
    elif args.tokenizer_type == "char":
        # Character Tokenizer
        # Character tokenizer'ı eğitmek için öncelikle tüm benzersiz karakterleri toplamamız gerekir.
        # Bu, büyük veri setleri için zaman alıcı olabilir.
        log.info("Karakter tokenizer eğitimi başlatılıyor. Tüm benzersiz karakterler toplanacak.")
        all_chars = set()
        for path_str in args.input_paths:
            current_path = Path(path_str)
            if current_path.is_file():
                with open(current_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_chars.update(line.strip())
            elif current_path.is_dir():
                for f in current_path.rglob("*"):
                    if f.is_file() and f.suffix in [".txt", ".jsonl", ".parquet"]:
                        with open(f, 'r', encoding='utf-8') as file_obj:
                            for line in file_obj:
                                all_chars.update(line.strip())
            else:
                log.warning(f"Geçersiz girdi yolu atlanıyor: {path_str}")
        
        # Özel tokenları da karakter setine ekle (zaten init'te ekleniyor)
        # all_chars.update(special_tokens.all_special_tokens_as_list)
        
        tokenizer_instance = CharTokenizer(chars=list(all_chars), unk_token=UNK_TOKEN)
        tokenizer_instance.save_vocabulary(str(tokenizer_output_path))
        log.info(f"Karakter tokenizer başarıyla oluşturuldu ve '{tokenizer_output_path}' konumuna kaydedildi. Kelime haznesi boyutu: {tokenizer_instance.vocabulary_size}")

    else:
        log.error(f"Desteklenmeyen tokenizer tipi: {args.tokenizer_type}. 'bpe' veya 'char' olmalı.")
        sys.exit(1)

    log.info("Tokenizer eğitim betiği tamamlandı.")


if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser(description="EchoModel için tokenizer eğitim betiği.")
    parser.add_argument(
        "--input_paths",
        nargs="+", # Birden fazla yol alabilir
        type=str,
        required=True,
        help="Tokenizer eğitimi için kullanılacak metin dosyalarının veya dizinlerin yolu (boşlukla ayrılmış)."
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="bpe",
        choices=["bpe", "char"],
        help="Eğitilecek tokenizer tipi ('bpe' veya 'char')."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=BaseConfig().vocab_size, # BaseConfig'ten varsayılan vocab_size'ı al
        help="Tokenizer kelime haznesinin boyutu."
    )
    parser.add_argument(
        "--tokenizer_output_dir",
        type=str,
        default="tokenizer_assets",
        help="Eğitilen tokenizer dosyalarının kaydedileceği dizin."
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

    args = parser.parse_args()
    main(args)

    # Test fonksiyonunu burada çalıştırmayalım, bu bir script.
    # Testleri ayrı bir test dosyasında tutacağız.