
# src/main.py

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import torch

# Kendi modüllerimizi içeri aktarıyoruz
from src.utils.logger import setup_logging
from src.config.base_config import BaseConfig
from src.config.model_config import ModelConfig

# Script'ler ve Trainer, Generator, Predictor sınıfları
from src.scripts.preprocess_data import main as preprocess_data_script_main
from src.scripts.train_tokenizer import main as train_tokenizer_script_main
from src.training.trainer import Trainer
from src.inference.generator import TextGenerator
from src.inference.predictor import ModelPredictor

# Diğer bağımlılıklar (sadece ihtiyaç duyulduğunda import edilecek)
# from src.dataset.custom_data_loader import CustomDataLoader
# from src.model.echo_transformer import EchoTransformer
# from src.tokenizer.bpe_tokenizer import BPETokenizer
# from src.tokenizer.char_tokenizer import CharTokenizer
# from src.utils.device_manager import DeviceManager
# from src.utils.checkpoint_manager import CheckpointManager
# from src.utils.metrics import MetricCalculator


log = logging.getLogger(__name__)

def run_train(args):
    """Eğitim sürecini başlatır."""
    log.info("Eğitim görevi başlatılıyor...")

    # Konfigürasyonları yükle
    base_cfg = BaseConfig()
    model_cfg = ModelConfig()

    # Data Loader'ları yükle (örnek olarak, gerçek uygulamada config'ten gelecek)
    from src.dataset.custom_data_loader import CustomDataLoader
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.tokenizer.special_tokens import PAD_TOKEN

    # Bu kısım gerçek bir veri yolu ve tokenizer yolu gerektirir.
    # Şimdilik placeholder olarak dummy dosyalar ve varsayılan config'ler kullanacağız.
    # Gerçek kullanımda, bu yollar CLI argümanlarından veya bir config dosyasından gelmeli.
    
    # Dummy tokenizer ve veri yolu (test veya ilk çalıştırma için)
    dummy_tokenizer_path = Path(base_cfg.output_dir) / "temp_tokenizer_assets"
    dummy_train_data_path = Path(base_cfg.output_dir) / "temp_train_data.jsonl"
    dummy_eval_data_path = Path(base_cfg.output_dir) / "temp_eval_data.jsonl"

    # Hızlı test için tokenizer ve veri oluştur (gerçek kullanımda preprocess/train_tokenizer çalıştırılmalı)
    if not dummy_tokenizer_path.exists() or not dummy_train_data_path.exists():
        log.warning("Dummy veri veya tokenizer bulunamadı. Lütfen 'preprocess_data' ve 'train_tokenizer' görevlerini çalıştırın.")
        log.warning("Şimdilik geçici dummy veriler oluşturulacak.")
        os.makedirs(dummy_tokenizer_path, exist_ok=True)
        with open(dummy_train_data_path, "w") as f: f.write("dummy data\n")
        with open(dummy_eval_data_path, "w") as f: f.write("dummy data\n")
        
        # Basit bir tokenizer eğit (BPE)
        temp_tokenizer_train_file = Path(base_cfg.output_dir) / "temp_tokenizer_train.txt"
        with open(temp_tokenizer_train_file, "w") as f:
            f.write("a b c d e f g h i j k l m n o p q r s t u v w x y z. A B C D E F G H I J K L M N O P Q R S T U V W X Y Z\n")
        temp_bpe_tokenizer = BPETokenizer(vocab_size=base_cfg.vocab_size)
        temp_bpe_tokenizer.train(files=[str(temp_tokenizer_train_file)], save_path=str(dummy_tokenizer_path))
        # Ensure pad_token_id is set
        if getattr(temp_bpe_tokenizer, 'pad_token_id', None) is None:
            temp_bpe_tokenizer.pad_token_id = temp_bpe_tokenizer.token_to_id(PAD_TOKEN)

    # Dataloader'ları başlat
    train_dataloader = CustomDataLoader(
        data_filepath=dummy_train_data_path,
        tokenizer_path=str(dummy_tokenizer_path),
        max_seq_len=base_cfg.max_seq_len,
        batch_size=base_cfg.train_batch_size,
        tokenizer_type="bpe", # Veya args.tokenizer_type'tan al
        num_workers=base_cfg.num_workers,
        shuffle=True,
        pin_memory=True
    ).get_dataloader()

    eval_dataloader = CustomDataLoader(
        data_filepath=dummy_eval_data_path,
        tokenizer_path=str(dummy_tokenizer_path),
        max_seq_len=base_cfg.max_seq_len,
        batch_size=base_cfg.eval_batch_size,
        tokenizer_type="bpe", # Veya args.tokenizer_type'tan al
        num_workers=base_cfg.num_workers,
        shuffle=False,
        pin_memory=True
    ).get_dataloader()

    # Modeli oluştur
    from src.model.echo_transformer import EchoTransformer
    model = EchoTransformer(base_config=base_cfg, model_config=model_cfg)

    # Yardımcı sınıfları oluştur
    from src.utils.device_manager import DeviceManager
    from src.utils.checkpoint_manager import CheckpointManager
    from src.utils.metrics import MetricCalculator

    device_manager = DeviceManager(device_name=base_cfg.device)
    checkpoint_manager = CheckpointManager(save_dir=os.path.join(base_cfg.output_dir, "checkpoints"))
    metric_calculator = MetricCalculator()

    # Trainer'ı başlat ve eğitimi başlat
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        base_config=base_cfg,
        model_config=model_cfg,
        device_manager=device_manager,
        checkpoint_manager=checkpoint_manager,
        metric_calculator=metric_calculator
    )
    trainer.train()

    log.info("Eğitim görevi tamamlandı.")


def run_generate(args):
    """Metin üretimi yapar."""
    log.info("Metin üretimi görevi başlatılıyor...")

    # Model ve tokenizer'ı yükle
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.tokenizer.char_tokenizer import CharTokenizer
    from src.utils.device_manager import DeviceManager

    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        log.error(f"Tokenizer dosyası/dizini bulunamadı: {tokenizer_path}")
        sys.exit(1)

    tokenizer_instance = None
    if args.tokenizer_type == "bpe":
        tokenizer_instance = BPETokenizer.from_pretrained(str(tokenizer_path))
    elif args.tokenizer_type == "char":
        tokenizer_instance = CharTokenizer.from_pretrained(str(tokenizer_path))
    else:
        log.error(f"Desteklenmeyen tokenizer tipi: {args.tokenizer_type}")
        sys.exit(1)

    device_manager = DeviceManager(device_name=args.device)

    # Predictor sınıfını kullanacağız, çünkü Generator'ı onun içinde kullanıyoruz.
    # Predictor, model yüklemeyi ve cihaz yönetimini zaten ele alıyor.
    predictor = ModelPredictor(
        model_checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        tokenizer_type=args.tokenizer_type,
        device_name=args.device
    )
    
    # Generator'ı Predictor'ın içindeki model ve tokenizer ile başlat
    generator = TextGenerator(
        model=predictor.model,
        tokenizer_instance=predictor.tokenizer,
        device_manager=predictor.device_manager
    )

    generated_text = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        add_bos_token=args.add_bos_token
    )
    log.info(f"Üretilen Metin:\n{generated_text}")
    log.info("Metin üretimi görevi tamamlandı.")


def run_predict(args):
    """Modelden doğrudan tahminler yapar (logits, olasılıklar vb.)."""
    log.info("Tahmin görevi başlatılıyor...")

    # Predictor sınıfını kullan
    predictor = ModelPredictor(
        model_checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        tokenizer_type=args.tokenizer_type,
        device_name=args.device
    )

    if args.predict_type == "logits":
        logits = predictor.predict_logits(args.prompt)
        log.info(f"Metin için logits boyutu: {logits.shape}")
        log.info(f"Logits (ilk 5 token, ilk 10 vocab_size): {logits[0, :5, :10]}")
    elif args.predict_type == "next_token_probs":
        probs, vocab_tokens = predictor.predict_next_token_probs(args.prompt)
        # En yüksek 5 olasılıklı tokenı ve olasılıklarını yazdır
        top_probs, top_indices = torch.topk(probs, k=min(10, len(vocab_tokens)))
        top_tokens = [vocab_tokens[idx.item()] for idx in top_indices]
        log.info(f"Bir sonraki token olasılıkları (ilk 10):")
        for token, prob in zip(top_tokens, top_probs):
            log.info(f"  '{token}': {prob:.4f}")
    else:
        log.error(f"Desteklenmeyen tahmin tipi: {args.predict_type}. 'logits' veya 'next_token_probs' olmalı.")
        sys.exit(1)
    
    log.info("Tahmin görevi tamamlandı.")


def main():
    parser = argparse.ArgumentParser(description="EchoModel ana çalıştırma betiği.")

    # Ortak argümanlar
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train", "preprocess_data", "train_tokenizer", "generate", "predict"],
        help="Çalıştırılacak ana görev."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs",
        help="Tüm çalıştırmaların (loglar, checkpoint'ler, tokenizer varlıkları) ana çıktı dizini."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Loglama seviyesi."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Kullanılacak cihaz ('auto' otomatik algılar)."
    )

    # Alt görevlere özel argümanlar (add_argument_group ile daha düzenli)
    
    # 1. Eğitim Argümanları
    train_parser = parser.add_argument_group("Eğitim Görevi Argümanları")
    train_parser.add_argument("--train_data_path", type=str, help="Eğitim veri dosyasının yolu.")
    train_parser.add_argument("--eval_data_path", type=str, help="Doğrulama veri dosyasının yolu.")
    train_parser.add_argument("--tokenizer_path", type=str, help="Tokenlayıcı dosyalarının bulunduğu dizin.")
    train_parser.add_argument("--tokenizer_type", type=str, default="bpe", choices=["bpe", "char"], help="Kullanılacak tokenlayıcı tipi.")
    train_parser.add_argument("--num_train_epochs", type=int, default=3, help="Eğitilecek epoch sayısı.")
    # Daha fazla eğitim ayarı BaseConfig'ten okunacak

    # 2. Veri Ön İşleme Argümanları
    preprocess_parser = parser.add_argument_group("Veri Ön İşleme Görevi Argümanları")
    preprocess_parser.add_argument("--input_dir", type=str, help="Ham veri dizini (preprocess_data için).")
    preprocess_parser.add_argument("--processed_output_dir", type=str, help="İşlenmiş verilerin kaydedileceği dizin (preprocess_data için).")
    preprocess_parser.add_argument(
        "--preprocessor_config",
        type=str,
        default='{"min_len": 50, "max_len": 20000, "remove_html": true, "normalize_whitespace": true, "remove_empty_lines": true, "deduplicate": true}',
        help="DatasetPreprocessor için JSON string olarak yapılandırma ayarları."
    )
    preprocess_parser.add_argument(
        "--split_data",
        action="store_true",
        help="İşlenmiş veriyi eğitim, doğrulama ve test setlerine ayır."
    )
    preprocess_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Veri ayırma için rastgelelik tohumu."
    )

    # 3. Tokenizer Eğitimi Argümanları
    train_tokenizer_parser = parser.add_argument_group("Tokenizer Eğitimi Görevi Argümanları")
    train_tokenizer_parser.add_argument(
        "--tokenizer_input_paths",
        nargs="+",
        type=str,
        help="Tokenizer eğitimi için kullanılacak metin dosyalarının veya dizinlerin yolu (boşlukla ayrılmış)."
    )
    train_tokenizer_parser.add_argument(
        "--tokenizer_output_dir",
        type=str,
        default="tokenizer_assets",
        help="Eğitilen tokenizer dosyalarının kaydedileceği dizin."
    )
    train_tokenizer_parser.add_argument(
        "--tokenizer_vocab_size",
        type=int,
        default=BaseConfig().vocab_size,
        help="Tokenizer kelime haznesinin boyutu."
    )
    train_tokenizer_parser.add_argument(
        "--tokenizer_type_train", # Çakışmayı önlemek için farklı isim
        type=str,
        default="bpe",
        choices=["bpe", "char"],
        help="Eğitilecek tokenizer tipi."
    )

    # 4. Üretim (Generate) Argümanları
    generate_parser = parser.add_argument_group("Metin Üretimi Görevi Argümanları")
    generate_parser.add_argument("--checkpoint_path", type=str, help="Yüklenecek model checkpoint dosyasının yolu.")
    generate_parser.add_argument("--prompt", type=str, help="Metin üretimi için başlangıç metni.")
    generate_parser.add_argument("--max_new_tokens", type=int, default=50, help="Üretilecek maksimum yeni token sayısı.")
    generate_parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "top_k", "nucleus"], help="Üretme stratejisi.")
    generate_parser.add_argument("--temperature", type=float, default=1.0, help="Örnekleme sıcaklığı.")
    generate_parser.add_argument("--top_k", type=int, default=50, help="Top-K örnekleme için K değeri.")
    generate_parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus örnekleme için Top-P değeri.")
    generate_parser.add_argument("--do_sample", action="store_true", help="Rastgele örnekleme yapılıp yapılmayacağı.")
    generate_parser.add_argument("--add_bos_token", action="store_true", help="Prompt'un başına BOS tokenı ekle.")

    # 5. Tahmin (Predict) Argümanları
    predict_parser = parser.add_argument_group("Tahmin Görevi Argümanları")
    predict_parser.add_argument("--predict_type", type=str, default="next_token_probs", choices=["logits", "next_token_probs"], help="Tahmin tipi.")
    # checkpoint_path, tokenizer_path, tokenizer_type, device zaten generate_parser'da tanımlı

    args = parser.parse_args()

    # Loglama sistemini kur (çalıştırma adına göre)
    current_run_name = f"{args.task}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    setup_logging(log_level=args.log_level, output_dir=args.output_dir, run_name=current_run_name)
    log = logging.getLogger(__name__)

    # Göreve göre ilgili fonksiyonu çağır
    if args.task == "train":
        run_train(args)
    elif args.task == "preprocess_data":
        preprocess_data_script_main(args) # scripts/preprocess_data.py'nin main'ini çağır
    elif args.task == "train_tokenizer":
        # train_tokenizer.py'nin main'i kendi argümanlarını bekliyor.
        # Burada argümanları args'tan alıp ona uygun bir args objesi oluşturmalıyız.
        # Veya train_tokenizer.py'nin main'ini doğrudan çağırmalıyız ve args'ı düzenlemeliyiz.
        log.info("Tokenizer eğitim görevi başlatılıyor (main.py üzerinden).")
        # Yeni bir Namespace objesi oluşturup train_tokenizer_script_main'e özel argümanları atıyoruz.
        tokenizer_args = argparse.Namespace(
            input_paths=args.tokenizer_input_paths,
            tokenizer_type=args.tokenizer_type_train,
            vocab_size=args.tokenizer_vocab_size,
            tokenizer_output_dir=args.tokenizer_output_dir,
            output_dir=args.output_dir, # Loglar için genel çıktı dizini
            log_level=args.log_level
        )
        train_tokenizer_script_main(tokenizer_args)
    elif args.task == "generate":
        run_generate(args)
    elif args.task == "predict":
        run_predict(args)
    else:
        log.error(f"Geçersiz görev: {args.task}")
        sys.exit(1)

    log.info("Ana betik çalışması tamamlandı.")


if __name__ == "__main__":
    main()