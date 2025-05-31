
# src/inference/predictor.py

import os
import torch
import logging
import torch.nn.functional as F
from typing import List, Union, Dict, Any, Tuple
from pathlib import Path

from src.model.echo_transformer import EchoTransformer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.char_tokenizer import CharTokenizer
from src.utils.device_manager import DeviceManager
from src.utils.checkpoint_manager import CheckpointManager
from src.config.base_config import BaseConfig
from src.config.model_config import ModelConfig

log = logging.getLogger(__name__)

class ModelPredictor:
    """
    Eğitilmiş bir EchoTransformer modelini yükler ve çeşitli tahmin görevleri için kullanır.
    """
    def __init__(
        self,
        model_checkpoint_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        tokenizer_type: str = "bpe", # 'bpe' veya 'char'
        device_name: str = "auto"
    ):
        """
        Args:
            model_checkpoint_path (Union[str, Path]): Eğitilmiş model checkpoint dosyasının yolu.
            tokenizer_path (Union[str, Path]): Tokenizer dosyalarının bulunduğu dizin veya dosya yolu.
            tokenizer_type (str): Kullanılacak tokenizer tipi ('bpe' veya 'char').
            device_name (str): Modelin yükleneceği cihaz ('auto', 'cuda', 'cpu').
        """
        self.device_manager = DeviceManager(device_name=device_name)
        self.checkpoint_manager = CheckpointManager(save_dir="temp_checkpoints") # Sadece yükleme için geçici bir dizin

        self.tokenizer = self._load_tokenizer(tokenizer_path, tokenizer_type)
        self.model = self._load_model_from_checkpoint(model_checkpoint_path)
        
        self.model.eval() # Modeli çıkarım (evaluation) moduna al
        self.model.to(self.device_manager.current_device) # Modeli doğru cihaza taşı

        log.info("ModelPredictor başlatıldı. Model yüklenmeye hazır.")

    def _load_tokenizer(self, path: Union[str, Path], tokenizer_type: str):
        """Yardımcı fonksiyon: tokenizer'ı yükler."""
        if tokenizer_type == "bpe":
            return BPETokenizer.from_pretrained(path)
        elif tokenizer_type == "char":
            return CharTokenizer.from_pretrained(path)
        else:
            raise ValueError(f"Desteklenmeyen tokenizer tipi: {tokenizer_type}. 'bpe' veya 'char' olmalı.")

    def _load_model_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> EchoTransformer:
        """Yardımcı fonksiyon: checkpoint'ten modeli yükler."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint dosyası bulunamadı: {checkpoint_path}")

        # Checkpoint'ten model ve base config'leri yükle (varsayalım checkpoint'te saklı)
        # Eğer checkpoint dosyası BaseConfig ve ModelConfig'i içermiyorsa,
        # bunları varsayılan veya ayrı bir config dosyasından yüklememiz gerekir.
        # Bizim CheckpointManager'ımızda model_state_dict, optimizer_state_dict, step, epoch, metrics var.
        # Modelin config'leri ayrı olarak sağlanmalı veya checkpoint'te saklanmalı.
        # OLMO'da config.yaml'den model config yükleniyor.
        # Biz de şimdilik config'leri varsayılan olarak başlatıp, sonra checkpoint'ten gelen config'leri kullanabiliriz.
        
        # Geçici olarak varsayılan config'leri kullan
        base_cfg = BaseConfig()
        model_cfg = ModelConfig()

        dummy_model = EchoTransformer(base_config=base_cfg, model_config=model_cfg)
        
        log.info(f"Checkpoint '{checkpoint_path}' yükleniyor...")
        loaded_state = self.checkpoint_manager.load_checkpoint(str(checkpoint_path), dummy_model)
        
        # Eğer config'ler checkpoint'te yoksa ve modelin init'inde varsayılanlar kullanılıyorsa,
        # bu yaklaşım geçerli olur. Daha sağlam bir çözüm için config'leri checkpoint'e dahil etmeliyiz.
        # Şimdilik, model başarıyla yüklendiği varsayımıyla devam edelim.
        log.info(f"Model step {loaded_state.get('step', 'N/A')} itibarıyla yüklendi.")
        
        return dummy_model

    @torch.no_grad()
    def predict_logits(self, text: str) -> torch.Tensor:
        """
        Verilen metin için modelin çıkış logits'lerini döndürür.
        Batch boyutu 1 olarak işlem yapar.

        Args:
            text (str): Tahmin yapılacak metin.

        Returns:
            torch.Tensor: Logits tensörü (1, seq_len, vocab_size).
        """
        if not text.strip():
            log.warning("Boş metin girdisi. Boş tensör döndürülecek.")
            return torch.empty(0, 0, self.model.vocab_size)

        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0) # (1, seq_len)
        input_ids = self.device_manager.to_device(input_ids)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attention_mask = self.device_manager.to_device(attention_mask)

        logits, _ = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
        return logits

    @torch.no_grad()
    def predict_next_token_probs(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """
        Verilen metnin ardından gelebilecek bir sonraki tokenların olasılık dağılımını döndürür.

        Args:
            text (str): Modelin çıktısını tahmin edeceği metin.

        Returns:
            Tuple[torch.Tensor, List[str]]:
                - next_token_probs (torch.Tensor): Bir sonraki tokenlar için olasılık dağılımı (vocab_size).
                - vocab_tokens (List[str]): Kelime haznesindeki tokenların sıralı listesi.
        """
        logits = self.predict_logits(text)
        if logits.numel() == 0:
            return torch.tensor([]), []

        # Sadece son tokenın logits'ini al
        # shape: (1, vocab_size)
        next_token_logits = logits[:, -1, :]
        
        # Softmax ile olasılıklara dönüştür
        next_token_probs = F.softmax(next_token_logits, dim=-1).squeeze(0) # (vocab_size)

        # Kelime haznesindeki tüm tokenları listele
        vocab_tokens = [self.tokenizer.id_to_token(i) for i in range(self.tokenizer.vocabulary_size)]

        return next_token_probs, vocab_tokens


# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("ModelPredictor testi başlatılıyor...")

    import shutil
    from src.utils.logger import setup_logging
    from src.config.base_config import BaseConfig
    from src.config.model_config import ModelConfig
    from src.model.echo_transformer import EchoTransformer
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.tokenizer.special_tokens import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
    
    test_output_dir = "test_runs_predictor"
    test_run_name = "predictor_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # 1. Dummy Configs ve Tokenizer Oluşturma
    base_cfg = BaseConfig(vocab_size=50) # Küçük vocab size
    model_cfg = ModelConfig(d_model=32, n_layers=1, n_heads=2) # Çok küçük model

    tokenizer_save_path = Path(test_output_dir) / "tokenizer_assets"
    os.makedirs(tokenizer_save_path, exist_ok=True)
    with open(Path(test_output_dir) / "tokenizer_train_data.txt", "w", encoding="utf-8") as f:
        f.write("a b c d e f g h i j k l m n o p q r s t u v w x y z. 1 2 3 4 5 6 7 8 9 0. " + BOS_TOKEN + " " + EOS_TOKEN + " " + PAD_TOKEN + " " + UNK_TOKEN + "\n")
    
    bpe_tokenizer_instance_for_predictor = BPETokenizer(vocab_size=base_cfg.vocab_size)
    bpe_tokenizer_instance_for_predictor.train(files=[str(Path(test_output_dir) / "tokenizer_train_data.txt")], save_path=str(tokenizer_save_path))
    
    # Tokenizer'ın pad_token_id'si güncellenmeli
    if getattr(bpe_tokenizer_instance_for_predictor, 'pad_token_id', None) is None:
        bpe_tokenizer_instance_for_predictor.pad_token_id = bpe_tokenizer_instance_for_predictor.token_to_id(PAD_TOKEN)
    if getattr(bpe_tokenizer_instance_for_predictor, 'eos_token_id', None) is None:
        bpe_tokenizer_instance_for_predictor.eos_token_id = bpe_tokenizer_instance_for_predictor.token_to_id(EOS_TOKEN)


    # 2. Dummy Model Eğitimi (Checkpoint oluşturmak için)
    # Gerçek eğitim döngüsüne girmeyeceğiz, sadece bir model state_dict'i oluşturup kaydedeceğiz.
    model_for_checkpoint = EchoTransformer(base_config=base_cfg, model_config=model_cfg)
    optimizer_for_checkpoint = torch.optim.Adam(model_for_checkpoint.parameters(), lr=0.001)
    
    # CheckpointManager kullanarak dummy model state'ini kaydet
    ckpt_manager_for_test = CheckpointManager(save_dir=Path(test_output_dir) / "model_checkpoints")
    dummy_checkpoint_path = ckpt_manager_for_test.save_checkpoint(
        model_for_checkpoint,
        optimizer_for_checkpoint,
        step=100,
        epoch=1,
        metrics={"dummy_loss": 0.5}
    )
    print(f"Dummy checkpoint oluşturuldu: {dummy_checkpoint_path}")


    # 3. ModelPredictor oluştur
    print("\n--- ModelPredictor Oluşturma Testi ---")
    predictor = ModelPredictor(
        model_checkpoint_path=dummy_checkpoint_path,
        tokenizer_path=str(tokenizer_save_path),
        tokenizer_type="bpe",
        device_name="cpu" # Test için CPU kullanalım
    )
    assert predictor.model is not None, "Model yüklenemedi!"
    print("ModelPredictor başarıyla oluşturuldu. ✅")

    # 4. predict_logits testi
    print("\n--- predict_logits Testi ---")
    test_prompt = "Merhaba dünya"
    logits_output = predictor.predict_logits(test_prompt)
    
    print(f"  Giriş metni: '{test_prompt}'")
    print(f"  Logits boyutu: {logits_output.shape}")
    
    # Beklenen boyut: (batch_size=1, seq_len, vocab_size)
    expected_seq_len = len(predictor.tokenizer.encode(test_prompt, add_special_tokens=True))
    assert logits_output.shape == (1, expected_seq_len, base_cfg.vocab_size), "Logits boyutu yanlış!"
    print("predict_logits testi başarılı. ✅")

    # 5. predict_next_token_probs testi
    print("\n--- predict_next_token_probs Testi ---")
    next_token_probs, vocab_tokens = predictor.predict_next_token_probs(test_prompt)
    
    print(f"  Bir sonraki token olasılıkları boyutu: {next_token_probs.shape}")
    print(f"  Kelime haznesi boyutu: {len(vocab_tokens)}")
    
    assert next_token_probs.shape == (base_cfg.vocab_size,), "Bir sonraki token olasılıkları boyutu yanlış!"
    assert len(vocab_tokens) == base_cfg.vocab_size, "Kelime haznesi token sayısı yanlış!"
    assert torch.isclose(next_token_probs.sum(), torch.tensor(1.0)), "Olasılıkların toplamı 1 değil!"
    print("predict_next_token_probs testi başarılı. ✅")

    print("\nModelPredictor tüm testleri tamamlandı. ✅")

    # Temizlik
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")