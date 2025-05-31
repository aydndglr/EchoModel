
# src/training/trainer.py

import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
from typing import Dict, Any, Optional, Tuple, List
from tqdm.auto import tqdm # İlerleme çubuğu için

from src.config.base_config import BaseConfig
from src.config.model_config import ModelConfig
from src.model.echo_transformer import EchoTransformer # Ana modelimiz
from src.training.loss_function import LanguageModelingLoss # Kayıp fonksiyonumuz
from src.training.optimizer import get_optimizer, get_scheduler # Optimizer ve scheduler fonksiyonlarımız
from src.utils.device_manager import DeviceManager # Cihaz yöneticimiz
from src.utils.checkpoint_manager import CheckpointManager # Checkpoint yöneticimiz
from src.utils.metrics import MetricCalculator # Metrik hesaplayıcımız

log = logging.getLogger(__name__)

class Trainer:
    """
    EchoModel için eğitim döngüsünü, değerlendirme sürecini ve checkpoint yönetimini yöneten sınıf.
    """
    def __init__(
        self,
        model: EchoTransformer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        base_config: BaseConfig,
        model_config: ModelConfig,
        device_manager: DeviceManager,
        checkpoint_manager: CheckpointManager,
        metric_calculator: MetricCalculator
    ):
        """
        Args:
            model (EchoTransformer): Eğitilecek dil modeli.
            train_dataloader (DataLoader): Eğitim veri yükleyicisi.
            eval_dataloader (DataLoader): Doğrulama veri yükleyicisi.
            base_config (BaseConfig): Genel yapılandırma ayarları.
            model_config (ModelConfig): Modele özel yapılandırma ayarları.
            device_manager (DeviceManager): Cihaz yönetimi objesi.
            checkpoint_manager (CheckpointManager): Checkpoint yönetimi objesi.
            metric_calculator (MetricCalculator): Metrik hesaplama objesi.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.base_config = base_config
        self.model_config = model_config
        self.device_manager = device_manager
        self.checkpoint_manager = checkpoint_manager
        self.metric_calculator = metric_calculator

        # Modeli ve veri yükleyicileri cihaza taşı
        self.model.to(self.device_manager.current_device)
        
        # Optimizer ve scheduler oluştur
        self.optimizer = get_optimizer(self.model, self.base_config)
        
        # Toplam eğitim adımı sayısını hesapla
        # Her bir epoch için batçe sayısı * epoch sayısı
        self.num_update_steps_per_epoch = len(self.train_dataloader) // self.base_config.gradient_accumulation_steps
        self.total_training_steps = self.num_update_steps_per_epoch * self.base_config.num_train_epochs # num_train_epochs config'e eklenecek
        
        # `num_train_epochs` BaseConfig'e eklenmediyse varsayılan olarak 3 epoch alalım.
        if not hasattr(self.base_config, 'num_train_epochs'):
            self.base_config.num_train_epochs = 3
            log.warning(f"BaseConfig'te 'num_train_epochs' bulunamadı. Varsayılan olarak {self.base_config.num_train_epochs} epoch kullanılacak.")
            self.total_training_steps = self.num_update_steps_per_epoch * self.base_config.num_train_epochs

        self.lr_scheduler = get_scheduler(self.optimizer, self.base_config, self.total_training_steps)
        
        self.loss_fn = LanguageModelingLoss(ignore_index=-100) # Padding tokenları -100
        
        self.global_step = 0
        self.current_epoch = 0
        
        log.info("Eğitimci başarıyla başlatıldı.")
        log.info(f"Toplam eğitim adımı sayısı (effective batch_size dahil): {self.total_training_steps}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Bir eğitim epoch'unu çalıştırır.
        """
        self.model.train() # Modeli eğitim moduna al
        total_loss = 0.0
        
        # tqdm ile ilerleme çubuğu
        train_iterator = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch+1}/{self.base_config.num_train_epochs} (Eğitim)")

        # Gradyan biriktirme için sayaç
        accumulation_counter = 0

        for batch_idx, batch in enumerate(train_iterator):
            # Batch'i cihaza taşı
            input_ids = self.device_manager.to_device(batch["input_ids"])
            labels = self.device_manager.to_device(batch["labels"])
            attention_mask = self.device_manager.to_device(batch["attention_mask"])

            # İleri besleme (forward pass)
            # EchoTransformer'ın çıktısı logits ve present_key_values_list
            logits, _ = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
            
            # Kayıp hesaplama
            loss = self.loss_fn(logits, labels)
            
            # Gradyan biriktirme (loss'ı batçe boyutuna göre ölçekle)
            # Her adımda kaybı accumulate_gradient_steps'e bölüyoruz, böylece
            # toplam kayıp effective_batch_size'a göre ölçeklenir.
            loss = loss / self.base_config.gradient_accumulation_steps
            loss.backward() # Geri yayılım (backpropagation)

            accumulation_counter += 1

            if accumulation_counter % self.base_config.gradient_accumulation_steps == 0:
                # Gradyan biriktirme adımları tamamlandığında optimizer'ı güncelle

                # Gradyan kırpma (gradient clipping)
                if self.base_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.base_config.max_grad_norm)

                self.optimizer.step() # Parametreleri güncelle
                self.lr_scheduler.step() # Öğrenme oranını güncelle
                self.optimizer.zero_grad() # Gradyanları sıfırla
                self.global_step += 1
                total_loss += loss.item() * self.base_config.gradient_accumulation_steps # Gerçek batçe başına toplam loss

                # Loglama
                if self.global_step % self.base_config.log_steps == 0:
                    avg_loss = total_loss / (batch_idx + 1) # Gerçek batçe sayısı
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    log.info(f"Adım: {self.global_step}/{self.total_training_steps}, Kayıp: {avg_loss:.4f}, LR: {current_lr:.6f}")
                    train_iterator.set_postfix(loss=avg_loss, lr=current_lr) # tqdm ilerleme çubuğuna bilgi ekle

                # Değerlendirme ve Checkpoint kaydetme
                if self.global_step % self.base_config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    log.info(f"Değerlendirme sonuçları (Adım {self.global_step}): {eval_metrics}")
                    
                    # Checkpoint kaydetme
                    self.checkpoint_manager.save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.global_step,
                        self.current_epoch,
                        eval_metrics
                    )
                    self.model.train() # Değerlendirme sonrası tekrar eğitim moduna geç

        avg_epoch_loss = total_loss / self.num_update_steps_per_epoch # Ortalama epoch kaybı
        log.info(f"Epoch {self.current_epoch+1} tamamlandı. Ortalama Kayıp: {avg_epoch_loss:.4f}")
        return {"avg_train_loss": avg_epoch_loss}

    @torch.no_grad() # Gradyan hesaplamasını devre dışı bırak
    def evaluate(self) -> Dict[str, float]:
        """
        Modeli doğrulama veri seti üzerinde değerlendirir.
        """
        self.model.eval() # Modeli değerlendirme moduna al
        total_eval_loss = 0.0
        total_eval_accuracy = 0.0
        total_eval_topk_accuracy = 0.0
        num_eval_batches = 0

        eval_iterator = tqdm(self.eval_dataloader, desc="Değerlendirme")

        for batch in eval_iterator:
            input_ids = self.device_manager.to_device(batch["input_ids"])
            labels = self.device_manager.to_device(batch["labels"])
            attention_mask = self.device_manager.to_device(batch["attention_mask"])

            # İleri besleme (forward pass)
            logits, _ = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
            
            # Kayıp hesaplama
            loss = self.loss_fn(logits, labels)
            total_eval_loss += loss.item()
            
            # Metrik hesaplama
            accuracy = self.metric_calculator.calculate_accuracy(logits.detach().cpu(), labels.detach().cpu())
            topk_accuracy = self.metric_calculator.calculate_topk_accuracy(logits.detach().cpu(), labels.detach().cpu(), k=5)
            
            total_eval_accuracy += accuracy
            total_eval_topk_accuracy += topk_accuracy
            num_eval_batches += 1

            eval_iterator.set_postfix(loss=loss.item(), acc=accuracy)

        avg_eval_loss = total_eval_loss / num_eval_batches
        avg_eval_accuracy = total_eval_accuracy / num_eval_batches
        avg_eval_topk_accuracy = total_eval_topk_accuracy / num_eval_batches
        
        # Perplexity hesapla
        perplexity = self.metric_calculator.calculate_perplexity(torch.tensor(avg_eval_loss))

        metrics = {
            "eval_loss": avg_eval_loss,
            "eval_perplexity": perplexity,
            "eval_accuracy": avg_eval_accuracy,
            "eval_topk_accuracy": avg_eval_topk_accuracy
        }
        return metrics

    def train(self):
        """
        Modelin tüm eğitim sürecini başlatır veya devam ettirir.
        """
        log.info("Eğitim süreci başlatılıyor...")
        
        # Checkpoint'ten devam etme (eğer varsa)
        latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
        if latest_checkpoint:
            log.info(f"En son checkpoint bulundu: {latest_checkpoint}. Eğitim bu checkpoint'ten devam edecek.")
            loaded_state = self.checkpoint_manager.load_checkpoint(str(latest_checkpoint), self.model, self.optimizer)
            self.global_step = loaded_state["step"]
            self.current_epoch = loaded_state["epoch"]
            log.info(f"Eğitime adım {self.global_step}, epoch {self.current_epoch} itibarıyla devam ediliyor.")
        else:
            log.info("Önceki checkpoint bulunamadı. Eğitim sıfırdan başlıyor.")

        for epoch in range(self.current_epoch, self.base_config.num_train_epochs):
            self.current_epoch = epoch
            log.info(f"Epoch {self.current_epoch+1}/{self.base_config.num_train_epochs} başlatılıyor...")
            
            self.train_epoch()
            
            # Epoch sonunda bir değerlendirme yap (eğer zaten eval_steps içinde yapılmadıysa)
            if (self.global_step % self.base_config.eval_steps != 0) or (self.global_step == 0):
                eval_metrics = self.evaluate()
                log.info(f"Epoch {self.current_epoch+1} sonu değerlendirme: {eval_metrics}")
            
            # Epoch sonunda modeli kaydet
            self.checkpoint_manager.save_checkpoint(
                self.model,
                self.optimizer,
                self.global_step,
                self.current_epoch,
                eval_metrics if 'eval_metrics' in locals() else {} # Eğer eval_metrics hesaplanmadıysa boş gönder
            )
            log.info(f"Epoch {self.current_epoch+1} checkpoint'i kaydedildi.")
        
        log.info("Eğitim süreci tamamlandı.")

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("Trainer testi başlatılıyor...")

    import shutil
    from src.utils.logger import setup_logging
    from src.dataset.custom_data_loader import CustomDataLoader
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.tokenizer.special_tokens import PAD_TOKEN # PAD_TOKEN'ı import etmeliyiz
    
    # Test için geçici dizinler ve temizlik
    test_output_dir = "test_runs_trainer"
    test_run_name = "trainer_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # 1. Konfigürasyonlar
    base_cfg = BaseConfig(
        num_train_epochs=1, # Sadece 1 epoch eğitelim hızlı test için
        train_batch_size=2,
        eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=10,
        eval_steps=10, # Her 10 adımda bir değerlendirme
        log_steps=5, # Her 5 adımda bir loglama
        output_dir=os.path.join(test_output_dir, "outputs") # Trainer çıktılarını buraya kaydet
    )
    model_cfg = ModelConfig(
        d_model=64, # Küçük bir model boyutu ile hızlı test
        n_layers=2,
        n_heads=4
    )
    # ModelConfig'te include_bias tanımlanmadı, varsayılan olarak True alınacak

    # 2. Tokenizer Oluşturma
    tokenizer_save_path = Path(test_output_dir) / "tokenizer_assets"
    os.makedirs(tokenizer_save_path, exist_ok=True)
    temp_tokenizer_train_file = Path(test_output_dir) / "trainer_tokenizer_train_data.txt"
    with open(temp_tokenizer_train_file, "w", encoding="utf-8") as f:
        f.write("Bu, trainer testi için tokenizer eğitimi metnidir. Lorem ipsum dolor sit amet. Alpha beta gamma delta. Test c.d.e.f.g.h.i.j.k.l.m.n.o.p.q.r.s.t.u.v.w.x.y.z. A B C D E F G H I J K L M N O P Q R S T U V W X Y Z\n")
    bpe_tokenizer_instance = BPETokenizer(vocab_size=base_cfg.vocab_size)
    bpe_tokenizer_instance.train(files=[str(temp_tokenizer_train_file)], save_path=str(tokenizer_save_path))
    
    # PAD_TOKEN_ID'nin mevcut olduğundan emin ol
    if getattr(bpe_tokenizer_instance, 'pad_token_id', None) is None:
        bpe_tokenizer_instance.pad_token_id = bpe_tokenizer_instance.token_to_id(PAD_TOKEN)
        if bpe_tokenizer_instance.pad_token_id is None:
            # Tokenizer'ın varsayılan kelime haznesindeki boş bir ID'yi kullanabiliriz.
            # Veya config'ten varsayılan olarak bir ID atayabiliriz.
            bpe_tokenizer_instance.pad_token_id = bpe_tokenizer_instance.vocabulary_size # En son ID
            log.warning(f"PAD_TOKEN ID'si bulunamadı, varsayılan olarak {bpe_tokenizer_instance.pad_token_id} atanıyor.")
            

    # 3. Dummy Veri Yükleyicileri Oluşturma
    dummy_data_path_train = Path(test_output_dir) / "train_data.jsonl"
    dummy_data_path_eval = Path(test_output_dir) / "eval_data.jsonl"
    
    train_texts = [f"Eğitim cümlesi {i}. Bu model eğitimi için kullanılan bir metin parçasıdır. " * 5 for i in range(50)] # 50 örnek
    eval_texts = [f"Doğrulama cümlesi {i}. Bu model doğrulaması için kullanılan bir metin parçasıdır. " * 5 for i in range(10)] # 10 örnek

    with open(dummy_data_path_train, "w", encoding="utf-8") as f:
        for text in train_texts:
            json.dump({"text": text}, f)
            f.write("\n")
    with open(dummy_data_path_eval, "w", encoding="utf-8") as f:
        for text in eval_texts:
            json.dump({"text": text}, f)
            f.write("\n")

    train_dl = CustomDataLoader(
        data_filepath=str(dummy_data_path_train),
        tokenizer_path=str(tokenizer_save_path),
        max_seq_len=base_cfg.max_seq_len,
        batch_size=base_cfg.train_batch_size,
        tokenizer_type="bpe",
        num_workers=0, # Test için 0 worker
        shuffle=True,
        pin_memory=False
    ).get_dataloader()

    eval_dl = CustomDataLoader(
        data_filepath=str(dummy_data_path_eval),
        tokenizer_path=str(tokenizer_save_path),
        max_seq_len=base_cfg.max_seq_len,
        batch_size=base_cfg.eval_batch_size,
        tokenizer_type="bpe",
        num_workers=0, # Test için 0 worker
        shuffle=False,
        pin_memory=False
    ).get_dataloader()

    # 4. Model, Cihaz Yöneticisi, Checkpoint Yöneticisi, Metrik Hesaplayıcı
    model = EchoTransformer(base_config=base_cfg, model_config=model_cfg)
    device_manager = DeviceManager(device_name=base_cfg.device)
    checkpoint_manager = CheckpointManager(save_dir=base_cfg.output_dir) # Output_dir'i kullanıyoruz
    metric_calculator = MetricCalculator()

    # 5. Trainer'ı oluştur ve eğit
    print("\n--- Trainer Oluşturma ve Eğitim Başlatma Testi ---")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        base_config=base_cfg,
        model_config=model_cfg,
        device_manager=device_manager,
        checkpoint_manager=checkpoint_manager,
        metric_calculator=metric_calculator
    )

    try:
        trainer.train()
        print("Eğitim süreci başarıyla tamamlandı. ✅")

        # Checkpointlerin kaydedildiğini kontrol et
        assert len(list(Path(base_cfg.output_dir).glob("*/checkpoints/*.pt"))) > 0, "Hiç checkpoint kaydedilemedi!"
        print(f"Checkpointler '{base_cfg.output_dir}' dizininde bulundu. ✅")

    except Exception as e:
        print(f"Trainer testi BAŞARISIZ: {e} ❌")
        log.error(f"Trainer testi BAŞARISIZ: {e}", exc_info=True)
    finally:
        # Test sonrası temizlik
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)
            log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")

    print("\nTrainer tüm testleri tamamlandı. ✅")