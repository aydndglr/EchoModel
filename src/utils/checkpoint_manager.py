
# src/utils/checkpoint_manager.py

from datetime import datetime
import torch
import os
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import torch.nn as nn

log = logging.getLogger(__name__)

class CheckpointManager:
    """
    Model ve optimizer checkpoint'lerini kaydetme ve yüklemeyi yönetir.
    Eğitim sürecinin güvenli bir şekilde durdurulup devam ettirilmesini sağlar.
    """
    def __init__(self, save_dir: str):
        """
        Args:
            save_dir (str): Checkpoint dosyalarının kaydedileceği ana dizin.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Checkpoint yöneticisi başlatıldı. Kayıt dizini: {self.save_dir}")

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None
    ) -> Path:
        """
        Modelin, optimizer'ın ve eğitim durumunun bir checkpoint'ini kaydeder.

        Args:
            model (nn.Module): Kaydedilecek PyTorch modeli.
            optimizer (torch.optim.Optimizer): Kaydedilecek optimizer.
            step (int): Mevcut global eğitim adımı.
            epoch (int): Mevcut eğitim epoch'u.
            metrics (Dict[str, float]): O anki değerlendirme metrikleri (örn. {"val_loss": 0.5}).
            filename (Optional[str]): Checkpoint dosyasının adı. Belirtilmezse otomatik oluşturulur.

        Returns:
            Path: Kaydedilen checkpoint dosyasının tam yolu.
        """
        if filename is None:
            filename = f"checkpoint_step_{step:07d}.pt"
        
        filepath = self.save_dir / filename
        
        # Checkpoint sözlüğü
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            torch.save(checkpoint, filepath)
            log.info(f"Checkpoint kaydedildi: {filepath}")
            return filepath
        except Exception as e:
            log.error(f"Checkpoint kaydederken hata oluştu: {e}")
            raise

    def load_checkpoint(self, filepath: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Bir checkpoint dosyasından modelin ve optimizer'ın durumunu yükler.

        Args:
            filepath (str): Yüklenecek checkpoint dosyasının tam yolu.
            model (nn.Module): Durumu yüklenecek PyTorch modeli.
            optimizer (Optional[torch.optim.Optimizer]): Durumu yüklenecek optimizer (isteğe bağlı).

        Returns:
            Dict[str, Any]: Yüklenen eğitim durumu (step, epoch, metrics vb.).
        """
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"Checkpoint dosyası bulunamadı: {filepath}")

        try:
            # map_location='cpu' ile yükleyip sonra modele `.to(device)` yapmak daha esnektir.
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage) # map_location='cpu'
            
            # Modeli yükle
            model.load_state_dict(checkpoint["model_state_dict"])
            log.info(f"Model durumu '{filepath}' adresinden yüklendi.")

            # Optimizer'ı yükle (eğer sağlanmışsa)
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                log.info(f"Optimizer durumu '{filepath}' adresinden yüklendi.")
            elif optimizer is not None and "optimizer_state_dict" not in checkpoint:
                log.warning(f"Checkpoint '{filepath}' optimizer durumu içermiyor, optimizer yüklenemedi.")

            # Eğitim durumunu döndür
            return {
                "step": checkpoint.get("step", 0),
                "epoch": checkpoint.get("epoch", 0),
                "metrics": checkpoint.get("metrics", {}),
                "timestamp": checkpoint.get("timestamp", None)
            }
        except Exception as e:
            log.error(f"Checkpoint yüklerken hata oluştu: {e}")
            raise

    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Kayıt dizinindeki en son kaydedilen checkpoint dosyasını bulur.
        Dosyaları `checkpoint_step_XXXXXXX.pt` formatına göre sıralar.
        """
        checkpoints = sorted(
            [f for f in self.save_dir.glob("checkpoint_step_*.pt") if f.is_file()],
            key=lambda f: int(f.stem.split('_')[-1]) # `checkpoint_step_0000001.pt` -> 1
        )
        if checkpoints:
            return checkpoints[-1] # En son dosya
        return None

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("CheckpointManager testi başlatılıyor...")

    # Loglama sistemini kur (test için gerekli)
    from src.utils.logger import setup_logging
    setup_logging(log_level="INFO", output_dir="test_runs_ckpt", run_name="ckpt_test_run")
    log = logging.getLogger(__name__) # Logger'ı tekrar alıyoruz

    # Test dizinleri
    test_save_dir = "test_runs_ckpt/ckpt_test_run/checkpoints"
    
    # CheckpointManager'ı başlat
    ckpt_manager = CheckpointManager(save_dir=test_save_dir)

    # Basit bir model ve optimizer oluştur
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)

    dummy_model = DummyModel()
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)

    # Test Senaryosu 1: Checkpoint kaydetme
    print("\n--- Test Senaryosu 1: Checkpoint Kaydetme ---")
    try:
        saved_path = ckpt_manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            step=100,
            epoch=1,
            metrics={"val_loss": 0.123, "val_accuracy": 0.95}
        )
        assert saved_path.is_file(), "Checkpoint dosyası kaydedilemedi!"
        print(f"Checkpoint başarıyla kaydedildi: {saved_path} ✅")
    except Exception as e:
        print(f"Checkpoint kaydetme testi BAŞARISIZ: {e} ❌")
        saved_path = None # Hata durumunda path'i None yap
    
    # Test Senaryosu 2: Checkpoint yükleme
    print("\n--- Test Senaryosu 2: Checkpoint Yükleme ---")
    if saved_path:
        new_dummy_model = DummyModel()
        new_dummy_optimizer = torch.optim.Adam(new_dummy_model.parameters(), lr=0.001)
        
        # Yüklemeden önceki parametreleri kaydet
        old_model_weight = new_dummy_model.linear.weight.clone()
        
        try:
            loaded_state = ckpt_manager.load_checkpoint(
                filepath=str(saved_path), # Path objesini string'e çevir
                model=new_dummy_model,
                optimizer=new_dummy_optimizer
            )
            assert loaded_state["step"] == 100, "Yüklenen adım yanlış!"
            assert loaded_state["epoch"] == 1, "Yüklenen epoch yanlış!"
            assert "val_loss" in loaded_state["metrics"], "Metrikler yüklenmedi!"
            
            # Parametrelerin değişip değişmediğini kontrol et (yüklenen model ve optimizer)
            assert not torch.equal(old_model_weight, new_dummy_model.linear.weight), "Model ağırlıkları yüklenmedi!"
            
            print(f"Checkpoint başarıyla yüklendi. Yüklenen Adım: {loaded_state['step']} ✅")
        except Exception as e:
            print(f"Checkpoint yükleme testi BAŞARISIZ: {e} ❌")
    else:
        print("Checkpoint kaydedilemediği için yükleme testi atlandı.")

    # Test Senaryosu 3: En son checkpoint'i bulma
    print("\n--- Test Senaryosu 3: En Son Checkpoint'i Bulma ---")
    try:
        latest_ckpt = ckpt_manager.find_latest_checkpoint()
        if saved_path:
            assert latest_ckpt == saved_path, "En son checkpoint doğru bulunamadı!"
            print(f"En son checkpoint doğru şekilde bulundu: {latest_ckpt} ✅")
        else:
            assert latest_ckpt is None, "Hiç checkpoint yokken yanlışlıkla bir tane bulundu!"
            print("En son checkpoint doğru şekilde bulunamadı (beklendiği gibi). ✅")
    except Exception as e:
        print(f"En son checkpoint bulma testi BAŞARISIZ: {e} ❌")
    
    # Test klasörlerini temizle
    import shutil
    if os.path.exists("test_runs_ckpt"):
        shutil.rmtree("test_runs_ckpt")
    print("\nCheckpointManager testleri tamamlandı. ✅")