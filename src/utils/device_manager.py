
# src/utils/device_manager.py

import torch
import logging
from typing import Union
import torch.nn as nn

log = logging.getLogger(__name__)

class DeviceManager:
    """
    Modelin ve tensörlerin GPU veya CPU cihazlarına yerleştirilmesini yönetir.
    Tek bir merkezi noktadan cihaz seçimini ve atamasını sağlar.
    """
    def __init__(self, device_name: str = "auto"):
        """
        Args:
            device_name (str): Kullanılacak cihazın adı ('auto', 'cuda', 'cpu').
                                'auto' seçildiğinde, CUDA varsa GPU, yoksa CPU kullanılır.
        """
        self.device = self._get_device(device_name)
        log.info(f"Cihaz yöneticisi başlatıldı. Kullanılan cihaz: {self.device}")

    def _get_device(self, device_name: str) -> torch.device:
        """
        Belirtilen cihaz adına göre bir torch.device objesi döndürür.
        """
        if device_name == "auto":
            if torch.cuda.is_available():
                selected_device = torch.device("cuda")
                log.info("CUDA (GPU) mevcut. Otomatik olarak GPU seçildi.")
            elif torch.backends.mps.is_available(): # Apple Silicon (MPS) desteği
                selected_device = torch.device("mps")
                log.info("MPS (Apple Silicon GPU) mevcut. Otomatik olarak MPS seçildi.")
            else:
                selected_device = torch.device("cpu")
                log.info("CUDA veya MPS GPU mevcut değil. Otomatik olarak CPU seçildi.")
        elif device_name == "cuda":
            if torch.cuda.is_available():
                selected_device = torch.device("cuda")
                log.info("CUDA (GPU) manuel olarak seçildi.")
            else:
                log.warning("CUDA seçildi ancak GPU mevcut değil. CPU'ya geri dönülüyor.")
                selected_device = torch.device("cpu")
        elif device_name == "mps":
            if torch.backends.mps.is_available():
                selected_device = torch.device("mps")
                log.info("MPS (Apple Silicon GPU) manuel olarak seçildi.")
            else:
                log.warning("MPS seçildi ancak MPS mevcut değil. CPU'ya geri dönülüyor.")
                selected_device = torch.device("cpu")
        elif device_name == "cpu":
            selected_device = torch.device("cpu")
            log.info("CPU manuel olarak seçildi.")
        else:
            raise ValueError(f"Geçersiz cihaz adı: {device_name}. 'auto', 'cuda', 'mps' veya 'cpu' olmalı.")
        
        return selected_device

    def to_device(self, item: Union[torch.Tensor, nn.Module]) -> Union[torch.Tensor, nn.Module]:
        """
        Bir tensörü veya nn.Module'ü mevcut seçili cihaza taşır.
        """
        return item.to(self.device)

    @property
    def current_device(self) -> torch.device:
        """
        Şu an kullanılan torch.device objesini döndürür.
        """
        return self.device

    def print_device_info(self):
        """
        Kullanılan cihaz hakkında detaylı bilgi basar.
        """
        print(f"\n--- Cihaz Bilgisi ---")
        print(f"Kullanılan Cihaz: {self.device}")
        if self.device.type == 'cuda':
            print(f"  Adı: {torch.cuda.get_device_name(0)}")
            print(f"  Toplam Bellek: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            print(f"  Tahsis Edilmiş Bellek: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
            print(f"  Önbelleğe Alınmış Bellek: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
        elif self.device.type == 'mps':
            print("  Apple Silicon (MPS) Cihazı")
            # MPS için detaylı bellek bilgisi almak CUDA kadar kolay olmayabilir.
        print("--------------------\n")

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("DeviceManager testi başlatılıyor...")

    # Otomatik cihaz seçimi testi
    print("\n--- Otomatik Cihaz Seçimi ---")
    dm_auto = DeviceManager(device_name="auto")
    dummy_tensor_auto = torch.randn(2, 2)
    dummy_tensor_auto_on_device = dm_auto.to_device(dummy_tensor_auto)
    print(f"Tensör cihazda: {dummy_tensor_auto_on_device.device}")
    dm_auto.print_device_info()

    # CPU seçimi testi
    print("\n--- CPU Cihaz Seçimi ---")
    dm_cpu = DeviceManager(device_name="cpu")
    dummy_tensor_cpu = torch.randn(2, 2)
    dummy_tensor_cpu_on_device = dm_cpu.to_device(dummy_tensor_cpu)
    print(f"Tensör cihazda: {dummy_tensor_cpu_on_device.device}")
    dm_cpu.print_device_info()

    # CUDA seçimi testi (eğer GPU varsa)
    if torch.cuda.is_available():
        print("\n--- CUDA Cihaz Seçimi ---")
        dm_cuda = DeviceManager(device_name="cuda")
        dummy_tensor_cuda = torch.randn(2, 2)
        dummy_tensor_cuda_on_device = dm_cuda.to_device(dummy_tensor_cuda)
        print(f"Tensör cihazda: {dummy_tensor_cuda_on_device.device}")
        dm_cuda.print_device_info()
    else:
        print("\n--- CUDA Cihaz Seçimi (GPU mevcut değil) ---")
        print("CUDA cihaz mevcut değil, test atlandı veya CPU'ya düşüldü.")

    # Geçersiz cihaz adı testi
    print("\n--- Geçersiz Cihaz Adı Testi ---")
    try:
        DeviceManager(device_name="invalid_device")
    except ValueError as e:
        print(f"Hata yakalandı (beklenen): {e}")

    print("\nDeviceManager testleri tamamlandı. ✅")