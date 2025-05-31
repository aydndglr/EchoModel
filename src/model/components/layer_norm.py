
# src/model/components/layer_norm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    Standart Layer Normalizasyon veya RMS Normalizasyon'u uygulayan bir katman.
    RMSNorm, Layer Normalizasyon'un basitleştirilmiş bir versiyonudur ve genellikle
    Transformer modellerinde kullanılır (örn. OLMO, Llama).

    Args:
        normalized_shape (int veya tuple): Normalizasyonun uygulanacağı şekil.
                                        Genellikle embedding_dim veya d_model.
        eps (float): Sayısal kararlılık için epsilon değeri.
        bias (bool): Ofset parametresi (bias) kullanılıp kullanılmayacağı. RMSNorm'da genellikle kullanılmaz.
        use_rms_norm (bool): Layer Normalizasyon yerine RMS Normalizasyon'u kullanıp kullanmayacağı.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = False, use_rms_norm: bool = True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) # PyTorch LayerNorm gibi tuple olmalı
        self.eps = eps
        self.use_rms_norm = use_rms_norm
        
        # Öğrenilebilir ağırlık parametresi (gain)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        
        # Bias parametresi (sadece standart LayerNorm'da kullanılır)
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMS Normalizasyon işlemini uygular.
        Kaynak: https://github.com/facebookresearch/llama/blob/main/llama/model.py
        """
        # RMS = sqrt(mean(x^2))
        # x_norm = x / RMS
        
        # mean(x^2) yerine (x * x).mean(-1, keepdim=True) daha verimli
        # Bu, karesi alınan değerlerin son boyutta ortalamasını alır.
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        return x / rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_rms_norm:
            # RMS Normalizasyon uygula
            x = self._rms_norm(x.float()).type_as(x) # float32'ye yükseltip tekrar orijinal dtype'a düşür (kararlılık için)
            output = x * self.weight
            if self.bias is not None: # RMSNorm'da bias genellikle yoktur
                output = output + self.bias
        else:
            # Standart Layer Normalizasyon uygula
            # PyTorch'un kendi nn.LayerNorm'unu kullanmak daha güvenli ve optimize.
            # Ancak biz kendi implementasyonumuzu göstermek için buraya manuel olarak ekliyoruz.
            # Gerçek projede doğrudan nn.LayerNorm kullanmak tercih edilebilir.
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False) # biased=False varsayılan
            x_normed = (x - mean) / torch.sqrt(var + self.eps)
            output = x_normed * self.weight
            if self.bias is not None:
                output = output + self.bias
        return output

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("LayerNorm (RMSNorm ve Standart) testi başlatılıyor...")

    # Parametreler
    batch_size = 2
    seq_len = 10
    d_model = 64
    eps = 1e-5

    # Rastgele giriş tensörü
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    print(f"Giriş tensörü boyutu: {dummy_input.shape}")

    # --- RMSNorm Testi ---
    print("\n--- RMS Normalizasyon Testi ---")
    rms_norm_layer = LayerNorm(d_model, eps=eps, bias=False, use_rms_norm=True)
    
    # Kendi RMSNorm'umuzun çıktısı
    output_rms_custom = rms_norm_layer(dummy_input)
    print(f"RMSNorm çıktı boyutu (Custom): {output_rms_custom.shape}")
    
    # RMSNorm'un özelliklerini kontrol et
    # Karesi alınmış elemanların ortalaması 1'e yakın olmalı (normalizasyon sonrası)
    mean_of_squares_rms = output_rms_custom.pow(2).mean(-1).mean().item()
    print(f"RMSNorm sonrası (karelerin ortalaması): {mean_of_squares_rms:.6f} (Beklenen ~1.0)")
    assert abs(mean_of_squares_rms - 1.0) < 0.1, "RMSNorm testi başarısız: ortalama kare 1'e yakın değil!"
    print("RMSNorm testi başarılı!")

    # --- Standart Layer Normalizasyon Testi ---
    print("\n--- Standart Layer Normalizasyon Testi ---")
    # Bias'lı ve bias'sız iki versiyonu test edelim
    
    # Kendi standart LayerNorm'umuz
    std_norm_layer_custom = LayerNorm(d_model, eps=eps, bias=True, use_rms_norm=False)
    output_std_custom = std_norm_layer_custom(dummy_input)
    print(f"Standart LayerNorm çıktı boyutu (Custom): {output_std_custom.shape}")
    
    # PyTorch'un kendi nn.LayerNorm'u ile karşılaştırma (referans)
    std_norm_layer_pytorch = nn.LayerNorm(d_model, eps=eps)
    std_norm_layer_pytorch.weight.data = std_norm_layer_custom.weight.data # Ağırlıkları kopyala
    if std_norm_layer_custom.bias is not None:
        std_norm_layer_pytorch.bias.data = std_norm_layer_custom.bias.data # Bias'ı kopyala
    output_std_pytorch = std_norm_layer_pytorch(dummy_input)

    # Çıktıları karşılaştır
    tolerance = 1e-4
    assert torch.allclose(output_std_custom, output_std_pytorch, atol=tolerance), \
        "Standart LayerNorm testi başarısız: Özel implementasyon PyTorch'a uymuyor!"
    print(f"Standart LayerNorm testi başarılı! (PyTorch ile fark: {torch.max(torch.abs(output_std_custom - output_std_pytorch)).item():.2e})")

    print("\nLayerNorm (RMSNorm ve Standart) tüm testleri tamamlandı.")