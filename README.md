# EchoModel: Gelişmiş, Modüler Büyük Dil Modeli

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Progress-blue" alt="Project Status">
  <img src="https://img.shields.io/badge/Built%20With-PyTorch-red" alt="Built With PyTorch">
  <img src="https://img.shields.io/badge/Contributors-aydndglr" alt="Contributors">
</p>

---

## 🚀 Projeye Genel Bakış

**EchoModel**, PyTorch kullanarak sıfırdan inşa edilen, modüler ve genişletilebilir bir **Decoder-Only Transformer** tabanlı büyük dil modelidir (LLM). Amacım, OLMO, Gemma veya GPT gibi modern LLM'lerin temel prensiplerini takip ederek, ancak kendi özel mimarim ile esnek veri işleme yeteneklerine sahip, yüksek performanslı bir model geliştirmektir. Proje yapısı, araştırma ve geliştirme için sağlam bir temel sunarken, gelecekte potansiyel olarak harici bellek entegrasyonu gibi yenilikçi özelliklere de kapı aralamaktadır.

Bu proje, özellikle dil modelleme ve metin üretimi alanında derinlemesine bilgi edinmek, kendi LLM'inizi baştan sona inşa etmek ve özelleştirmek isteyen geliştiriciler ve araştırmacılar için bir referans noktası olmayı hedeflemektedir.

---

## ✨ Temel Özellikler ve Yaklaşım

* **Decoder-Only Transformer Mimarisi:** Metin üretme (text generation) görevleri için optimize edilmiş, modern Transformer mimarisinin temelini kullanırız.
* **Tamamen Modüler Tasarım:** Her bir bileşen (gömme katmanları, dikkat mekanizmaları, Transformer blokları, veri işleme, eğitim döngüsü vb.) ayrı modüller halinde tasarlanmıştır. Bu, kodun okunabilirliğini, bakımını ve yeniden kullanılabilirliğini artırır.
* **Esnek Veri İşleme Pipeline'ı:** Wikipedia, Hugging Face `datasets`, Alpaca formatı gibi çeşitli veri kaynaklarını işleyebilecek genel bir veri ön işleme ve yükleme sistemi.
* **Özelleştirilebilir Tokenizer:** Kendi veri setlerimiz üzerinde eğitebileceğimiz Byte-Pair Encoding (BPE) veya karakter tabanlı tokenleyiciler için destek.
* **Temiz Eğitim ve Çıkarım Çerçevesi:** Sağlam bir eğitim döngüsü (`Trainer` sınıfı) ve eğitilmiş modellerle metin üretme (`Generator`) ve tahmin yapma (`Predictor`) araçları.
* **Aşamalı Geliştirme Stratejisi:** İlk aşamada sağlam bir temel Transformer modeli oluşturup, kazanılan tecrübe ile gelecekte "RAM tabanlı dinamik bellek" entegrasyonu gibi daha ileri seviye özelliklere geçiş ve dinamik eğitim için altyapı yapı için çalışma.

---

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Projeyi Klonlayın:**
    ```bash
    git clone https://github.com/aydndglr/EchoModel.git
    cd EchoModel
    ```

2.  **Python Ortamı Oluşturun (Önerilen: Conda veya venv):**
    ```bash
    # Conda ile
    conda create -n echomodel python=3.9
    conda activate echomodel

    # veya venv ile
    python -m venv .venv
    source .venv/bin/activate # Linux/macOS
    # .venv\Scripts\activate   # Windows
    ```

3.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **CUDA/cuDNN Kurulumu (GPU Kullanımı İçin):**
    Eğer NVIDIA GPU kullanıyorsanız, PyTorch'un GPU desteği için CUDA Toolkit ve cuDNN'in doğru sürümlerini yüklemeniz gerekmektedir. PyTorch resmi web sitesindeki kurulum yönergelerini (https://pytorch.org/get-started/locally/) takip edin.

---

## 🚀 Kullanım

### 1. Veri Hazırlığı

* Ham veri setlerinizi (örn. Wikipedia dump'ı, Alpaca JSON'ları) `data/raw/` klasörü altına uygun alt klasörlerde (örn. `data/raw/wikipedia/`, `data/raw/alpaca/`) yerleştirin.
* Veriyi işlemek ve `processed/` klasörüne kaydetmek için `preprocess_data.py` betiğini kullanın.
    ```bash
    python src/scripts/preprocess_data.py --config configs/data_config.yaml
    ```
    *(`data_config.yaml` dosyasını veri setinizin yol ve format bilgileriyle güncellemeniz gerekecektir.)*

### 2. Tokenizer Eğitimi

* İşlenmiş veri üzerinde kendi tokenizer'ınızı eğitmek için `train_tokenizer.py` betiğini çalıştırın. Eğitilen tokenizer varlıkları `tokenizer_assets/` altına kaydedilecektir.
    ```bash
    python src/scripts/train_tokenizer.py --config configs/tokenizer_config.yaml
    ```
    *(Bu config dosyası henüz oluşturulmadı, gerektiğinde eklenecektir.)*

### 3. Model Eğitimi

* Modelin hiperparametrelerini `configs/model_config.yaml` ve eğitim ayarlarını `configs/training_config.yaml` dosyalarında düzenleyin.
* Modeli eğitmek için `train_model.py` betiğini çalıştırın:
    ```bash
    python src/main.py --task train --model_config configs/model_config.yaml --train_config configs/training_config.yaml --data_config configs/data_config.yaml
    ```
    *(Veya doğrudan `python src/scripts/train_model.py` gibi bir betik yazabiliriz.)*
* Eğitilmiş model ağırlıkları `saved_models/` dizinine kaydedilecektir. Eğitim logları ve metrikler `logs/` dizininde bulunacaktır (TensorBoard ile izlenebilir).

### 4. Metin Üretimi (Inference)

* Eğitilmiş bir modeli yükleyin ve metin üretimi yapmak için `generate_text.py` betiğini kullanın:
    ```bash
    python src/main.py --task generate --checkpoint_path saved_models/your_model.pt --prompt "Your starting text here."
    ```
    *(Veya doğrudan `python src/inference/generator.py` gibi bir betik yazabiliriz.)*

---

## 🤝 Katkıda Bulunma

Proje, açık kaynaklıdır ve katkılara açıktır. Her türlü katkı (hata düzeltmeleri, yeni özellikler, dokümantasyon iyileştirmeleri, model eğitimi için teknolojk altyapı vb.) memnuniyetle karşılanır. Lütfen bir katkıda bulunmadan önce [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını (ileride eklenecektir) inceleyin.

---

## 📜 Lisans

Bu proje, **Apache 2.0** altında lisanslanmıştır ayrıca ticari kullanim icin ek sartlar bulunmaktadır . Daha fazla bilgi için `LICENSE` dosyasına bakın.

---

## 📧 İletişim

Sorularınız, geri bildirimleriniz veya işbirliği için lütfen [aydin.daglar@outlook.com](mailto:aydin.daglar@outlook.com) adresinden iletişime geçin.

---
