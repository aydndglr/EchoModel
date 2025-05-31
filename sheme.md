EchoModel/                                 # Ana proje dizini. Tüm projenin en üst seviye klasörü.
├── data/                                  # Veri setlerini depolamak için ana dizin.
│   ├── raw/                               # İndirilen, ham ve işlenmemiş veri setleri.
│   │   └── raw_dataset_name/              # Belirli bir ham veri setinin klasörü (örn. 'wikipedia_2024_01', 'alpaca_raw').
│   │       └── part1.jsonl / part1.xml.bz2 / ... # Ham veri setinin parçaları, orijinal formatlarında.
│   └── processed/                         # Ham verinin temizlenmiş, dönüştürülmüş ve modele hazır hale getirilmiş hali.
│       ├── train_data.parquet             # Eğitime hazır, işlenmiş eğitim verisi (Parquet gibi verimli formatlarda).
│       ├── val_data.parquet               # Eğitime hazır, işlenmiş doğrulama (validation) verisi.
│       └── test_data.parquet              # Eğitime hazır, işlenmiş test verisi (modelin son değerlendirmesi için).
├── src/                                   # Projenin tüm Python kaynak kodunu içeren ana dizin.
│   ├── __init__.py                        # 'src' dizinini bir Python paketi olarak işaretler. Modüllerin içeri aktarılmasını sağlar.
│   ├── config/                            # Model ve eğitim ayarlarını içeren yapılandırma dosyaları.
│   │   ├── __init__.py                    # 'config' dizinini bir Python paketi yapar.
│   │   ├── base_config.py                 # Genel proje ayarları ve temel hiperparametreler (örn. batch_size, device).
│   │   └── model_config.py                # Modele özgü hiperparametreler (örn. d_model, n_layers, n_heads).
│   ├── data_processing/                   # Ham veriyi işleme, temizleme ve modele uygun formata dönüştürme mantığı.
│   │   ├── __init__.py                    # 'data_processing' dizinini bir Python paketi yapar.
│   │   ├── dataset_preprocessor.py        # Çeşitli veri kaynaklarını temizleyen, filtreleyen ve standardize eden ana sınıf.
│   │   ├── format_converters/             # Farklı veri formatlarını (Alpaca, Hugging Face vb.) ortak bir ara formata dönüştüren modüller.
│   │   │   ├── __init__.py                # 'format_converters' dizinini bir Python paketi yapar.
│   │   │   ├── alpaca_converter.py        # Alpaca instruction-tuning veri formatını dönüştürme.
│   │   │   ├── hf_dataset_converter.py    # Hugging Face 'datasets' kütüphanesinden verileri entegre etme.
│   │   │   └── generic_parser.py          # Genel metin dosyalarını, XML veya JSON gibi formatları ayrıştırma.
│   │   └── data_splitter.py               # İşlenmiş veriyi eğitim, doğrulama ve test kümelerine ayıran mantık.
│   ├── tokenizer/                         # Metni sayılara dönüştüren ve geri döndüren tokenizer ile ilgili modüller.
│   │   ├── __init__.py                    # 'tokenizer' dizinini bir Python paketi yapar.
│   │   ├── bpe_tokenizer.py               # Byte-Pair Encoding (BPE) tokenizer'ı eğitme, kaydetme ve yükleme.
│   │   ├── char_tokenizer.py              # Karakter tabanlı tokenizer (alternatif veya basit senaryolar için).
│   │   └── special_tokens.py              # Model için özel token'ların (örn. <PAD>, <BOS>, <EOS>, <UNK>) tanımları.
│   ├── model/                             # Dil modelinin mimarisini ve bileşenlerini tanımlayan dizin.
│   │   ├── __init__.py                    # 'model' dizinini bir Python paketi yapar.
│   │   ├── components/                    # Transformer mimarisinin temel yapı taşları.
│   │   │   ├── __init__.py                # 'components' dizinini bir Python paketi yapar.
│   │   │   ├── embeddings.py              # Token gömme ve konumsal gömme katmanlarının tanımı.
│   │   │   ├── multi_head_attention.py    # Çoklu Kafa Maskeli Dikkat mekanizmasının kodu.
│   │   │   ├── feed_forward.py            # Konum bazlı ileri besleme ağının kodu.
│   │   │   └── layer_norm.py              # Katman Normalizasyonu katmanının kodu.
│   │   ├── transformer_decoder_block.py   # Yukarıdaki component'leri bir araya getiren tek bir Transformer Decoder bloğunun tanımı.
│   │   └── echo_transformer.py            # Modelin ana sınıfı; birden çok Transformer Decoder bloğunu birleştirir.
│   ├── dataset/                           # PyTorch Dataset ve DataLoader sınıfları.
│   │   ├── __init__.py                    # 'dataset' dizinini bir Python paketi yapar.
│   │   ├── base_text_dataset.py           # Tokenleştirilmiş veriyi okuyan ve PyTorch'un beklediği formatta sunan temel Dataset sınıfı.
│   │   ├── data_collator.py               # Farklı uzunluktaki örnekleri aynı batçeye sığdırmak için padding uygulayan araç.
│   │   └── custom_data_loader.py          # Veri setini yükleme, işleme ve PyTorch DataLoader'a besleme mantığını yöneten ana sınıf.
│   ├── training/                          # Model eğitim süreciyle ilgili modüller.
│   │   ├── __init__.py                    # 'training' dizinini bir Python paketi yapar.
│   │   ├── trainer.py                     # Ana eğitim ve doğrulama döngüsünü yöneten sınıf.
│   │   ├── optimizer.py                   # Optimizasyon algoritmalarının (örn. AdamW) ve öğrenme oranı çizelgelerinin (scheduler) yönetimi.
│   │   └── loss_function.py               # Modelin tahminleri ile gerçek etiketler arasındaki farkı ölçen kayıp fonksiyonunun tanımı.
│   ├── inference/                         # Eğitilmiş modelle tahmin yapma ve metin üretme (generation) ile ilgili modüller.
│   │   ├── __init__.py                    # 'inference' dizinini bir Python paketi yapar.
│   │   ├── generator.py                   # Metin üretme algoritmaları (örn. greedy decoding, beam search, top-k, nucleus sampling).
│   │   └── predictor.py                   # Eğitilmiş modeli yükleyip yeni girdilerle tahmin/üretim yapma arayüzü.
│   ├── utils/                             # Çeşitli yardımcı fonksiyonlar ve genel araçlar.
│   │   ├── __init__.py                    # 'utils' dizinini bir Python paketi yapar.
│   │   ├── logger.py                      # Eğitim ve diğer süreçler için loglama sistemi.
│   │   ├── device_manager.py              # GPU/CPU cihaz seçimini ve yönetimi için yardımcılar.
│   │   ├── checkpoint_manager.py          # Model ağırlıklarını kaydetme ve yükleme mantığı.
│   │   └── metrics.py                     # Eğitim ve değerlendirme metriklerini (örn. perplexity, BLEU) hesaplama.
│   ├── scripts/                           # Doğrudan çalıştırılabilir betikler.
│   │   ├── preprocess_data.py             # Ham veriyi işleme ve 'processed' dizinine kaydetme ana betiği.
│   │   └── train_tokenizer.py             # Tokenizer'ı veri üzerinde eğiten ve 'tokenizer_assets'e kaydeden ana betik.
│   └── main.py                            # Projenin ana giriş noktası; komut satırı argümanlarına göre farklı scriptleri tetikler.
├── notebooks/                             # Jupyter veya Colab not defterleri, hızlı denemeler ve veri keşfi için.
│   ├── 01_data_exploration.ipynb          # Ham veriyi keşfetmek ve analiz etmek için not defteri.
│   ├── 02_tokenizer_dev.ipynb             # Tokenizer'ı geliştirmek ve test etmek için not defteri.
│   ├── 03_model_architecture_test.ipynb   # Model mimarisinin küçük ölçekte test edilmesi için not defteri.
│   └── 04_data_preprocessing_test.ipynb   # Veri ön işleme ve dönüştürme süreçlerini test etmek için not defteri.
├── saved_models/                          # Eğitilmiş model ağırlıklarının (checkpoint'lerin) kaydedildiği yer.
│   └── echo_model_checkpoint_epoch_X.pt   # Kaydedilmiş bir model ağırlık dosyası örneği.
├── logs/                                  # Eğitim sürecine ait metriklerin, TensorBoard kayıtlarının ve diğer logların depolandığı yer.
├── configs/                               # YAML/JSON formatında harici yapılandırma dosyaları.
│   ├── training_config.yaml               # Eğitim döngüsü ayarları (epoch sayısı, öğrenme oranı çizelgesi vb.).
│   └── model_config.yaml                  # Model mimarisiyle ilgili tüm ayarlar.
│   └── data_config.yaml                   # Veri setlerinin yolları, kullanılacak formatlar ve ön işleme adımları.
├── tokenizer_assets/                      # Eğitilmiş tokenizer'ın sözlük ve birleştirme kuralları gibi dosyaları.
├── README.md                              # Projenin genel açıklaması, kurulum talimatları ve kullanım yönergeleri.
├── requirements.txt                       # Projenin ihtiyaç duyduğu tüm Python kütüphanelerini listeleyen dosya.
├── .gitignore                             # Git versiyon kontrol sistemi tarafından göz ardı edilecek dosyaları belirtir.
├── LICENSE                                # Projenin açık kaynak lisansı (örn. MIT Lisansı).
└── setup.py                               # Python paketi olarak kurulabilir hale getirmek için kullanılan kurulum betiği.