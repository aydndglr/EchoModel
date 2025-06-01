# setup.py

from setuptools import setup, find_packages
import os

# README.md dosyasını oku
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# requirements.txt dosyasını oku
with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read().splitlines()

# Sürüm numarasını belirle
# Daha sonra bir 'version.py' dosyası oluşturup buradan okuyabiliriz.
# Şimdilik sabit bir değer verelim.
VERSION = "0.1.0"

setup(
    name='echomodel',
    version=VERSION,
    author='https://github.com/aydndglr (Aydın DAĞLAR) ',
    description='A modular and extensible Decoder-Only Transformer-based language model.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aydndglr/EchoModel', 
    project_urls={
        "Bug Tracker": "https://github.com/aydndglr/EchoModel/issues",
        "Source Code": "https://github.com/aydndglr/EchoModel",
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Education',
    ],
    keywords='deep learning, language model, transformer, pytorch, nlp, ai, llm',
    package_dir={'': 'src'}, # Kaynak kodların src dizini altında olduğunu belirtir
    packages=find_packages(where='src'), # src dizini altındaki tüm Python paketlerini bul
    python_requires='>=3.8',
    install_requires=install_requires,
    # Ek dosyaları dahil etmek için (config'ler, tokenizer varlıkları vb.)
    include_package_data=True,
    package_data={
        '': [
            'configs/*.yaml',
            'tokenizer_assets/*',
            'data/processed/*', # Eğer işlenmiş verileri de paketlemeyi düşünürseniz
            'notebooks/*.ipynb',
            'scripts/*.py',
            'utils/*.py',
            'data_processing/*.py',
            'data_processing/format_converters/*.py',
            'model/*.py',
            'model/components/*.py',
            'dataset/*.py',
            'training/*.py',
            'inference/*.py',
            'tokenizer/*.py'
        ]
    },
    # Betiklerin doğrudan çalıştırılmasını sağlamak için entry_points (isteğe bağlı)
    entry_points={
        'console_scripts': [
            'echomodel=main:main', # `echomodel` komutunu `src/main.py`'deki main fonksiyonuna bağlar
            'echomodel_preprocess=scripts.preprocess_data:main',
            'echomodel_train_tokenizer=scripts.train_tokenizer:main',
            # Diğer scriptleri de buraya ekleyebiliriz
        ],
    },
)