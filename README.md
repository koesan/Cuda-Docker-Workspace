# Cuda-Docker-Workspace

Bu proje, GPU destekli TensorFlow ve PyTorch ortamını doğrudan ana bilgisayara karmaşık kurulumlar yapmadan, Docker konteyneri içinde kolayca çalıştırmak için hazırlandı. CUDA, cuDNN, sürücü uyumsuzlukları, versiyon çakışmaları gibi sorunlarla uğraşmadan, sadece birkaç komutla kullanıma hazır bir geliştirme ortamına sahip olabilirsiniz. Özellikle birden fazla projede farklı framework ve CUDA sürümleri gerekiyorsa, bunları izole konteynerlerde yönetmek hem daha güvenli hem de çok daha pratiktir. Bu rehber, Linux sistemler için adım adım Docker kurulumu, NVIDIA GPU entegrasyonu ve örnek bir ML workflow’u içerir.

> Dikkat: Bu rehber Debain(Ubuntu)/Arch/Fedora Linux dağıtımları içindir. Her dağıtımın paket isimleri ve komutları farklı olabilir; sistem güncel olmalıdır.

---

## Hızlı Özet (1 satır)

1. Docker kur
2. NVIDIA sürücüsünü + NVIDIA Container Toolkit kur
3. Dockerfile oluştur ve image build et
4. `docker run --gpus all -v $(pwd):/workspace ...` ile çalıştır
5. VS Code ile bağlan (Attach to Container)

---

## 1) Docker Kurulumu (kısa)

Aşağıdakiler en basit ve yaygın yollar. İstediğin dağıtıma göre uygula.

### Debian / Ubuntu

```bash
# Hızlı kurulum (paket yöneticisi)
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
# Kullanıcıyı docker grubuna ekle (çıkış/giriş yap sonra etkili olur)
sudo usermod -aG docker $USER
```

Eğer daha güncel Docker istiyorsan Docker'ın resmi reposunu kullan.

### Arch Linux

```bash
sudo pacman -Syu docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

### Fedora

```bash
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

> Değişikliklerin etkili olması için çıkış yapıp tekrar giriş yapın veya terminali yeniden başlatın.

---

## 2) NVIDIA Sürücüleri ve Container Toolkit (GPU Erişimi)

Docker konteynerlerinin host GPU’yu kullanabilmesi için iki şey gerekir:

* Ana sistemde **NVIDIA sürücüleri** kurulu ve çalışıyor olmalı
* Ayrıca **nvidia-container-toolkit** kurulmalı ve Docker ile yapılandırılmalı

### Önce kontrol edelim:

```bash
# GPU sürücüleri yüklü mü? (çıktı geliyorsa kurulu ve çalışıyor demektir)
nvidia-smi
```

> Eğer `nvidia-smi` çıktısı başarılıysa NVIDIA sürücüleri kurulu demektir. Bu durumda sadece **nvidia-container-toolkit** kurmanız yeterlidir.

---

### Debian / Ubuntu

```bash
# (Eğer sürücü kurulu değilse — kontrol: nvidia-smi)
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-utils
```

```bash
# NVIDIA Container Toolkit kurulumu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### Arch Linux

```bash
# (Eğer sürücü kurulu değilse)
sudo pacman -Syu nvidia nvidia-utils
```

```bash
# NVIDIA Container Toolkit (AUR üzerinden - yay veya paru gerekir)
yay -S nvidia-container-toolkit
# veya
paru -S nvidia-container-toolkit

# Docker ile entegrasyon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### Fedora

```bash
# (Eğer sürücü kurulu değilse)
sudo dnf install -y akmod-nvidia
```

```bash
# NVIDIA Container Toolkit kurulumu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### Kurulumdan Sonra Doğrulama

```bash
# Host sistemde GPU tanınıyor mu?
nvidia-smi
```

```bash
# Container içinden GPU erişimi test et
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

> Eğer `nvidia-smi` çıktısı başarılıysa NVIDIA sürücüleri kurulu demektir. Bu durumda sadece **nvidia-container-toolkit** kurmanız yeterlidir.

---

## 3) Dockerfile hazırlama (örnek — TensorFlow 2.15 / CUDA12.2 uyumlu)

`Dockerfile`'ı proje klasörüne koyun:

```dockerfile
# CUDA 12.2.2 + cuDNN 8 + Ubuntu 22.04 tabanlı imaj
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Ortam değişkeni - apt sorularını engelle
ENV DEBIAN_FRONTEND=noninteractive

# Python ve gerekli paketler
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

# Pip ve temel araçları güncelle
RUN python3 -m pip install --upgrade pip setuptools wheel

# TensorFlow 2.15 (CUDA 12.2 desteği ile geliyor)
RUN pip install --no-cache-dir tensorflow==2.15.*

# Çalışma dizini
WORKDIR /workspace

# Varsayılan komut
CMD ["/bin/bash"]

```

> Not: PyTorch'u yüklemek isterseniz, PyTorch resmi sitesinden (`pytorch.org`) CUDA sürümünüze uygun `--index-url` komutunu kopyalayın ve Docker kurulumundan sonra pip ile yükleyin.

---

## 4) Image build etme ve container çalıştırma

### Build (legacy builder ile hızlı)

```bash
# Dockerfile bulunduğu dizinde
DOCKER_BUILDKIT=0 docker build -t gpu-dl-workspace:tf2.15 .
# veya (normal build)
docker build -t gpu-dl-workspace:tf2.15 .
```

### Çalıştırma — proje klasörünü şu anki dizine bağlama

```bash
# Interaktif, çıkınca container silinir
docker run --gpus all -it --rm -v $(pwd):/workspace --name dl-temp gpu-dl-workspace:tf2.15 /bin/bash

# Arka planda kalıcı container (VS Code ile attach için)
docker run --gpus all -d --name dl-container -v $(pwd):/workspace gpu-dl-workspace:tf2.15 tail -f /dev/null

# Container içine bağlan
docker exec -it dl-container /bin/bash
```

> Eğer bulunduğun dizini (`$(pwd)`) doğrudan `/workspace` yapmak istiyorsan `-v $(pwd):/workspace` kullan; eğer alt klasör bağlamak istersen `-v $(pwd)/my-project:/workspace`.

---

## 5) GPU erişimini test etme

Konteyner içinde:

```bash
# GPU lista
nvidia-smi
```

> Eğer yukarıdaki komut konteyner içinden host GPU bilgilerini gösteriyorsa, kurulum başarılıdır ve `nvidia-container-toolkit` doğru yapılandırılmıştır.

```bash
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

> Eğer yukarıdaki komutun çıktısı aşağıdaki gibi görünüyorsa:
> `GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`
> bu, kurulumun başarılı olduğunu ve `nvidia-container-toolkit`'in doğru şekilde yapılandırıldığını gösterir.

---

## 6) Konteyner yönetimi: durdurma, başlatma, silme

```bash
# Çalışan konteynerleri göster
docker ps
# Tüm konteynerleri (durmuş olanlar dahil)
docker ps -a

# Durdur
docker stop dl-container
# Başlat
docker start dl-container
# Konteyner içinde komut çalıştır
docker exec -it dl-container /bin/bash
# Sil
docker rm dl-container

# Image sil
docker rmi gpu-dl-workspace:tf2.15
```
---

## 7) VS Code ile bağlanma (kod host'ta, çalıştırma container içinde)

1. VS Code'a **Dev Containers / Remote - Containers** eklentisini yükleyin.
2. Container'ı arka planda çalıştırın (`--name dl-container -d` olarak).
3. VS Code: `F1` → **Remote-Containers: Attach to Running Container...** → `dl-container` seçin.
4. VS Code açıldığında `/workspace` dizinini göreceksiniz. Terminal ve debug konteyner içindedir — kodu hostta düzenleyip kaydedin, çalıştırma container içinden olur.

> Alternatif: `.devcontainer/devcontainer.json` ile tam bir Dev Container yapılandırması hazırlayıp GitHub Codespaces veya VS Code Remote ile uyumlu hale getirebilirsiniz.

---

## 8) İpuçları & sık karşılaşılan sorunlar

* **`Could not find cuda drivers`**: `nvidia-container-toolkit` kurulu mu, `docker run --gpus all` kullanıldı mı kontrol edin.
* **`manifest unknown`**: `FROM nvidia/cuda:...` satırındaki tag Docker Hub'da yok. `docker pull` ile önce test edin.
* **Paket izin uyarıları (pip root warnings)**: Geliştirme için container içinde virtualenv veya conda kullanabilirsiniz.
* **Disk / I/O**: Büyük veri setlerini host üzerinde tutup `-v /data:/data` ile bağlayın.

---

## 9) Örnek: Hızlı workflow

1. Proje dizininde `Dockerfile` koy
2. `DOCKER_BUILDKIT=0 docker build -t myworkspace:tf .`
3. `docker run --gpus all -d --name myws -v $(pwd):/workspace myworkspace:tf tail -f /dev/null`
4. `docker exec -it myws bash` → `python train.py`
5. Bitince `docker stop myws` veya bırak `--restart unless-stopped` ile her zaman açık kalsın.

---

## 🔗 Kaynaklar

### 🔧 Docker & NVIDIA Kurulumu

* **Docker Kurulum Dökümantasyonu**:
  [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
* **NVIDIA Container Toolkit (nvidia-docker)**:
  [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
* **NVIDIA Container Toolkit GitHub (güncel repo)**:
  [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* **NVIDIA CUDA Docker İmajları (Docker Hub)**:
  [https://hub.docker.com/r/nvidia/cuda](https://hub.docker.com/r/nvidia/cuda)

---

### 📦 TensorFlow (GPU Destekli)

* **Resmi TensorFlow Kurulum Rehberi**:
  [https://www.tensorflow.org/install](https://www.tensorflow.org/install)
* **TensorFlow GPU Desteği ve Sürümler Uyumluluğu**:
  [https://www.tensorflow.org/install/source#gpu](https://www.tensorflow.org/install/source#gpu)
* **TensorFlow + Docker Kullanımı**:
  [https://www.tensorflow.org/install/docker](https://www.tensorflow.org/install/docker)

---

### 🔥 PyTorch (GPU Destekli)

* **Resmi PyTorch Kurulum Sayfası**:
  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
* **PyTorch Docker İmajları (NVIDIA destekli)**:
  [https://hub.docker.com/r/pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch)

---

### 🧰 VS Code — Remote Container Geliştirme

* **Remote - Containers (Dev Containers) Eklentisi**:
  [https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
* **VS Code ile Docker Geliştirme Ortamı Oluşturma**:
  [https://code.visualstudio.com/docs/devcontainers/containers](https://code.visualstudio.com/docs/devcontainers/containers)

---

### 🐳 Ekstra Faydalı Bağlantılar

* **NVIDIA Sürücü Uyumluluk Tablosu (CUDA / cuDNN)**:
  [https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
* **cuDNN İndir (NVIDIA hesabı gerekli)**:
  [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
* **Docker Compose (çoklu servis için)**:
  [https://docs.docker.com/compose/](https://docs.docker.com/compose/)
* **Linux Dağıtımlarına Göre CUDA Kurulumu (manuel)**:
  [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
