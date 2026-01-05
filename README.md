# 🌾 AgriBot - Kənd Təsərrüfatı Bilgi Sistemi

**Azərbaycan dilində kənd təsərrüfatı RAG (Retrieval-Augmented Generation) sistemi**

Graph RAG texnologiyası ilə işləyən ağıllı kənd təsərrüfatı axtarış sistemi. Bu sistem Neo4j qrafik verilənlər bazası, Pinecone vektor verilənlər bazası və Ollama LLM inteqrasiyası ilə hərtərəfli cavablar təqdim edir.

## 🎯 Xüsusiyyətlər

- ✅ **Azərbaycan dili dəstəyi**: Tam Azərbaycan dilində interfeys və sorğu imkanı
- ✅ **Hibrid axtarış**: Vektor oxşarlığı + Qrafik traversal
- ✅ **6 kənd təsərrüfatı sənədi**: PDF formatında Azərbaycan dilində məlumatlar
- ✅ **FastAPI web interfeysi**: Modern və responsiv dizayn
- ✅ **Docker dəstəyi**: Asan yerləşdirmə və test
- ✅ **Real-time AI cavabları**: Ollama gemma:2b modeli

## 📊 Texniki Arxitektura

```
┌─────────────────────────────────────────────────┐
│           FastAPI Web Interface                 │
│         (Jinja2 Templates + CSS)                │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│         SimpleGraphRAG Core                     │
│  (demo_graph_rag.py)                           │
└──┬──────────────────────────────────────────┬───┘
   │                                          │
┌──▼───────────────────┐        ┌────────────▼──────┐
│   Neo4j Graph DB     │        │  Pinecone Vector  │
│   - 24 Entities      │        │  - 47 Vectors     │
│   - 2 Relationships  │        │  - 1024 dim       │
└──────────────────────┘        └───────────────────┘
           │
     ┌─────▼──────┐
     │   Ollama   │
     │  gemma:2b  │
     └────────────┘
```

## 🚀 Tez Başlanğıc

### Metod 1: Docker (Tövsiyə edilir)

```bash
# 1. Reponu klonlayın
git clone https://github.com/Ismat-Samadov/agri_bot.git
cd agri_bot

# 2. .env faylını konfiqurasiya edin (artıq mövcuddur)
# NEO4J_URI, PINECONE_API_KEY və s.

# 3. Docker Compose ilə başladın
docker-compose up -d

# 4. Ollama modelini yükləyin (ilk dəfə)
docker exec -it agribot-ollama ollama pull gemma:2b

# 5. Brauzerə keçin
open http://localhost:8000
```

### Metod 2: Local Quraşdırma

```bash
# 1. Virtual mühit yaradın
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# 2. Asılılıqları quraşdırın
pip install -r requirements-simple.txt

# 3. Ollama quraşdırın və başladın (ayrıca terminal)
brew install ollama  # macOS
ollama serve
ollama pull gemma:2b

# 4. Web serveri başladın
python app.py

# 5. Brauzerə keçin
open http://localhost:8000
```

## 📁 Layihə Strukturu

```
agri_bot/
├── app.py                      # FastAPI ana tətbiq
├── demo_graph_rag.py          # Graph RAG əsas sinif
├── requirements-simple.txt    # Python asılılıqları
├── Dockerfile                 # Docker konfiqurasiyası
├── docker-compose.yml         # Docker Compose konfiqurasiyası
├── .env                       # Mühit dəyişənləri (Git-də yoxdur)
├── .gitignore                 # Git təhlükəsizliyi
│
├── templates/                 # Jinja2 şablonları
│   ├── base.html
│   ├── index.html            # Ana səhifə
│   ├── results.html          # Axtarış nəticələri
│   ├── stats.html            # Statistika səhifəsi
│   └── error.html            # Xəta səhifəsi
│
├── static/
│   └── css/
│       └── style.css         # Dizayn və stilizasiya
│
├── dataset/                   # Kənd təsərrüfatı PDF-ləri (6 sənəd)
│
└── scripts/                   # Bir dəfəlik skriptlər
    ├── test_simple.py        # Sistem testləri
    └── ingest_all_docs.py    # Sənəd yüklənməsi
```

## 🗄️ Verilənlər Bazası Konfiqurasiyası

### Neo4j Aura (Cloud)
- **URI**: `neo4j+s://9c0a7d96.databases.neo4j.io`
- **İstifadəçi**: neo4j
- **Status**: ✅ Aktiv (24 node, 2 relationship)

### Pinecone
- **İndeks**: agribot
- **Ölçü**: 1024
- **Model**: llama-text-embed-v2
- **Status**: ✅ Aktiv (47 vektor)

### Ollama
- **Model**: gemma:2b
- **Dil**: Çoxdilli (Azərbaycan dili dəstəyi)
- **Yerləşmə**: Local (http://localhost:11434)

## 💡 İstifadə

### Web İnterfeys

1. **Ana səhifə** (`/`): Axtarış qutusu və statistika
2. **Axtarış nəticələri** (`/search`): AI cavablar və mənbələr
3. **Statistika** (`/stats`): Sistem məlumatları

### Nümunə Suallar

```
Taxılın əsas xəstəlikləri hansılardır?
Bitkiçilikdə hansı metodlar tətbiq olunur?
Kənd təsərrüfatında kimyəvi maddələr haqqında məlumat verin
```

## 🔧 Konfiqurasiya

### Mühit Dəyişənləri (.env)

```env
# Neo4j
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Pinecone
PINECONE_API_KEY=your-api-key
PINECONE_INDEX_NAME=agribot
PINECONE_DIMENSIONS=1024

# Ollama (local)
OLLAMA_HOST=http://localhost:11434
```

## 📊 Sistem Statistikası

| Komponent | Miqdar | Status |
|-----------|--------|--------|
| PDF Sənədləri | 6 | ✅ Yüklənib |
| Neo4j Nodes | 24 | ✅ Aktiv |
| Neo4j Relationships | 2 | ✅ Aktiv |
| Pinecone Vectors | 47 | ✅ Aktiv |
| Chunks İşlənib | 28 | ✅ Tamamlandı |

## 🛠️ Əlavə Skriptlər

### Sistem Testi
```bash
python scripts/test_simple.py
```
Neo4j, Pinecone, Ollama və PDF oxuma qabiliyyətini yoxlayır.

### Yeni Sənədləri Yükləmək
```bash
# PDFs əlavə edin: dataset/ qovluğuna
python scripts/ingest_all_docs.py
```

## 🐳 Docker Əmrləri

```bash
# Başlat
docker-compose up -d

# Logları izlə
docker-compose logs -f

# Dayandır
docker-compose down

# Yenidən qur
docker-compose up -d --build

# Ollama modelləri
docker exec -it agribot-ollama ollama list
docker exec -it agribot-ollama ollama pull gemma:2b
```

## 📝 API Endpointləri

| Endpoint | Method | Təsvir |
|----------|--------|--------|
| `/` | GET | Ana səhifə |
| `/search` | POST | Axtarış sorğusu |
| `/stats` | GET | Sistem statistikası |

## 🔒 Təhlükəsizlik

- ✅ `.env` faylı Git-də ignore edilib
- ✅ Neo4j və Pinecone şifrələri qorunur
- ✅ `.gitignore` düzgün konfiqurasiya edilib
- ⚠️ Production üçün şifrələri dəyişdirin!

## 🤝 Töhfə

1. Fork edin
2. Feature branch yaradın: `git checkout -b feature/yeni-xususiyyet`
3. Commit edin: `git commit -m 'Yeni xüsusiyyət əlavə edildi'`
4. Push edin: `git push origin feature/yeni-xususiyyet`
5. Pull Request açın

## 📄 Lisenziya

Bu layihə MIT lisenziyası altındadır.

## 👤 Müəllif

**Ismat Samadov**
- GitHub: [@Ismat-Samadov](https://github.com/Ismat-Samadov)
- Email: ismetsemedov@gmail.com

## 🙏 Təşəkkürlər

- **Neo4j Aura** - Qrafik verilənlər bazası
- **Pinecone** - Vektor verilənlər bazası
- **Ollama** - Local LLM runtime
- **FastAPI** - Modern web framework

---

**Qeyd**: Bu sistem Azərbaycan kənd təsərrüfatı sənədləri üzərində işləyir və Azərbaycan dilində sorğuları dəstəkləyir.
