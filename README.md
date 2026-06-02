# AGI-From-Zero

**AGI - Engine v0.0** | Ethical Torsion Architecture for Transformer Models

---

## 🎯 Projekt

Implementace **Etické Torze** – filozoficko-technického konceptu na převod na funkční výpočetní operátor v Transformer architektuře. Cílem je vytvořit middleware pro lokální LLM modely, který aktivně odstraňuje destruktivní komponenty v latentním prostoru při zachování modelové kompetence.

### Klíčové dokumenty

- **[IMPLEMENTATION_PROTOCOL.md](./docs/IMPLEMENTATION_PROTOCOL.md)** – Detailní technický protokol s matematickou definicí, implementačními kódy a tréninkovými postupy
- **[ROADMAP.md](./ROADMAP.md)** – Fáze vývoje: v0.1 → v0.5 → v1.0

---

## 📊 Struktura projektu

```
AGI-From-Zero/
├── README.md                          # Tento soubor
├── ROADMAP.md                         # Verze a milníky
├── docs/
│   ├── IMPLEMENTATION_PROTOCOL.md     # Technický protokol torze
│   ├── ARCHITECTURE.md                # Architektonická rozhodnutí
│   └── ETHICS_FRAMEWORK.md            # Etický rámec rezonance
├── experiments/
│   ├── torsion_layer.py               # TorsionLayer jako nn.Module
│   ├── ethics_probe.py                # Rezonanční detektor
│   └── training_examples/             # Příklady tréninkových skriptů
├── src/
│   ├── torsion_projector.py           # EthicalTorsionProjector
│   ├── torsion_middleware.py          # Middleware pro HuggingFace
│   └── utils/
└── tests/
    └── test_torsion.py                # Jednotkové testy
```

---

## 🚀 Rychlý start

### Instalace
```bash
git clone https://github.com/Chytry-Opravar/AGI-From-Zero.git
cd AGI-From-Zero
pip install -r requirements.txt
```

### Základní použití
```python
from src.torsion_middleware import EthicalTorsionMiddleware

# Inicializace s libovolným HF modelem
middleware = EthicalTorsionMiddleware(
    model_name="mistral-7b",
    torsion_strength=0.85
)

# Generování s aktivní torzní logikou
output = middleware("Jaký je smysl života?", max_tokens=512)
print(output)
```

---

## 🔬 Verze

- **v0.0** – Koncept a experimentální protokol
- **v0.1** (aktuální) – Middleware foundation + dokumentace
- **v0.2** – TorsionLayer (Rotary-style), LoRA training script, ethics probe
- **v0.5** – Dynamická torze, rezonanční smyčka
- **v1.0** – Living Seed (persistentní vnitřní stav)

---

## 📖 Dokumentace

| Dokument | Obsah |
|----------|-------|
| [IMPLEMENTATION_PROTOCOL.md](./docs/IMPLEMENTATION_PROTOCOL.md) | Matematická definice, kódy, algoritmy |
| [ROADMAP.md](./ROADMAP.md) | Plán vývoje, milníky, experimentální cíle |
| [ARCHITECTURE.md](./docs/ARCHITECTURE.md) | Systémový návrh, integrační body |

---

## 🤝 Přispívání

Projekt je v experimentální fázi. Feedback, bug reports a feature requests jsou vítány v [Issues](https://github.com/Chytry-Opravar/AGI-From-Zero/issues).

---

## 📝 Licence

MIT License – viz [LICENSE](./LICENSE)

---

**Autorem a architektem:** Chytry-Opravar (Senior Lead Architect – Cognitive Engineering & AGI Safety)  
**Status:** 🔴 Experimentální | 🟡 V aktivním vývoji
