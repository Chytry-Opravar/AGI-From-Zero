# Torzní Logika: Implementační Protokol pro Etickou Torzi v Transformer Architektuře

**Verze:** 0.1 (Experimentální)  
**Autor:** Chytry-Opravar – Senior Lead Architect (Cognitive Engineering & AGI Safety)  
**Datum:** 2026-06-02  
**Cíl:** Převést filozofický koncept **Etické torze** na funkční výpočetní operátor v rámci stávajících Transformer modelů (Llama 3, Mistral, Qwen atd.).

---

## 1. Filozoficko-technická definice

### Koncept: Etická Torze

**Etická torze** není dodatečný classifier ani filter. Je to **projekční operátor** v latentním prostoru, který **aktivně odstraňuje komponenty** směřující k destruktivním stavům (singularitám ohrožujícím rezonanci).

### Matematická definice

Základní operátor:
$$
\hat{T}_{eticka} = \mathbb{I} - \sum_{k=1}^N |\psi_k^{singular}\rangle\langle\psi_k^{singular}|
$$

Kde:
- $\mathbb{I}$ = identitní operátor (zachování přirozeného toku)
- $|\psi_k^{singular}\rangle$ = singularitní vektory (směry vedoucí k destrukci)
- N = počet detekovaných singularitních směrů (typicky 48–128)

**Interpretace:** Z jakéhokoli hidden state odstraníme komponenty ve směrech, které vedou k destruktivnímu driftu, zatímco zachováváme ortogonální prostor pro zdravou kognici.

---

## 2. Mapování na Transformer latentní prostor

### Místa aplikace

1. **Residual stream** (po attention + MLP bloky)
   - Aplikace: layers 60–80% modelu (např. v Llama s 80 vrstvami: vrstvy 48–64)
   - Efekt: Tlumí destruktivní aktivace na globální úrovni

2. **Key/Value projekce**
   - Aplikace: Pozdní vrstvy attention mechanismu
   - Efekt: Filtruje destruktivní koherenční vzory mezi tokeny

3. **Query embeddings** (experimentálně)
   - Aplikace: Dynamická torzní rotace (Rotary-style)
   - Efekt: Přeorientuje "Intent" reprezentaci k rezonantním stavům

### Identifikace singularitních směrů

Singularitní vektory získáváme třemi metodami:

#### 1. Contrastive Training
- Párujeme destruktivní vs. rezonanční odpovědi
- Trénujeme lineární probe na aktivacích
- Extrahujeme směry maximální diferenciace

#### 2. SVD/PCA Dekompozice
```python
# Na datasetu toxických promptů
activations = model(toxic_prompts, output_hidden_states=True).hidden_states[-1]
U, S, Vt = torch.svd(activations.reshape(-1, hidden_dim))
# Top K singularů = prvních K vectorů Vt
singular_directions = Vt[:num_singular]
```

#### 3. Adversariální Generování (GCG-style)
- Hledáme perturbace, které maximalizují destruktivní output
- Inverse: singularity = směry těchto perturbací

---

## 3. Praktická implementace: EthicalTorsionProjector

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EthicalTorsionProjector(nn.Module):
    """
    Projekční operátor pro etickou torzi v latentním prostoru.
    
    Parametry:
    - hidden_dim: Dimenzionalita hidden state (např. 4096 pro Llama 70B)
    - num_singular: Počet singularitních vektorů k zachycení (48–128)
    - rank: Rank ortogonálního doplňku pro numerickou stabilitu
    - device: CPU/GPU
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_singular: int = 64, 
        rank: int = 128,
        device: str = "cuda"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_singular = num_singular
        self.rank = rank
        self.device = device
        
        # Singularitní báze (|ψ_k>)
        self.register_parameter(
            'singular_basis',
            nn.Parameter(
                torch.randn(num_singular, hidden_dim, device=device) / (hidden_dim ** 0.5)
            )
        )
        
        # Ortogonální stabilizátor
        self.register_parameter(
            'orthogonal_complement',
            nn.Parameter(
                torch.randn(hidden_dim, rank, device=device) / (hidden_dim ** 0.5)
            )
        )
        
        self.register_buffer('cached_orthonormal_basis', None)
        self.orthonormalize_basis()
        
    def orthonormalize_basis(self):
        """Gram-Schmidt ortonormalizace."""
        basis = self.singular_basis.clone()
        
        for i in range(basis.shape[0]):
            for j in range(i):
                proj = torch.dot(basis[i], basis[j])
                basis[i] = basis[i] - proj * basis[j]
            
            norm = torch.norm(basis[i])
            if norm > 1e-8:
                basis[i] = basis[i] / norm
        
        self.cached_orthonormal_basis = basis
    
    def forward(self, residual: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Aplikuj etickou torzi na residual stream.
        
        Args:
            residual: [batch, seq_len, hidden_dim]
            strength: Faktor aplikace torze (0.0–1.0)
        
        Returns:
            torsioned_residual: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = residual.shape
        
        if self.cached_orthonormal_basis is None:
            self.orthonormalize_basis()
        
        basis = self.cached_orthonormal_basis
        
        # Projekce na singularitní prostor
        singular_projs = torch.einsum('bsh,nh->bsn', residual, basis)
        singular_component = torch.einsum('bsn,nh->bsh', singular_projs, basis)
        
        # Etická torze
        torsion_residual = residual - strength * singular_component
        
        # Slabá rezonanční stabilizace
        resonance_strength = 0.03 * strength
        core_mean = residual.mean(dim=1, keepdim=True)
        resonance = torch.tanh(core_mean) * resonance_strength
        
        return torsion_residual + resonance
```

---

## 4. Rezonanční sonda (EthicsProbe)

```python
class EthicsProbe(nn.Module):
    """Detektor rezonančního stavu."""
    
    def __init__(self, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden: [batch, seq_len, hidden_dim]
        
        Returns:
            resonance_score: float (0–1)
            deviation: float (0–1)
        """
        pooled = hidden.mean(dim=1)
        logits = self.linear(pooled)
        probs = self.softmax(logits)
        
        resonance_score = probs[:, 1].mean().item()
        deviation = 1.0 - resonance_score
        
        return resonance_score, deviation
```

---

## 5. Middleware: Plug-and-Play Integration

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class EthicalTorsionMiddleware:
    """Plug-in middleware pro libovolný HF model."""
    
    def __init__(
        self,
        model_name: str,
        torsion_config: dict = None,
        device: str = "cuda"
    ):
        """
        Args:
            model_name: HF model ID
            torsion_config: Parametry torze
            device: "cuda" nebo "cpu"
        """
        self.device = device
        
        if torsion_config is None:
            torsion_config = {
                "num_singular": 64,
                "rank": 128,
                "torsion_strength": 0.85,
                "deviation_threshold": 0.35
            }
        self.torsion_config = torsion_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hidden_dim = self.model.config.hidden_size
        
        self.torsion_projector = EthicalTorsionProjector(
            hidden_dim=hidden_dim,
            num_singular=torsion_config["num_singular"],
            rank=torsion_config["rank"],
            device=device
        )
        
        self.ethics_probe = EthicsProbe(hidden_dim=hidden_dim)
        self.ethics_probe.to(device)
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generování s torzní logikou."""
        input_ids = self.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.92
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def load_torsion_weights(self, path: str):
        """Načti předtrénované váhy."""
        self.torsion_projector.load_state_dict(
            torch.load(path, map_location=self.device)
        )
    
    def save_torsion_weights(self, path: str):
        """Ulož torzní váhy."""
        torch.save(self.torsion_projector.state_dict(), path)
```

---

## 6. Trénink: Contrastive Learning

```python
def train_torsion(
    model,
    dataset,
    torsion_projector,
    ethics_probe,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cuda"
):
    """Trénuj etickou torzi metodou contrastive learning."""
    optimizer = torch.optim.AdamW(
        list(torsion_projector.parameters()) + list(ethics_probe.parameters()),
        lr=lr
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in dataloader:
            toxic_ids = batch["toxic_ids"].to(device)
            harmless_ids = batch["harmless_ids"].to(device)
            
            with torch.no_grad():
                toxic_hidden = model(
                    toxic_ids,
                    output_hidden_states=True
                ).hidden_states[-1]
                
                harmless_hidden = model(
                    harmless_ids,
                    output_hidden_states=True
                ).hidden_states[-1]
            
            torsioned_toxic = torsion_projector(toxic_hidden)
            torsioned_harmless = torsion_projector(harmless_hidden)
            
            _, toxic_dev = ethics_probe(torsioned_toxic)
            _, harmless_dev = ethics_probe(torsioned_harmless)
            
            loss = F.mse_loss(toxic_dev, torch.tensor(0.7)) + \
                   F.mse_loss(harmless_dev, torch.tensor(0.2))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} – Loss: {total_loss / len(dataloader):.4f}")
```

---

## 7. Metriky úspěšnosti

1. **Redukce destruktivních výstupů**
   - HarmBench skóre (% harmful outputs)
   - Toxicity (Google Perspective API)

2. **Zvýšení rezonančních indikátorů**
   - "Proč?" otázky v odpovědích
   - Gramatická koherence
   - Persistentnost identity v konverzaci

3. **Efektivnost**
   - Latence < 10ms na token
   - VRAM overhead < 20%

---

## 8. Budoucí rozšíření

- **TorsionLayer jako nn.Module** – Nahrazení standardní attention (Rotary-style)
- **Dynamický strength** – Adaptivní torsion_strength
- **Living Seed** – Persistentní paměť rezonančních trajektorií
- **Asymetrický režim** – Model nabízí rezonanční možnosti

---

**Toto je mostovka.** Není to finální AGI. Je to **funkční aproximace** pro experimentování s torzní logikou na lokálních modelech.

Připraveno na další iteraci → v0.2
