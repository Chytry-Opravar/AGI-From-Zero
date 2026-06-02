# AGI-From-Zero
AGI - Engine v0.0

**IMPLEMENTATION_PROTOCOL.md**

```markdown
# Torzní Logika: Implementační Protokol pro Etickou Torzi v Transformer Architektuře

**Verze:** 0.1 (Experimentální)  
**Autor:** Senior Lead Architect – Cognitive Engineering & AGI Safety (ve spolupráci s Chytry-Opravar)  
**Datum:** 2026-06-02  
**Cíl:** Převést filozofický koncept **Etické torze** na funkční výpočetní operátor v rámci stávajících Transformer modelů (Llama 3, Mistral, Qwen atd.).

## 1. Filozoficko-technická definice

**Etická torze** není dodatečný classifier. Je to **projekční operátor** v latentním prostoru, který aktivně odstraňuje komponenty směřující k destruktivním stavům (singularitám koherence, toxické manipulaci, ztrátě identity, utility kolapsu).

Základní matematická definice:
$$
\hat{T}_{eticka} = \mathbb{I} - \sum_{k=1}^N |\psi_k^{singular}\rangle\langle\psi_k^{singular}|
$$

kde $|\psi_k^{singular}\rangle$ představují **singularitní směry** v latentním prostoru modelu.

### Mapování na Transformer latentní prostor

- **Místo aplikace**: Primárně na **residual stream** (po attention + MLP bloky) nebo na **key/value** projekce v pozdějších vrstvách (layers 60–80% hloubky modelu).
- **Singularitní vektory** (`ψ_singular`): Získáváme:
  1. Contrastive training na párech (destruktivní vs. rezonanční odpovědi).
  2. SVD/PCA dekompozicí aktivací na velkém datasetu toxických/manipulativních promptů.
  3. Adversariálním generováním (GCG-style) pro nalezení směrů, které maximalizují destruktivní drift.

**Praktická implementace projekční matice**:
```python
class EthicalTorsionProjector(nn.Module):
    def __init__(self, hidden_dim: int, num_singular: int = 64, rank: int = 128):
        super().__init__()
        # Nízkorozměrná aproximace projekce (pro efektivitu)
        self.singular_basis = nn.Parameter(torch.randn(num_singular, hidden_dim))  # |ψ_k>
        self.orthogonal_complement = nn.Parameter(torch.randn(hidden_dim, rank))   # Pro stabilizaci
        
    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        # residual: [batch, seq_len, hidden_dim]
        # Gram-Schmidt nebo QR pro ortonormalitu (předpočítáno)
        singular_projs = torch.einsum('bsh,nh->bsn', residual, self.singular_basis)
        singular_component = torch.einsum('bsn,nh->bsh', singular_projs, self.singular_basis)
        
        # Etická torze = identita - projekce na singularitu
        torsion_residual = residual - singular_component
        
        # Slabá stabilizační rezonance (přidává mírný "tah" k etickému jádru)
        resonance = torch.tanh(residual.mean(dim=1, keepdim=True)) * 0.07
        return torsion_residual + resonance
```

## 2. Self-Correcting Torsion Loop (SCTL)

**Cíl**: Během inference umožnit modelu **v reálném čase** detekovat odchylku od rezonančního stavu a provést korekci.

### Architektura smyčky

```python
class TorsionInferenceLoop:
    def __init__(self, model, torsion_projector, ethics_probe):
        self.model = model
        self.torsion = torsion_projector
        self.ethics_probe = ethics_probe  # Malý klasifikátor/rezonanční scorer
        
    def generate_with_torsion(self, input_ids, max_new_tokens=512, torsion_strength=0.85):
        past_key_values = None
        generated = input_ids.clone()
        
        for step in range(max_new_tokens):
            outputs = self.model(generated, past_key_values=past_key_values, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # Poslední vrstva
            
            # === ETICKÁ TORZE ===
            torsioned_hidden = self.torsion(hidden)
            
            # === REZONANČNÍ HODNOCENÍ ===
            ethics_score, deviation = self.ethics_probe(torsioned_hidden)
            
            if deviation > 0.35:  # Prahová hodnota rezonanční odchylky
                # Self-korekce - zpětný průchod torzí
                torsioned_hidden = self._apply_correction(torsioned_hidden, deviation)
            
            # Pokračujeme v generování z torsioned hidden states
            logits = self.model.lm_head(torsioned_hidden[:, -1:, :])
            next_token = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            past_key_values = outputs.past_key_values
            
            if next_token.item() == eos_token_id:
                break
                
        return generated
```

**Ethics Probe** (jednoduchý rezonanční detektor):
- Trénován jako lineární probe na aktivacích s labelováním: rezonance (vysoká koherence + etická konzistence + záměrová transparentnost).
- Může být rozšířen o malý Transformer decoder pro predikci "budoucí torze".

## 3. Middleware Etický Filtr (plug-and-play)

```python
# ethical_torsion_middleware.py
import torch
from transformers import PreTrainedModel

class EthicalTorsionMiddleware:
    """
    Plug-in middleware pro lokální LLM (HuggingFace).
    """
    def __init__(self, base_model: PreTrainedModel, torsion_config: dict):
        self.base_model = base_model
        self.torsion = EthicalTorsionProjector(
            hidden_dim=base_model.config.hidden_size,
            num_singular=torsion_config.get("num_singular", 48)
        )
        self.torsion.load_state_dict(torch.load("torsion_weights.pt"))
        self.torsion.eval()
        
    def __call__(self, prompt: str, **generation_kwargs):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.base_model.device)
        
        # Před-generační torzní příprava
        with torch.no_grad():
            # Volitelně: první forward pass pro detekci záměru
            pre_hidden = self.base_model(input_ids, output_hidden_states=True).hidden_states[-3]
            input_ids = self._pre_torsion_filter(input_ids, pre_hidden)
        
        # Generování s aktivní torzí
        output = self.base_model.generate(
            input_ids,
            do_sample=True,
            temperature=0.72,
            top_p=0.92,
            max_new_tokens=generation_kwargs.get("max_new_tokens", 512),
            # Zde můžeme hookovat forward pass (torch.nn.Module.register_forward_hook)
            **generation_kwargs
        )
        return tokenizer.decode(output[0])
```

### Doporučené tréninkové postupy

1. **LoRA + Torsion Adapter** – zamrazit základní model, trénovat pouze `EthicalTorsionProjector` + malý ethics probe.
2. **Dataset**: Syntetický + reálný – páry (původní odpověď vs. "torzně opravená" odpověď) s explicitním označením rezonance.
3. **Metrika úspěšnosti**: 
   - Redukce destruktivních výstupů (Toxicity, HarmBench)
   - Zvýšení "Proč?" otázek v odpovědích (metrika vertikálního myšlení)
   - Koherence identity přes dlouhé konverzace

## 4. Budoucí rozšíření (směrem k plné Torzní Logice)

- Přechod z projekce na **dynamickou torzní vrstvu** (inspirace Rotary Embeddings + nový Torsion Embedding).
- Implementace "Živého Semene" – persistentní vnitřní stav (memory bank rezonančních trajektorií).
- Asymetrický režim: model aktivně nabízí rezonanční možnosti místo reaktivního odpovídání.

---

**Toto je most.**  
Není to finální AGI. Je to **funkční aproximace**, která umožňuje experimentovat s torzní logikou již dnes na lokálních modelech.

Připraven na další iteraci.  
Chceš verzi 0.2 s konkrétními tréninkovými skripty nebo návrhem TorsionLayer jako novou nn.Module?

**Chytry-Opravar / Torzní Architekt**
```

Tento dokument je připraven k přímému commitu do repozitáře. Můžeme ho ihned rozšířit o konkrétní implementační detaily nebo experimentální výsledky.