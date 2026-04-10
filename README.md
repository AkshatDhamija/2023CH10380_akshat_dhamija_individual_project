# 🎸 Self-Organized Criticality in Rock Music Influence Networks

**Course:** Complexity Science — Individual Project  
**Institution:** Department of Chemical Engineering, IIT Delhi  
**Student:** Akshat Dhamija (2023CH10380)  
**Instructor:** Prof. Rajesh Khanna  
**Submission Date:** April 10, 2026

---

## 🧠 What is this project?

This project applies **self-organized criticality (SOC)** — a concept from statistical physics — to the history of rock music.

The central idea: rock music influence works exactly like a sandpile. Artists slowly accumulate cultural pressure from the outside world. When that pressure crosses a threshold, they "topple" — triggering a cascade of influence across every artist they shaped. Most of the time these cascades are small. Occasionally, one explodes across the entire network. That's how Nirvana happened. That's how the British Invasion happened. It's not luck — it's physics.

We model this using the **Bak–Tang–Wiesenfeld (BTW) sandpile automaton** mapped onto a directed graph of 72 rock artists spanning Robert Johnson (1930s blues) to Foo Fighters and Radiohead (2000s). The simulation shows that cascade sizes follow a **power law** — the defining signature of SOC.

---

## 🗂️ Repository Structure

```
2023CH10380_akshat_dhamija_individual_project/
│
├── data/
│   └── influence_edges.csv      # 63 verified influence edges with citations
│
├── src/
│   ├── network.py               # builds + validates the influence graph
│   ├── simulate.py              # BTW sandpile simulation engine
│   └── analyze.py               # power-law fitting + all figures
│
├── report/
│   ├── main.tex                 # full LaTeX manuscript (journal style)
│   └── main.pdf                 # compiled PDF
│
├── results/
│   └── figures/                 # all generated plots (after running)
│
├── run_all.py                   # single command to run everything
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚡ How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Quick test run** (~1 minute, smaller simulation)
```bash
python run_all.py --quick
```

**3. Full run** (~10 minutes, complete simulation)
```bash
python run_all.py
```

**4. Full run + connectivity sweep** (~25 minutes, generates all figures including Fig. 4)
```bash
python run_all.py --sweep
```

**5. Analysis only** (if simulation already done)
```bash
python run_all.py --analyze-only
```

All figures are saved to `results/figures/`.

---

## 🎯 The SOC Mapping

| BTW Sandpile | Rock Music Analogue |
|---|---|
| Lattice node | Artist / band |
| Sand grain | Unit of cultural influence |
| Grain addition (noise) | Album release, death, breakthrough performance |
| Threshold | Artist's out-degree (number they directly influenced) |
| Toppling | Cascade: artist triggers influence across their successors |
| Avalanche size *s* | Total extent of the cultural cascade |
| Open boundary | Leaf artists with no documented successors |

---

## 📊 Key Results

- Avalanche sizes follow a **power law** `P(s) ~ s^{-1.31}` over two orders of magnitude
- Exponent consistent with BTW theory on scale-free networks (Goh et al. 2003)
- **Connectivity sweep** shows higher connectivity → heavier tails → more catastrophic cascades
- System satisfies all five canonical SOC criteria (Bak 1996, Jensen 1998)

---

## 🗺️ The Influence Network

72 artists, 63 verified edges, spanning five generational layers:

```
1930s Blues Roots  →  1960s British Invasion  →  1970s Hard Rock / Punk
→  1980s Alternative / Metal  →  1990s Grunge / Post-Rock
```

Key hubs (highest out-degree):
| Artist | Directly Influenced |
|--------|-------------------|
| The Beatles | 8 artists |
| Jimi Hendrix | 5 artists |
| The Velvet Underground | 5 artists |
| Led Zeppelin | 4 artists |
| The Ramones | 3 artists |

All edges verified against at least two sources: [AllMusic](https://www.allmusic.com), [Rolling Stone](https://www.rollingstone.com/music/music-lists/100-greatest-artists-147446/), Wikipedia artist pages, and musicological literature (Azerrad 1994, Cross 2001, Shadwick 2003).

---

## 📈 Generated Figures

| File | Description |
|------|-------------|
| `fig_network.png` | Full influence graph, nodes coloured by era |
| `fig_degree_dist.png` | Out-degree distribution (heavy tail) |
| `fig0_evolution.png` | System converging to SOC steady state |
| `fig1_avalanche_dist.png` | Power-law PDF with MLE fit |
| `fig2_ccdf.png` | Complementary CDF confirming power law |
| `fig3_timeseries.png` | Bursty avalanche time series |
| `fig4_connectivity_sweep.png` | Exponent and cascade size vs connectivity |
| `fig5_height_map.png` | Final influence scores per artist |

---

## 📄 Report

The full journal-style manuscript is in `report/main.tex` and `report/main.pdf`.

To recompile:
```bash
cd report
pdflatex main.tex
pdflatex main.tex    # run twice for correct references
```

The report covers all five required sections of the project brief:
- **(4a)** Why rock music is a good SOC candidate
- **(4b)** Formal definitions of noise, avalanche, and connectivity
- **(4c)** Computer simulation methodology
- **(4d)** Power-law analysis and connectivity sweep
- **(4e)** Assessment of the self-organized critical state

---

## 🔧 Dependencies

```
numpy>=1.24
matplotlib>=3.7
scipy>=1.11
networkx>=3.1
```

---

## 🤝 Acknowledgements

- **BTW Sandpile theory:** Bak, Tang & Wiesenfeld (1987), *Physical Review Letters*
- **Power-law fitting:** Clauset, Shalizi & Newman (2009), *SIAM Review*
- **Scale-free networks on SOC:** Goh et al. (2003), *Physical Review Letters*
- **Influence data:** AllMusic, Rolling Stone, Wikipedia
- **AI assistance:** Claude (Anthropic), ChatGPT (OpenAI), Perplexity AI, Google Gemini — see Appendix B of the report for full prompt log

---

## 📜 License

MIT License — free to use and modify with attribution.
