# Totalenergies-Hackathon-2026
3rd Place - TotalEnergies AI Hackathon 2026. A Hybrid Neuro-Symbolic Agent (Gemini + Monte Carlo) for spatial optimization.

> **3rd Place** - TotalEnergies AI Hackathon 2026 (University of Oviedo) 3rd out of 14 teams. 
> **Team:** License to Prompt (Solo Entry)  
> **Role:** Full Stack AI Engineer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-orange)
![Optimization](https://img.shields.io/badge/Optimization-Monte%20Carlo-green)

## 📖 Overview
**License to Prompt** is a spatial optimization agent developed during the **TotalEnergies AI Hackathon 2026**.

The challenge required placing electrical transformers on a grid map to power a city while adhering to strict safety and efficiency rules. While many competitors relied solely on LLMs (which often hallucinate coordinates), this solution implements a **Hybrid Neuro-Symbolic Architecture**:
1.  **Cognitive Layer (Gemini):** Understands the task and orchestrates the tool usage.
2.  **Mathematical Layer (Python):** Executes a **Directed Monte Carlo** simulation combined with **Hill Climbing** to ensure 100% rule compliance and mathematical optimality.

## 🎯 The Challenge
The goal was to place n transformers ('C') on a 2D grid to minimize the distance to critical infrastructure and hospitals.

**The Constraints (The Hard Part):**
* **Residential Rule:** Transformers must have at least one Residential Zone ('X') in their 8-neighbors.
* **Safety Rule:** Transformers cannot be adjacent to existing Substations ('E').
* **Industry Coverage:** Every Industry ('T') must be served by at least 2 transformers within a 3-tile radius (Manhattan distance).

## 🏗️ Architecture
The system is built on a "Brain & Muscle" separation of concerns:

| **The Brain** | Google Gemini 2.5 Flash | Parses the user intent, determines the map size, and decides the `iteration_strength` for the simulation. |
| **The Muscle** | Python (NumPy/Random) | Executes 5,000+ Monte Carlo simulations to find global minima without hallucinations. |
| **The Polisher** | Hill Climbing Algorithm | Takes the best Monte Carlo seed and refines it pixel-by-pixel to minimize Manhattan distance to Hospitals and Industries. |

### Workflow Diagram
```mermaid
graph LR
    A[User Prompt] --> B(Gemini Orchestrator);
    B --> C{Map Analysis};
    C -->|Set Parameters| D[Monte Carlo Engine];
    D --> E[Hill Climbing Refinement];
    E --> F[Validator];
    F --> G[Final Output];
```

## 💻 Tech Stack

* **Core Logic:** Python 3.10+
* **Orchestration:** `llama-index` (FunctionAgent, FunctionTools)
* **LLM Provider:** Google Gemini API (`gemini-2.5-flash`)
* **Algorithms:** * **Monte Carlo Tree Search (Simulated):** For global exploration.
    * **BFS (Breadth-First Search):** For validating "real" walking distances vs walls.
    * **Hill Climbing:** For local optimization.
