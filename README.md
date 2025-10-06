# PyTorch FX Model Optimization

This project demonstrates **graph-based model optimizations using PyTorch FX**, including layer fusion, redundant operation elimination, and control-flow rewrites, with correctness validation and performance benchmarking.

---

## 🔹 Project Overview

Deep learning models often contain repetitive patterns and redundant operations that can slow down inference and increase memory usage. This notebook showcases:

1. **Linear → BatchNorm → ReLU fusion**  
   - Combines consecutive layers into a single module  
   - Reduces memory reads/writes and execution time

2. **Redundant operation removal**  
   - Eliminates identity operations (e.g., consecutive transposes)  
   - Keeps the output numerically identical

3. **Control-flow optimization**  
   - Converts Python loops and masked operations into vectorized, graph-friendly tensor operations  
   - Improves GPU kernel fusion

4. **Correctness and benchmarking**  
   - Validates outputs after each pass  
   - Measures inference time and memory usage before and after optimization

---

## 🔹 Technologies Used

- **PyTorch 2.x** (`torch.fx`)  
- **Python 3.10**  
- **Jupyter Notebook**  
- Optional: GPU (CUDA 12.1) for performance benchmarking

---

## 🔹 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/somya2703/fx_optimization.git
   cd fx_optimization
   ```

2. (Recommended) Create a Python environment:

   **Using Conda:**
   ```bash
   conda env create -f environment.yml
   conda activate fx_opt
   ```

   
3. Launch JupyterLab:
   ```bash
   jupyter lab
   ```
4. Open `fx_graph_optimization.ipynb` and run all cells.

---

## 🔹 Expected Outcomes

- Intermediate correctness checks after each optimization pass (fusion, redundant operation removal, control-flow rewrite)  
- Reduced inference time and memory usage for optimized models  
- Outputs remain numerically identical to the original model

---
