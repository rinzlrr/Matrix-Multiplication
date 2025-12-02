# HPC Lab 2 – Hybrid MPI + OpenMP Matrix Multiplication  
Parallel Matrix Multiplication on Titan Supercomputer & Local Machine

## 📌 Overview
This project implements and benchmarks three hybrid **MPI + OpenMP** matrix-multiplication programs written in C.  
The goal was to evaluate runtime scalability across:

- **Local machine:** Apple Silicon M3 MacBook Pro  
- **Titan Supercomputer:** SB compute nodes (multi-CPU, multicore)  
- Matrix sizes up to **6000 × 6000**

The analysis compares how performance changes as we scale **nodes**, **cores per node**, and **OpenMP parallelization strategy**.

---

## ⚙️ Implementations Included

### **1. OMP (Baseline Hybrid Version)**
- MPI handles data distribution (scatter + broadcast + gather)
- OpenMP parallelizes the innermost dot-product loop  
- File: `mmmpiOMP.c`

### **2. OMP2 (Improved Inner-Loop Parallelization)**
- More efficient OpenMP region placement  
- Reduced thread scheduling overhead  
- File: `mmmpiOMP2.c`

### **3. OMP3 (Outer-Loop Parallelization — Fastest)**
- OpenMP parallelizes at the row level  
- Maximizes concurrency for large matrices  
- Best performance across all Titan runs  
- File: `mmmpiOMP3.c`

---

## 🧠 Key Concepts Demonstrated
- Hybrid **MPI + OpenMP** programming  
- Distributed-memory row partitioning  
- Shared-memory multithreading inside compute kernels  
- Cache and NUMA behavior in HPC nodes  
- Strong scaling across nodes and cores  
- SLURM job scheduling and cluster execution  

---

## 🧪 Experimental Setup

### **Systems Used**

#### **MacBook Pro (M3, 2023)**
- High per-core performance  
- Unified memory  
- Excellent for small matrix benchmarks  

#### **Titan SB Node**
- Dual Intel Xeon CPUs  
- 40+ cores per node  
- DDR4 memory subsystem  
- Ideal for large distributed workloads  

### **Matrix Sizes Tested**
- 1000 × 1000  
- 2000 × 2000  
- 4000 × 4000  
- 6000 × 6000  

### **Parallel Configurations (Titan)**
Nodes × Cores per Node:
- 1×1, 1×2, 1×4, 1×8, 1×16  
- 8×8, 8×16  
- 16×8, 16×16  

---

## 🚀 Fastest Configurations (OMP3)
Used with:

- **8 nodes × 16 cores**
- **16 nodes × 16 cores**

---

## 💻 Laptop vs 🖥️ Titan
**The M3 laptop outperformed Titan on smaller matrices (≈1000×1000) due to:**

- Lower MPI overhead  
- Faster unified memory  
- Higher per-core performance  
- Titan’s communication startup cost dominating small workloads  

---

## 🧵 OMP2 vs OMP3
- **OMP2** → Minimal improvement beyond ~8 cores  
- **OMP3** → Truly scalable; best performance for large matrices  

---

## ▶️ Compilation & Execution (Titan)

### **Compile**
```bash
mpiicx -O2 -qopenmp mmmpiOMP.c  -o mmmpiOMP
mpiicx -O2 -qopenmp mmmpiOMP2.c -o mmmpiOMP2
mpiicx -O2 -qopenmp mmmpiOMP3.c -o mmmpiOMP3
```

## ▶️ Run with SLURM

**Example — 8 nodes × 16 cores per node:**

```bash
sbatch --partition=sb -n 8 --ntasks-per-node=1 -c 16 \
       mmmpi.bat mmmpiOMP3 6000 1
```

**Program Arguments**
  - Executable name
  - Matrix size
  - Transpose flag (1 = enabled)

## 📝 Deliverables Included
- Parallel C source code
- Full performance analysis report
- Charts & benchmark tables
- Final PowerPoint summary presentation

## 🎯 What This Project Demonstrates
- Ability to write optimized parallel C
- Deep understanding of MPI communication
- Efficient use of OpenMP multithreading
- Real HPC experience on a multi-node cluster
- Strong-scaling and performance interpretation
- Memory hierarchy & cache-aware reasoning
- Clean benchmarking and reproducible experiments
