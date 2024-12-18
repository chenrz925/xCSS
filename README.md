# Extendable Crowd-Source Splatting (xCSS) 🚀

Welcome to **xCSS**, an extendable framework for 3D Gaussian Splatting (3DGS) reconstruction, evolving from the original **CSS** framework presented in the paper:

> **[CSS: Overcoming Pose and Scene Challenges in Crowd-Sourced 3D Gaussian Splatting](https://arxiv.org/abs/2409.08562)**
> 
> *Runze Chen, Mingyu Xiao, Haiyong Luo, Fang Zhao, Fan Wu, Hao Xiong, Qi Liu, Meng Song*

---

## 🌟 Introduction

**Crowd-Sourced Splatting (CSS)** introduces a novel pipeline for 3D Gaussian Splatting, specifically tailored to address challenges in pose-free scene reconstruction using crowd-sourced imagery. 

Traditional 3D reconstruction techniques often falter due to:
- 🚫 Missing camera poses
- 👁️ Limited viewpoints
- 🌥️ Inconsistent lighting conditions

CSS tackles these challenges by leveraging:
- 📐 Robust geometric priors
- 💡 Advanced illumination modeling

### ✨ Key Features of xCSS
- Enhanced **extendability** for customizations in AR/VR applications.
- Support for **large-scale 3D reconstruction** under complex real-world conditions.
- Improvements in **novel view synthesis** quality, bridging research and practical applications.

---

## 📖 Citing Our Work

If you use **xCSS** in your research or application, please consider citing our foundational paper:

```bibtex
@misc{chen2024cssovercomingposescene,
      title={CSS: Overcoming Pose and Scene Challenges in Crowd-Sourced 3D Gaussian Splatting}, 
      author={Runze Chen and Mingyu Xiao and Haiyong Luo and Fang Zhao and Fan Wu and Hao Xiong and Qi Liu and Meng Song},
      year={2024},
      eprint={2409.08562},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.08562}, 
}
```

---

## 💻 Get Started

1. Clone the repository:
   ```bash
   git clone https://github.com/chenrz925/xCSS.git
   ```

2. Install dependencies:
   ```bash
   source install.sh
   ```

<!-- 3. Run the demo:
   ```bash
   python demo.py --input ./example_data/
   ``` -->

---

## 📂 Project Structure

```plaintext
xcss/
├── CMakeLists.txt      # Build configuration for C++ components
├── install.sh          # Installation script
├── LICENSE             # License file
├── pyproject.toml      # Project configuration for Python
├── README.md           # This file 😎
├── setup.py            # Setup script for Python package
├── src/                # Core implementation of xCSS pipeline
├── tests/              # Unit tests for framework validation
├── weights/            # Pretrained model weights
└── xcss/               # Main module directory
```

---

## 🤝 Contributions

We welcome contributions from the community! Please feel free to:
- Submit bug reports 🐛
- Propose new features 🌟
- Share your usage experience ✨

---

## 📬 Contact

For inquiries or support, please reach out to:
- **Runze Chen** - [GitHub Profile](https://github.com/chenrz925)

---

**Enjoy building with xCSS! 🎉**