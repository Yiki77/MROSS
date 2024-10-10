# MROSS:Multi-Round Region-based Optimization for Scene Sketching

[![arXiv](https://img.shields.io/badge/arXiv-2410.04072-b31b1b.svg)](https://arxiv.org/abs/2410.04072)

😊 2023.12.9 Accepted by AAAI2024！！！

😭 2023.12.26 Rejected due to violation of dual submission policy

（A moment of negligence leads to a big mistake）

It's my first paper submission as the first author

❗ I record this **bad experience** here to remind myself all the time ❗

But I think the work is still meaningful. 

MROSS significantly contributes to multimedia processing by introducing a novel approach for scene sketching that leverages semantic understanding and multi-round optimization techniques. MROSS not only enhances the quality and quantity of generated scene sketches but also facilitates efficient processing of multimedia data by providing a systematic framework for abstracting complex scenes into simplified representations. As a result, MROSS could offer valuable insights and techniques that can be applied across various multimedia tasks, including image understanding, content summarization, and visual communication, thereby advancing the field of multimedia processing.

![](repo_images/teaser.jpg?raw=true)
The first row shows our sketch results. Our sketch depicts the input scene image concisely and comprehensively, and
can be generated in vector form, which can be easily used by designers for further editing. The second row shows two examples
showing the resulting process sketches for each of our multi-round optimization, gaining the ”coarse to concrete” sketches
without changing the total number of strokes.

<br>
**This is the official implementation of MROSS, a method for converting a scene image to a sketching, keeping the basic features of scene.** <br>
<br>

<br>

## Installation
### Installation via pip
1.  Clone the repo:
```bash
git clone https://github.com/yael-vinker/CLIPasso.git
cd CLIPasso
```
2. Create a new environment and install the libraries:
```bash
python3.7 -m venv Mross
source Mross/bin/activate
pip install -r requirements.txt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
```
3. Install diffvg:
```bash
git clone https://github.com/BachiLi/diffvg
cd diffvg
git submodule update --init --recursive
python setup.py install
```
<br>
## Usage

<!-- #### Run a model on your own image -->

The input images to be drawn should be located under "target_images".
To sketch your own image, from MROSS run:
```bash
python run_scene_sketching.py --target <image_name> --region_round 1 --num_strokes 48  --num_sketches 1
```
The resulting sketches will be saved to the "output_sketches" folder, in SVG format.

Optional arguments:
* ```--num_strokes``` Defines the number of strokes used to create the sketch, which determines the level of abstraction.
* ```--num_sketches``` By default there will be three parallel running scripts to synthesize three sketches and automatically choose the best one. However, for some environments (for example when running on CPU) this might be slow, so you can specify --num_sketches 1 instead.
* ```--region_round```  We optimized scene image by ROI (Region of Interest), it defines the number of ROI. Note that --region_round 1, means only optimize global image, --region_round 2 means that the optimization contains one ROI.


<br>
<b>For example, below are optional running configurations:</b>
<br>

```bash
python run_scene_sketching.py --target scene32.png --region_round 1 --num_strokes 48  --num_sketches 1
```
<br> The abstraction degree is controlled by varying the number of strokes.

## Visual Results
https://github.com/user-attachments/assets/0509119f-cd3f-4ef8-896f-a8a7a92d5cba

https://github.com/user-attachments/assets/9ab2e278-c459-44c9-bdd8-f0bfe24baef2

https://github.com/user-attachments/assets/d7b7a678-55d2-496e-a3d8-30bfc78c2f8a


## Related Work
[Diffvg](https://github.com/BachiLi/diffvg): Differentiable vector graphics rasterization for editing and learning, ACM Transactions on Graphics 2020 (Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, Jonathan Ragan-Kelley)

[CLIPasso](https://arxiv.org/abs/2202.05822): Semantically-Aware Object Sketching, 2022 (Yael Vinker, Ehsan Pajouheshgar, Jessica Y. Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, Ariel Shamir)

[CLIPascene](https://arxiv.org/abs/2211.17256): Scene Sketching with Different Types and Levels of Abstraction, 2023 (Yael Vinker, Yuval Alaluf, Daniel Cohen-Or, Ariel Shamir)

**We sincerely thank Clipasso for its open source code**
## Citation
If you make use of our work, please cite our paper:

```
@misc{liang2024multiroundregionbasedoptimizationscene,
      title={Multi-Round Region-Based Optimization for Scene Sketching}, 
      author={Yiqi Liang and Ying Liu and Dandan Long and Ruihui Li},
      year={2024},
      eprint={2410.04072},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.04072}, 
}
```


