# 游戏智能NPC角色训练

## 介绍

这是一个探索性的项目，目的是测试在当前的游戏中，如果希望NPC具有智能对话的能力，是否可以用现有开源方案的大语言模型实现。我相信ChatGPT3.5和ChatGPT4.0通过微调是可以实现一个可用的智能NPC对象的，但是ChatGPT对于网络游戏存在以下问题，使其暂时还不适合实际生产环境。

1. ChatGPT的finetune和query调用都是有很高成本的，这对于独立游戏或者免费运营的游戏是很大的资金负担。
2. ChatGPT的响应速度和稳定性可能受到大环境影响，不能保持稳定
3. ChatGPT可能存在数据泄露的风险

因此使用独立部署的大语言模型是更加的选择，当然目前的开源的模型无法达到ChatGPT的智能程度，尤其是对于非英文语言，需要有大量的调优工作。

## 项目说明

本项目使用了一款网络游戏的世界观和文案设定，形成了10K的问题集，大约654K汉字量，去调优Vicuna模型，调优在单3090 GPU的机器上运营，因此选用了7B参数量的模型做基础。具体用到的模型如下：

1. [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna) 这是一个优秀的中文低资源的llama+lora方案，这个模型使用了guanaco_belle数据集在vicuna标准模型上做微调，使其支持中文，但因为中文的语料不足够多，使其中文的语言能力仍有缺陷。
2. [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) 这是另一个优秀的中文LLaMA/Alpaca权重模型，在4月28日发布了7B的优化版本，参见：[民间版中文羊驼模型（Plus）v3.0](https://github.com/ymcui/Chinese-LLaMA-Alpaca/releases/tag/v3.0)

## 环境准备

创建项目的可执行环境

```bash
cd $PROJECT_DIR
git clone https://github.com/wangqi/game-npc-vicuna.git
cd game-npc-vicuna
conda create -n gamenpc python=3.10
conda activate gamenpc
pip install -r ./requirements.txt
```

## 基础模型准备

首先制作中文的基础模型，我选择 [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b) 作为基础模型，合并以下中文权重

1. [ziqingyang/chinese-llama-plus-lora-7b](https://huggingface.co/ziqingyang/chinese-llama-plus-lora-7b)
2. [ziqingyang/chinese-alpaca-plus-lora-7b](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-7b)
3. [Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco](https://huggingface.co/Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco)

执行以下脚本完成合并，合并后的模型存放于：`models/huggyllama_chinese-alpaca-plus-lora-7b-vicuna` 目录中，并转换为ggml-f16格式，方便llama.cpp调用。

```bash
./merge_base.sh
```

使用llama.cpp测试模型效果

```bash
./tools/llama -ins --color -m models/huggyllama_chinese-alpaca-plus-lora-7b-vicuna/ggml-f16.bin --repeat_penalty 4

> 你是一个资深导游，你能介绍一下中国的首都吗?
```

## 微调步骤

本项目 `data/data.json` 文件是游戏世界观的训练文件，此游戏是关于少女猎人在被龙兽毁灭的世界上冒险的故事。其中包含了60几名角色和数万字的对话数据。

对项目微调的步骤如下：

1. 合并网络小说训练权重，并测试效果

   ```bash
   ./merge.sh
   ```

   测试

   ```bash
   ./tools/llama -ins --color -m models/game_npc_vicuna_huntress/ggml-f16.bin --repeat_penalty 4
   ```

   

2. 使用`data/data.json`文件对合并后的模型进行微调

   ```bash
   # fine tune on a 4 gpu p3.8xlarge machine.
   # change the GPU settings accordingly
   ./finetune.sh
   
   # fine tune on a single 3090 GPU
   ./finetune_single.sh
   ```

   微调后的LoRA存放在`lora_out`目录下，可以拷贝到 `lora_out/final-7b`或者 `lora_out/final-13b`子目录下保存

3. 将生成的LoRA权重与原版LLaMA模型合并，以便于模型推理和量化
   ```bash
   ./merge.sh [lora_path]
   ```

   其中lora_path是可选参数，默认为 `lora_out/final-7b`

   合并后的模型保存在 `models/game_npc_vicuna`目录下

4. 将合并后的模型转换为ggml FP16格式

   ```bash
   ./convert_ggml.sh [outtype]
   ```

   outtype支持f32, f16, q4_1和q4_0

5. []

## 注意事项

因为Facebook的[LLaMA](https://github.com/facebookresearch/llama)模型禁止商用，所以本项目暂时无法用于商业游戏项目，需要等待Facebook更新授权或者替换其他可商用模型。

## Citation

```
@inproceedings{leng2023chinese-vicuna,
  title={Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model},
  author={Chenghao Fan, Zhenyi Lu and Jie Tian},
  url={https://github.com/Facico/Chinese-Vicuna},
  year={2023}
}
```

```
@article{chinese-llama-alpaca,
      title={Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca}, 
      author={Cui, Yiming and Yang, Ziqing and Yao, Xin},
      journal={arXiv preprint arXiv:2304.08177},
      url={https://arxiv.org/abs/2304.08177},
      year={2023}
}
```