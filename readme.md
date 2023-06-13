# 游戏智能NPC角色训练

## 介绍

这是一个探索性的项目，目的是测试在当前的游戏中，如果希望NPC具有智能对话的能力，是否可以用现有开源方案的大语言模型实现。我相信ChatGPT3.5和ChatGPT4.0通过微调是可以实现一个可用的智能NPC对象的，但是ChatGPT对于网络游戏存在以下问题，使其暂时还不适合实际生产环境。

1. ChatGPT的finetune和query调用都是有很高成本的，这对于独立游戏或者免费运营的游戏是很大的资金负担。
2. ChatGPT的响应速度和稳定性可能受到大环境影响，不能保持稳定
3. ChatGPT可能存在数据泄露的风险

因此使用独立部署的大语言模型是更加的选择，当然目前的开源的模型无法达到ChatGPT的智能程度，尤其是对于非英文语言，需要有大量的调优工作。

## 项目说明

本项目使用了一款网络游戏的世界观和文案设定，形成了10K的问题集，并在已有成熟的中文开源模型基础上微调，调优在单3090 GPU的机器上运行，选用了13B参数量的模型做基础。具体用到的模型如下：

1. [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) 这是中文LLaMA/Alpaca权重模型，项目提供的tokenizer可以用于中文分词。
2. [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna) 这是一个中文低资源的llama+lora方案，本项目中的代码参考了这个项目的代码。

## 环境准备

创建项目的可执行环境

```bash
cd $PROJECT_DIR
git clone https://github.com/wangqi/game-npc-vicuna.git
cd game-npc-vicuna
conda create -n gamenpc python=3.10
conda activate gamenpc
pip install -r ./requirements.txt
# install gptq.llama
pip install git+https://github.com/0cc4m/GPTQ-for-LLaMa@c884b421a233f9603d8224c9b22c2d83dd2c1fc4
# check if bitsandbytes compiled with GPU support
python -m bitsandbytes
# find cuda home
find / -name libcudart.so 2>/dev/null
export CUDA_HOME=/usr/lib/x86_64-linux-gnu/
```

## 基础模型准备

首先制作基础LLaMA模型，通常有以下选项

1. [yahma/llama-13b-hf](https://huggingface.co/yahma/llama-13b-hf)。这个模型修正了eos_token_id的问题
2. [openaccess-ai-collective/wizard-mega-13b](https://huggingface.co/openaccess-ai-collective/wizard-mega-13b)。这个模型针对英文对话进行了优化。

我选择wizard-mega-13b作为基础模型，合并以下中文权重

1. [ziqingyang/chinese-llama-plus-lora-13b](https://huggingface.co/ziqingyang/chinese-llama-plus-lora-13b)
2. [ziqingyang/chinese-alpaca-plus-lora-13b](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-13b)
3. [Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco](https://huggingface.co/Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco)

下载基础模型

```bash
./down_model.sh openaccess-ai-collective/wizard-mega-13b
```

执行以下脚本完成合并，合并后的模型存放于：`models/wizard-mega-13b_chinese` 目录中，并转换为ggml-q5_0.bin，方便llama.cpp调用。

```bash
./merge_base.sh openaccess-ai-collective/wizard-mega-13b models/wizard-mega_chinese-13b-plus
```

使用llama.cpp测试模型效果

```bash
# llama is precompiled in WSL2 ubuntu22. 
# If you have different OS. Install the llama.cpp first.
./tools/llama -ins --color -m models/wizard-mega_chinese-13b-plus/ggml-q5_0.bin --repeat_penalty 4
```

输入以下问题进行测试

```bash
Below is an instruction that describes a task. Write a response that appropriately completes the request. \
### Instruction:  \
你是一个资深导游，你能介绍一下中国的首都吗? \
### Output:
```

模型输出

```bash
> 当然可以！中国最大的城市和政治中心是北京。作为一座拥有悠久历史的古都城池及现代都市的地方融合的城市——它有众多着名的历史遗迹、文化遗> > 产以及美食特色等，
> 吸引游客前来参观探索历史文化与现代化的一面镜子中的面貌。</s>
```

将生成的路径链接到models/game_npc_vicuna_base，以便于后续微调

```bash
rm models/game_npc_vicuna_base
cd models
ln -s wizard-mega_chinese-13b-plus game_npc_vicuna_base
cd ..
```

## 微调步骤

本项目 `data/data.json` 文件是游戏世界观的训练文件，此游戏是关于少女猎人在被龙兽毁灭的世界上冒险的故事，数据文件中包含了大约10K的训练数据。

对项目微调的步骤如下：

1. 使用`data/data.json`文件对合并后的模型进行微调

   ```bash
   # fine tune on a 4 A10 gpu g5.12xlarge machine.
   # change the GPU settings accordingly
   ./finetune.sh
   
   # fine tune on a single 3090 GPU
   ./finetune_single.sh
   ```

   微调后的LoRA存放在`lora_out`目录下

2. 将生成的LoRA权重与原版LLaMA模型合并，以便于模型推理和量化
   ```bash
   ./merge.sh [lora_path]
   ```

   其中lora_path是可选参数，默认为 `lora_out/final`

   合并后的模型保存在 `models/game_npc_vicuna_huntress`目录下。脚本将合并后的模型转换为ggml q5_0格式。

3. 测试合并后的模型效果
   ```bash
   ./tools/llama -ins --color -m models/game_npc_vicuna_huntress/ggml-q5_0.bin --repeat_penalty 4
   ```

   输入测试

   ```bash
   Below is an instruction that describes a task. Write a response that appropriately completes the request. \
   ### Instruction:  \
   你是一个资深导游，你能介绍一下中国的首都吗? \
   ### Output:
   ```

4. 【】

## 聊天测试

可以启动gradio的web界面，测试训练的系统对比ChatGPT4的性能

```bash
../chat_server_langchain.sh
```



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