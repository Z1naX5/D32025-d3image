### 预期解
此题的预期解是选手能够意识对网络模型进行逆向建模，只需要正确写出反向的网络与运算规则，把题目给出的模型加载进新的框架即可得到flag。
在这样的情况下，选手端需要的资源并不多，仅需满足以下环境：
- Python 解释器版本需与题目要求的 PyTorch 版本匹配
- 其余缺失库选择pip默认安装版本即可
- cuda并非必须的，你可以把所有的张量都移动到cpu上运算。

在收到的 wp 中，很高兴大部分选手都使用了正确的思路，下面详细的解释一下这个思路的实现与可行性。

首先，分析网络框架：
![20250601174309](https://raw.githubusercontent.com/Z1naX5/PicGo/master/images/20250601174309.png)

![20250601174613](https://raw.githubusercontent.com/Z1naX5/PicGo/master/images/20250601174613.png)

每个block大致如上图，标注的文字信息与代码截图对应。
- 蓝色部分：基础张量运算（均存在数学逆运算）；
- 绿色箭头：输入输出变换（方向固定，不可逆）；
- 黑色箭头：可逆运算路径。

![20250601174825](https://raw.githubusercontent.com/Z1naX5/PicGo/master/images/20250601174825.png)

根据如上思路，我们只需要在 model\.py，d3net\.py 和 block\.py 中作如下修改即可：

![20250601175335](https://raw.githubusercontent.com/Z1naX5/PicGo/master/images/20250601175335.png)

![20250601175235](https://raw.githubusercontent.com/Z1naX5/PicGo/master/images/20250601175235.png)

![20250601175216](https://raw.githubusercontent.com/Z1naX5/PicGo/master/images/20250601175216.png)

其次，dwt 与 iwt 在图像处理中本身就互为一对逆运算。

最后，逆向网络还需要 y2 ,在题目附件中提供了一个名为 auxiliary_variable 的函数来辅助完成。经实验与数学验证，y2 可为任意与 y1 同形状的张量，不影响最终结果。

修改后的decode如下：
```python
def decode(steg):
    output_steg = transform2tensor(steg)
    output_steg = dwt(output_steg)
    backward_z = gauss_noise(output_steg.shape)

    output_rev = torch.cat((output_steg, backward_z), 1)
    bacward_img = d3net(output_rev, rev=True)
    secret_rev = bacward_img.narrow(1, 4 * 3, bacward_img.shape[1] - 4 * 3)
    secret_rev = iwt(secret_rev)

    image = secret_rev.view(-1) > 0

    candidates = Counter()
    bits = image.data.int().cpu().numpy().tolist()
    for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
        candidate = bytearray_to_text(bytearray(candidate))
        if candidate:
            candidates[candidate] += 1
    if len(candidates) == 0:
        raise ValueError('Failed to find message.')
    candidate, count = candidates.most_common(1)[0]
    print(candidate)
```

### 预期中的非预期解
当然，由于这道题的定位是`趣味`，所以在这种需要看出可逆向的结构之外，亦可基于以下方法求解（力大砖飞需更高算力，但笔记本电脑可胜任）：

具体做法是，选手自行选一个图像数据集，通过题目模型加密生成“密文-明文”配对数据；然后以加密数据为输入，原始数据为标签，训练一个自定义解密模型。

这个方法的可行性在于，由于题目最终输出为二进制形式，解密问题被简化为二分类任务，再加上校验码可以降低对最后精确度的要求，所以对解密网络的训练难度与网络深度要求大大降低。

如果题目是把图片隐写到图片里，那么训练一个解密网络的难度将会大大增加，因为本意只是想在一天内给选手们一个有趣的简单轻量AI题，专注在网络本身，而不是演变为算力比拼或者过于脑电波，所以在任务上没有继续加深难度，也给出了大部分的代码。

非常遗憾也因为题目代码量较小，选手可以使用 AI 来理解工程代码，并快速的得到解答。但还是希望选手能够从中感受到题目本身“不需要训练 AI 的 AI 题”的趣味性。

参考：
[HiNet: Deep Image Hiding by Invertible Network](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html?ref=https://githubhelp.com)
[SteganoGAN: High Capacity Image Steganography with GANs](https://arxiv.org/abs/1901.03892)