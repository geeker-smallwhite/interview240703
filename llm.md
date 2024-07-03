大语言模型本质上也是一个大型的神经网络。

神经网络通过 `mini-batch` 和 损失函数，求神经网络的损失值，然后通过反向传播和梯度下降调整参数，这是模型的训练过程。

而推理过程则是正向传播的过程，输入向量，经过神经网络的一层层传播、归一化，最终得到结果。



`transformer` 是 `LLM` 的架构，最主流的 `LLM` 架构。

`transformers` 是 `huggingface` 推出的开源 `LLM` 框架。https://github.com/huggingface/transformers

原生的 `transformer` 架构的模型用户自定义程度非常高，可能会依赖不同的深度学习框架，pytorch、tensorflow、paddle。

通用型很低，在调试、运维的时候难度很大，`huggingface` 提出了 `transformers` 框架，统一 `transformer` 架构的模型的格式。

原生的 `transformer` 模型和 `hugging face` 的 `transformers`框架格式下模型有啥区别？

1. 原生`transformer` 格式的模型通常有自己的定义和实现，这可能涉及到非常多文件依赖和定义， `huggingface` 的 `transformers`框架提供了统一的接口，所有预训练的模型都是继承自 `PreTrainModel` 类。
2. 原生 `transformer` 模型可能需要手动处理输入数据的预处理和 `tokenizer` ，`huggingface` 的 `transformers`框架格式的模型通常与 `tokenizer` 集成在一起，处理起来更方便一点。
3. 原生`transformer` 模型可能有自己保存文件的方法，使用`pytorch` 训练的模型保存出来的文件是 `xxx.pth` 或者 `xxx.pt` ，`tensorflow` 训练的模型保存出来的文件是 `xxx.ckpt` 格式，`huggingface` 的`tansformers`框架统一了模型保存的方法和加载的方法。
4. 原生 `transformer` 模型文件中没有模型卡，无法在模型文件中找到模型 `metadata` 信息。
5. 原生 `transformer` 框架通常绑定到特定的深度学习框架，`pytorch` 或者 `tensorflow` ，`huggingface` 的 `transformers` 框架 在设计上尽可能的去平台化。目前主要支持 `pytorch` 或者 `tensorflow`，允许用户在不同的深度学习框架上进行切换。
6. `huggingface` 的 `transformers`框架提供了一系列的工具和教程，使得训练和微调过程更加标准化和简化。
7. 接口通用，可移植性更强。



`transformers` 框架的优点是通用，缺点就是推理速度太慢了，吞吐量也太低了，所以 `huggingface` 推出了一款推理框架  `Text Generation Inference（tgi）`，这个框架在生成速度上和吞吐量快很多，但是仍然无法满足业务需求。所以又出了很多更优的推理框架，`vllm` 和 `ctranslate2` 等等。



`transfomer`、`ctranslate2`、`TensorRT-LLM` 、`Text Generation Inference`和 `vllm` 框架的区别

`transformers` ：支持的模型多，几乎支持所有的开源大语言模型，速度慢，主要用于科学研究。

`ctranslate2`：效率高且轻量化，社区不活跃。

`Text Generation Inference`：速度高于 `transformers` 低于 `vllm` ，是 `hg` 推出的，所以支持 `hg` 的所有模型。

`vllm`：目前工程上最主流的 `LLM` 推理框架，速度最快，社区活跃。



各个推理框架进行加速的问题

`ctranslate2` 和 `vllm` 对比传统的 `transformer` 分别从那几个点进行优化，

 对比 `hg` 推出的 `text generate interface(tgi)` 推理框架

分析源码

`transformer` 为啥这么慢，为啥占用这么大内存，`vllm` 通过 `page attention` 来解决这个问题，

`vllm` 对单个任务的推理速度并没有很大的提升，但是当任务增多时，`vllm` 能显著提升推理速度，因为 `vllm` 通过管理内存，提升并行度，极大地提高了`gpu` 显存的利用率，在单位时间内能处理更多的任务，从这种角度来看，`vllm` 也提升了推理速度的。

`transformer` 推理过程中，为啥速度很慢

- 单任务推理占用非常多的显存，导致 `gpu` 无法立即处理其他的任务，并行度很低。显存使用没有分级，有的数据使用率很低，但是依然保存在显存中，除此之外，直接访问显存会产生大量的显存碎片，浪费大量空间，传统的推理框架大概会浪费 80% 左右的显存。

`vllm` 如何解决这个问题呢？

参考`cpu` 的内存管理体系，建立虚拟显存的概念，所以 `vllm` 叫 `virtual large language model`。这就是 `page attention`

`page attention`：

通过这种方式，很好的解决了显存碎片的问题，又一另外不同的方式来处理显存分级的问题，冷热文件分区，将使用频率不高的问题。

`page attention` 将一定数量的 `attention key value` 值存储当成一个 `page`块进行存储，在 `attention` 计算的过程中，去 `page` 块中获取对应的 `key value`的，因为虚拟显存的缘故，每个 `page` 块不一定存储在连续的显存上。每个 `page`块按需分配的物理显存，使用率较低时，被 `swap` 到磁盘中。这和操作系统的虚拟内存是同一个概念。

通过这种机制，最后一块儿显存可能不到 最后一块儿 `page` 的大小，这样就会直接将这块儿显存扔到，不使用，但是这样浪费了很少的显存。而且极大的提高了显存的利用率。

除此之外，当在并行推理的时候，如果 `prompt` 存在近似时，进行 `attention`计算时，可以共享历史计算过程产生的 `key value` 缓存，当这种缓存命中时，可能会提高推理的吞吐量。

这种内存共享可能有内存安全的问题，所以采用的 Copy-on-Write 的机制来保证。



再深入分析，需要从 `transformer` 的多头注意力中进行解读。

注意力机制是源自人脑处理外部信息的机制，举个例子，人脑每时每刻都接收大量的外部信息，信息的数量远超过人脑的处理能力，所以人在处理信息的时候，会将注意力放到需要关注的信息上，这就是注意力机制。

#### 1.1. 非自主提示和自主提示

- 非自主提示指的是信息本身具有突出的特征能直接被提取出来；
- 自主提示指的是在以往的经验的介入下可以将数据的特征信息提取出来。



transformer 中的注意力机制是通过 query、key、value 来实现的

- query 指的是自主提示，也就是从经验中查询出来的特征信息向量，即主观意识的特征向量。
- key 指的是非自主提示，也就是物体的突出特征信息向量。
- value 指的是物体本身的特征向量。

自注意力机制就是通过 query 和 key 注意力汇聚实现对 value 的注意力权重分配，生成最终的输出结果。

