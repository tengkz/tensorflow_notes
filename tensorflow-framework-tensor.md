## 目录
1. 核心概念
2. tensor
3. tensor_reference
4. tensor_shape
5. tensor_slice
5. protos

## 1. 核心概念
TF的核心数据结构Tensor表示一个张量，它基于eigen3库，并提供了丰富的API。为了方便引用张量的底层数据，设计了TensorInference类。TensorShape用于表示张量的形状和数据类型等信息，TensorSlice用于表示张量的索引。

## 2. tensor
TF全称叫做TensorFlow，可见tensor的重要性。TF中的tensor基于eigen3库，是对多维数据的一个封装。Tensor类包含的数据成员非常简单：
```
class Tensor {
    //...
  private:
    TensorShape shape_;
    TensorBuffer* buffer_;
}
```
顾名思义，一个是张量的形状，一个是指向底层数据的指针。Tensor作为一个核心数据结构，必然提供了很多API接口，比如常规的构造、析构、赋值、复制、数值属性获取等。除此之外，还提供了两类比较特殊的接口，我们举例说明：
```
class Tensor {
  public:
    //...
    //与proto数据的相互转化
    bool FromProto(const TensorProto& other);
    void AsProtoField(TensorProto* proto);
    //为底层数据创建新视图
    template <typename T> typename TTypes<T>::Vec vec();
    template <typename T> typename TTypes<T>::Matrix matrix();
    template <typename T> typename TTypes<T, NDIMS>::Tensor tensor();
}
```
其中第一类将Tensor与序列化的proto之间相互转化，在设备间相互传递Tensor时，需要先将其序列化。第二类是为当前的Tensor的底层数据提供另外一种视图，我们重点来说一下视图的概念。
回顾Tensor包含的私有数据，TensorBuffer* buffer_是一个指向底层数据的指针，关于它的结构在下文中会详细说明。也就是说，Tensor并不包含实际的底层数据，它实际上只是对底层数据的一种视图。同样一份底层数据，可以提供多种视图。比如对于一个长度为12的数组，如果把它看做向量，它是一个1x12的向量，如果把它看作矩阵，可以认为是3x4或者2x6的矩阵，如果把它当作张量，可以认为是3x2x2的张量。通过这种方法，我们可以对同一份底层数据进行复用，避免了重复申请内存空间，提升了效率。numpy中对多维数组的实现，也是同样的道理。
接下来我们看一下TensorBuffer到底是什么样的结构。找到它的定义，发现它只是一个继承自引用计数类的虚拟接口，不包含任何实现：
```
class TensorBuffer : public core::RefCounted {
    //...
}
```
因此怀疑，TensorBuffer只是一个提供接口的基类，实际上能用的只是它的子类。我们看下它的继承结构：
```
class BufferBase : public TensorBuffer {
    //...
}
class Buffer : public BufferBase {
    //...
  private:
    T* data_;
    int64 elem_;
}
```
结构已经非常清晰了，BufferBase类继承自TensorBuffer，它除了包含一个内存分配器指针外，还对基类中的部分API进行了实现。而Buffer类是实际可用的，它包含了指向实际数据的指针data_以及元素数量elem_。
另外还要说明一点，Buffer除了申请内存之外，还能调用目标类的构造和析构函数，初始化Buffer的内容，TF为此设计了很多辅助类和函数，这里就不一一赘述了。

## 3. tensor_reference
Tensor类的对象除了包含指向底层数据的指针外，还包含了对数据形状和类型的描述，如果我们并不关心这些，直接使用Tensor会增加构建或者移动的负担。因此TF推出了tensor_reference这个类，它仅包含了一个指向TensorBuffer的指针，并且每增加一个TensorReference对象，就会增加一个针对底层TensorBuffer的引用计数。因此针对TensorReference来说，我们唯一能做的就是在用完之后Unref掉，否则会造成内存泄漏。

## 4. tensor_shape
TensorShape相关的核心类继承体系如下：
```
graph LR
TensorShape-->TensorShapeBase
TensorShapeBase-->TensorShapeRep
```
首先来看一下，最底层的TensorShapeRep的私有数据成员：
```
class TensorShapeRep {
    //...
  private:
    union {
        uint8 buf[16];
        Rep64* unused_aligner;//除了强制u_与指针对齐外，没有任何作用
    } u_;
    int64 num_elements_;
    }
}
```
buf这个数组很有意思，它的前12个元素用来存储形状，虽然Tensor最高能支持到256维的张量，但最常用的不超过3维，为了效率，TF提供了三种利用这12个字节的方式，如下：
```
struct Rep16 {
    uint16 dims_[6];//最多可表示6维的张量，每一维的长度不超过2^16-1
};
struct Rep32 {
    uint32 dims_[3];//最多可表示3维的张量，每一维的长度不超过2^32-1
};
struct Rep64 {
    gtl::InlinedVector<int64, 4>* dims_;//支持任意维度的张量
};
```
剩下的4个字节也不能浪费，在第14-16个字节中，分别存储了张量中的数据类型编号、张量的维度数目、张量维度的表示类型（Rep16, Rep32, Rep64）。由于张量维度的数目是用一个字节存储的，因此最多支持256维。可惜笔者目前仍没有发现第13个字节的作用，有发现的读者欢迎告知我。
TensorShapeBase类并没有添加额外的数据成员，它只是添加了一些允许我们修改张量维度的API接口。
最后再来看下PartialTensorShape类，在构造一个张量的形状时，如果对于某些维度我们还不知道具体的维度值，可以把这个维度设为未知，因此就会用到PartialTensorShape类，这个类中也包含了一些未知维度操作的API，这里就不详述了。

## 5. tensor_slice
TensorSlice类表示一个张量的索引，它的数据结构非常简单：
```
class TensorSlice {
    //...
  private:
    gtl::InlinedVector<int64,4> starts_;
    gtl::InlinedVector<int64,4> lengths_;
}
```
分别是每一个维度索引的开始位置和索引长度，由此我们也知道，TF对Tensor只支持连续索引，不支持间隔索引。
由于TensorSlice用途广泛，对其进行初始化的方法也多种多样，包括：
- 创建空索引
- 从单个维度创建（当创建全索引时）
- 从一个整数对数组创建
- 从一个TensorSliceProto创建
- 从一个字符串描述中创建

## 6. protos
为了方便对张量和与之相关的数据结构进行序列化，TF设计了很多protos，理解起来相对简单，现只说明下它们的用途，感兴趣的读者可以去看源代码。
```
message TensorDescription;//张量的描述，包括数据类型、形状、内存分配信息
message TensorProto;//张量的数据类型，版本，原始数据等
message VariantTensorDataProto;//对DT_VARIANT类型的序列化表示
message TensorShapeProto;//张量形状
message TensorSliceProto;//张量索引
```