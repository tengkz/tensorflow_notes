把framework中剩余的内容，按照文件名进行了简单解析。时间原因写的很仓促，算是占个坑，后面有了新的理解再来补充。

## allocation_description.proto
一个对单次内存分配结果进行信息描述的proto。

## attr_value
之前在讲op的时候提到过，操作是有参数的。而AttrValue表示的就是参数的值。先看一下它的proto定义：
```
message AttrValue {
    message ListValue {
        repeated bytes s = 2;
        repeated int64 i = 3;
        repeated float f = 4;
        repeated bool b = 5;
        repeated DataType type = 6;
        repeated TensorShapeProto shape = 7;
        repeated TensorProto tensor = 8;
        repeated NameAttrList func = 9;
    }
    
    oneof value {
        bytes s = 2;
        int64 i = 3;
        float f = 4;
        bool b = 5;
        DataType type = 6;
        TensorShapeProto shape = 7;
        TensorProto tensor = 8;
        ListValue list = 1;
        
        //func代表一个函数，func.name代表一个函数的名称，或者核心操作的名称，func.attr.first是为函数定义的参数的名称，func.attr.second是上述参数的值
        NameAttrList func = 10;
        
        //placeholder仅在函数内部的节点中使用，它意味着，这个属性值直到函数被初始化时才会提供。例如，我们假设函数FN中有一个节点N，节点N拥有属性A，A的属性值是"foo"，如果FN在初始化时将foo设置为bar，那么N节点的A属性的值也已被设置为bar
        string placeholder = 9;
    }
}
message NameAttrList {
    string name = 1;
    map<string, AttrValue> attr = 2;
}
```
可见，操作属性的可取值类型很丰富，可以是字符串、整型、浮点型、布尔型、元类型（指代一种数据类型）、张量形状、张量、列表、函数、填充值（placeholder）等等。

attr_value_util中包含了一些方便对参数值进行设置和表示的辅助函数。

## bfloat16
Google发现用16位精度的浮点数，代替32位精度的浮点数，进行神经网络的计算，模型的精度并没有明显下降，但模型大小明显降低，因此TF推出了这种16位的浮点数。

熟悉单精度浮点数表示的朋友应该知道，单精度float的32位表示是由1位符号位+8位阶码+23位尾数组成的。IEEE也提出过一个16位精度的浮点数，但与目前的32位精度浮点数之间的转换比较复杂，因此TF设计了一个新的16位浮点数表示，1位符号位+8位阶码+7位尾数。跟float的符号位和阶码是相同的，仅尾数位数不同，因此方便了相互转换。

## cancellation
在计算图的执行过程中，如果需要临时终止计算图的执行，比如发现输入填写错误，或者编码错误，计算图并不会马上停下来，因为一方面有很多异步运算正在进行，另一方面很多计算是在远程设备上执行的，我们必须通知到正在执行的远程设备。这些操作就需要一个实体来负责，这就是CancellationManager。
```
class CancellationManager {
  public:
    //...
    void StartCancel();//运行所有跟当前的取消管理器有关的回调函数
    bool IsCancelled();//当且仅当StartCancel被调用后，返回true
    CancellationToken get_cancellation_token();//在注册和解注册回调函数时，需要用到的令牌
    bool RegisterCallback(CancellationToken token, CancellCallback callback);//在一个token上注册取消回调函数
    bool DeregisterCallback(CancellationToken token);//在一个token上解注册取消回调函数
  private:
    bool is_cancelling_;
    std::atomic_bool is_cancelled_;
    
    mutex mu_;
    Notification cancelled_notification_;
    CancellationToken next_cancellation_token_ GUARDED_BY(mu_);
    gtl::FlatMap<CancellationToken, CancelCallback> callbacks_ GUARDED_BY(mu_);
};
```

## common_shape_fns
包含了一些通用的，形状推断函数中可能会用到的功能函数，比如，卷积运算，怎样通过输入形状，核大小，padding大小，stride大小来判断输出的形状。

## control_flow
控制流是一个大问题，TF官方有一篇文章解释的很好，后续会专门为此写一篇博文。当前需要理解的是，在TF中为了实现控制流（条件、循环等），需要给张量添加一些附加的属性。比如，考虑如下的代码：
```
int fun(int t){
    int s = 0;
    for(int i=1;i<=100;++i){
        s += t;
    }
    return s;
}

int x(){
    return fun(2);
}

int y(){
    return fun(3);
}
```
这段代码如果放到TF计算图中实现，同样的一个变量s，会有多个不同的值，在fun函数内部，不同迭代轮次中s的值是不同的，在x函数中调用，和在y函数中调用，相同轮次s的值也是不同的。为了对不同调用，不同迭代轮次内的s做区分，TF提出了一个结构体：
```
struct FrameAndIter {
    uint64 frame_id = kIllegalFrameId;
    int64 iter_id = kIllegalIterId;
    
    FrameAndIter(){}
    FrameAndIter(uint64 frame, int64 iter){
        frame_id = frame;
        iter_id = iter;
    }
    
    bool operator==(const FrameAndIter& other) const {
        return (frame_id == other.frame_id && iter_id == other.iter_id);
    }
};
```
仔细看过之前博文的朋友已经看出来，这个结构跟执行器中的TaggedNode很像，只不过FrameAndIter针对张量，TaggedNode针对节点，感兴趣的读者可以去回顾下[executor-下](https://www.cnblogs.com/jicanghai/p/9572217.html)。

## cost_graph
在对计算图进行优化时，一个很重要的信息，是对计算图中各节点的计算消耗进行估计。TF专门提出了一个统计计算图消耗（内存消耗、计算时间消耗）的模型，下面看一下它的结构：
```
message CostGraphDef {
    message Node {
        string name = 1;
        string device = 2;
        int32 id = 3;
        
        message InputInfo {
            int32 preceding_node = 1;
            int32 preceding_port = 2;
        }
        repeated InputInfo input_info = 4;
        
        message OutputInfo {
            int64 size = 1;
            int64 alias_input_port = 2;
            TensorShapeProto shape = 3;
            DataType dtype = 4;
        }
        repeated OutputInfo output_info = 5;
        
        int64 temporary_memory_size = 6;//临时内存损耗
        int64 host_temp_memory_size = 10;//host临时内存损耗
        int64 device_temp_memory_size = 11;//device临时内存损耗
        int64 host_persistent_memory_size = 12;//host永久内存损耗
        int64 device_persistent_memory_size = 16;//device永久内存损耗
        
        int64 compute_cost = 9;//该节点计算时长的估计，单位毫秒
        int64 compute_time = 14;//纯计算损耗，不包括内存访问损耗
        int64 memory_time = 15;//内存访问损耗，不包含计算损耗
        bool is_final = 7;//当前节点的输出，是否是整个计算图的输出，如果是，则这个输出不能被抛弃
        
        repeated int32 control_input = 8;//当前节点的控制输入
    }
    repeated Node node = 1;
}
```
可见，对计算图损耗的统计，就是对节点损耗统计的集合。而对于节点，除了基础的属性信息和输入输出信息之外，主要包含了内存消耗和时间消耗的信息。

## fake_input
为了测试NodeDefBuiler的功能，我们需要为节点准备一些输入，但其实大部分功能的测试并不需要真实的数据，我们只需要有一个输入的形式在。因此TF推出了FakeInput结构，它是一个内部使用的结构，具体实现是FakeInputImpl，感兴趣的读者可以去看下源码。

## load_library
当运行时环境初始化的时候，需要把op和kernel的定义载入内存。首次载入时，这些资源会被放入一个全局的数据结构中，后续需要时可以从中检索。
```
Status LoadLibrary(const char* library_filename, void** result, const void** buf, size_t* len){
    static mutex mu;
    static std::unordered_map<string, Library> loaded_libs;
    //...
}
```
也就是说，被载入的库其实被存在一个全局的map中，其中key为库所在的文件名，value为一个Library结构，下面看下它的定义：
```
struct Library {
    void* handle = nullptr;
    OpList op_list;
};
```
即，这里的库实际上包含的是一个指向自身的句柄，以及一个操作的集合。

## log_memory
在程序运行时，我们经常需要分配内存空间，内存的分配通常被分为两种情况，第一是在OpKernel计算时分配内存，这些分配会由一个进程级别的编号（step_id）来标识，第二是各种特殊的内存分配场景，包括：
- 当进行即时的常量折叠优化时；
- 当进行OpKernel的构建时；
- 当使用外部代码，比如C API分配张量时；
- 当为网络传输分配内存时；
- 当为GPU传输过来的proto分配内存时；
- 当调用者并没有指明step_id时；

了解了哪些情况需要记录内存分配，还需要知道，在不同的情况下需要记录哪些信息，为了区分内存分配信息记录的不同场景，TF做出了如下分类：
- 记录普通张量的内存分配，普通张量包括OpKernel计算时申请的内存，以及上述特殊情况；
- 记录普通张量的内存回收；
- 当把某个张量作为输出时，需要记录；
- 原始内存的分配，包括的场景有Eigen内存的分配，内存拷贝；
- 原始内存的回收；

有了这些基本概念，理解内存分配的相关结构就容易了，首先我们看下，TF为5种内存分配记录的情况，设计的proto：
```
//在哪一步进行了内存分配
message MemoryLogStep {
    int64 step_id = 1;//进程级别的步骤id，进程内部相同，进程之间不同
    string handle = 2;//描述当前步输入和输出的句柄
};

//张量内存分配的信息
message MemoryLogTensorAllocation {
    int64 step_id = 1;
    
    //进行内存分配的kernel名称，比如"/affine2/weights/Assign"
    string kernel_name = 2;
    TensorDescription tensor = 3;//分配的张量的细节
};

//张量内存回收的信息
message MemoryLogTensorDeallocation {
    int64 allocation_id = 1;
    string allocator_name = 2;
};

//张量设置为输出的信息
message MemoryLogTensorOutput {
    int64 step_id = 1;
    string kernel_name = 2;
    int32 index = 3;//被设置的输出的索引
    TensorDescription tensor = 4;
};

//原始的内存分配信息
message MemoryLogRawAllocation {
    int64 step_id = 1;
    string operation = 2;
    int64 num_bytes = 3;
    uint64 ptr = 4;
    int64 allocation_id = 5;
    string allocator_name = 6;
};

//原始的内存回收信息
message MemoryLogRawDeallocation {
    int64 step_id = 1;
    string operation = 2;
    int64 allocation_id = 3;
    string allocator_name = 4;
    bool deferred = 5;
};
```
其次，我们看下内存分配的功能类LogMemory：
```
class LogMemory {
  public:
    static bool IsEnabled();
    static void RecordStep(int64 step_id, const string& handle);
    static void RecordTensorAllocation(const string& kernel_name, int64 step_id, const Tensor& tensor);
    static void RecordTensorDeallocation(int64 allocation_id, const string& allocator_name);
    static void RecordTensorOutput(const string& kernel_name, int64 step_id, int index, const Tensor& tensor);
    static void RecordRawAllocation(const string& operation, int64 step_id, size_t num_bytes, void* ptr, Allocator* allocator);
    static void RecordRawDeallocation(const string& operation, int64 step_id, void* ptr, Allocator* allocator, bool deferred);
};
```
可以看到，主要的API基本上跟前面的proto一一对应。
这些API内部如何实现的呢？我们挑一个最简单的来看下：
```
void LogMemory::RecordStep(const int64 step_id, const string& handle){
    MemoryLogStep step;
    step.set_step_id(step_id);
    step.set_handle(handle);
    OutputToLog(step);
}
```
可见，这些API的主要作用就是把输入参数放入相应的proto，然后以日志的形式将这些proto输出，具体的函数如下：
```
template <typename T>
void OutputToLog(const T& proto){
    string type_name = proto.GetTypeName();
    const size_t index = type_name.find_last_of(".");
    if (index != string::npos) type_name = type_name.substr(index + 1);
    LOG(INFO) << LogMemory::kLogMemoryLabel << " " << type_name << " { " << ProtoShortDebugString(proto) << " }";
}
```

## memory_types
提供了一个，根据NodeDef，获取节点输入输出内存类型的函数，接口如下：
```
Status MemoryTypesForNode(const OpRegistryInterface* op_registry, const DeviceType& device_type, const NodeDef& ndef, MemoryTypeVector* input_memory_types, MemoryTypeVector* output_memory_types);
```

## numeric_op
定义了四类最常见的数值操作类型：
- 单输入单输出，输入输出的类型相同，比如自增运算；
- 双输入单输出，相同类型，比如标量加法；
- 输入和输出拥有相同的形状，且输入输出一一对应，比如矩阵元素翻倍运算；
- 输入和输出拥有相同的形状，且两个输入对应一个输出，比如两个矩阵相加运算；

代码中，这四种运算的定义如下：
```
class UnaryOp : public OpKernel;
class BinaryOp : public OpKernel;
template <class T, class CHILD> class UnaryElementWiseOp : public UnaryOp<T>;
template <class T, class CHILD> class BinaryElementWiseOp : public BinaryOp<T>;
```

## numeric_types
定义了常用的数值类型。

## register_types
在定义一个操作的时候，我们往往会提供一个类型参数，但一方面不一定所有的操作都支持所有的数据类型，另一方面，当前的硬件也不一定支持所有的数据类型。因此有必要为操作设计一个可以快速实例化为各数据类型具体操作的宏，也有必要为不同的硬件设计不同的可用宏。

因此，这里的宏分为两类，一类是TF_CALL_float这种针对具体数据类型的具体宏，另一类是TF_CALL_ALL_TYPES这种类组宏，第二类宏通过调用第一类宏来工作。例如：
```
#define TF_CALL_INTEGRAL_TYPES(m) \
  TF_CALL_int64(m) TF_CALL_int32(m) TF_CALL_uint16(m) TF_CALL_int16(m) TF_CALL_uint8(m) TF_CALL_int8(m)
```

## register_types_traits
这个文件的功能是，在由CPU向GPU拷贝数据时，为POD数据类型提供代理类型。什么POD数据类型呢？POD的全程是Plain Old Data，简单来说，一个类或者结构体，在经过二进制拷贝后还能保持数据不变，这就是POD数据类型。在由CPU向GPU拷贝数据时，实际上拷贝的是二进制的数据，因此它们实际的数据类型，在传输时可以忽略掉，直接拷贝二进制数据就好了。

## rendezvous
一个Rendezvous是一个从生产者向消费者传输张量的抽象，它由一个通道映射表组成。每一个通道由一个Rendezvous键唯一标识，这个键由"producer,consumer"组成，生产者和消费者都是TF中的设备。

生产者通过调用Send()函数，将一个张量通过通道传递给消费者，消费者通过调用Recv()函数，从一个通道中接收传递过来的张量。消费者按照生产者生产的顺序接收传输的张量。

消费者可以在生产者将张量生产出来之前或之后，索要这个张量。消费者可以选择进行一个阻塞调用，或者提供一个回调函数，在任何一种情况下，只要张量生产出来，消费者都会第一时间得到它。生产者从不阻塞。

由于比较简单，我们仅列出API的名称，具体签名和实现，大家可以参考源代码：
```
class Rendezvous : public core::RefCounted {
  public:
    static string CreateKey(...);//创建一个传输的键
    static Status ParseKey(...);//解析一个传输的键
    virtual Status Send(...) = 0;
    virtual void RecvAsync(...) = 0;
    Status Recv(...);
    virtual void StartAbort(...) = 0;
}
```

## session_state
这里面包含了两个类，SessionState保存了我们需要在不同运行中保存的张量，比如深度学习需要迭代的训练，每次训练之间需要共享一些数据，就保存在这里。而TensorStore保存了我们在当前运行中需要共享的张量，它被所有的op_kernel共享。
```
class SessionState {
  public:
    Status GetTensor(const string& handle, Tensor* tensor);
    Status AddTensor(const string& handle, const Tensor& tensor);
    Status DeleteTensor(const string& handle);
    int64 GetNewId();
    static const char* kTensorHandleResourceTypeName;
  private:
    mutex state_lock_;
    int64 tensor_id_ = 0;
    std::unordered_map<string, Tensor> tensors_;
};
class TensorStore {
  public:
    struct TensorAndKey {
        Tensor tensor;
        int64 id;
        string device_name;
        string GetHandle(const string& tensor_name){
            return strings::StrCat(tensor_name, ";", id, ";", device_name);
        }
    };
    Status AddTensor(const string& name, const TensorAndKey& tk);
    Status SaveTensors(const std::vector<string>& output_names, SessionState* session_state);
  private:
    mutex lock_;
    std::unordered_map<string, TensorAndKey> tensors_ GUARDED_BY(lock_);
};
```

## step_stats
主要是关于运行时间和内存使用的统计。单步统计由设备统计构成，而设备统计由节点统计构成。下面看下它们的核心结构：
```
message NodeExecStats {
    string node_name = 1;
    int64 all_start_micros = 2;
    int64 op_start_rel_micros = 3;
    int64 op_end_rel_micros = 4;
    int64 all_end_rel_micros = 5;
    repeated AllocatorMemoryUsed memory = 6;
    repeated NodeOutput output = 7;
    string timeline_label = 8;
    int64 scheduled_micros = 9;
    uint32 thread_id = 10;
    repeated AllocationDescription referenced_tensor = 11;
    MemoryStats memory_stats = 12;
};

message DeviceStepStats {
    string device = 1;
    repeated NodeExecStats node_stats = 2;
};

message StepStats {
    repeated DeviceStepStats dev_stats = 1;
};
```
剩余的AllocatorMemoryUsed，NodeOutput，AllocationDescription，MemoryStats，比较简单，大家感兴趣可以直接去看源码。

## summary.proto
用在Tensorboard中，用于显示每个节点元素的汇总信息。

## type_index
c++中的std::type_index提供了RTTI，即运行时类型识别的功能。它本质上包含了一个类型的哈希值，可以作为一个无序映射的键。cppreference上的这段示例代码非常清楚：
```
#include <iostream>
#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <string>
#include <memory>
 
struct A {
    virtual ~A() {}
};
 
struct B : A {};
struct C : A {};
 
int main()
{
    std::unordered_map<std::type_index, std::string> type_names;
 
    type_names[std::type_index(typeid(int))] = "int";
    type_names[std::type_index(typeid(double))] = "double";
    type_names[std::type_index(typeid(A))] = "A";
    type_names[std::type_index(typeid(B))] = "B";
    type_names[std::type_index(typeid(C))] = "C";
 
    int i;
    double d;
    A a;
 
    // 注意我们正在存储指向类型 A 的指针
    std::unique_ptr<A> b(new B);
    std::unique_ptr<A> c(new C);
 
    std::cout << "i is " << type_names[std::type_index(typeid(i))] << '\n';
    std::cout << "d is " << type_names[std::type_index(typeid(d))] << '\n';
    std::cout << "a is " << type_names[std::type_index(typeid(a))] << '\n';
    std::cout << "b is " << type_names[std::type_index(typeid(*b))] << '\n';
    std::cout << "c is " << type_names[std::type_index(typeid(*c))] << '\n';
}
```
这里，std::type_index(typeid(i))实际上是通过std::type_index的构造函数，构造了一个TypeIndex对象。虽然b和c被定义为指向A的指针，但通过typeid我们仍然能辨识出它实际上指向的是什么类型。

这种附加的类型信息是有代价的。在某些平台上，我们希望通过避免掉这种类型信息，来获得更小的二进制存储空间。因此TF提出了一个简化版的TypeIndex类，它模拟了std::type_index的功能，但是没有使用RTTI信息，因此也就不能提供真正的类型信息，只是返回一个标志，表示RTTI已被禁用。类中包含的哈希码对每个类是唯一的，然而它是在运行时产生的，因此这个哈希值被序列化后并没有意义，因为每次运行的时候这个哈希值都不一定相同。

下面我们来看下TF自定义的TypeIndex的实现：
```
class TypeIndex {
  public:
    TypeIndex(const TypeIndex& src) : hash_(src.hash_){}
    //...
  private:
    TypeIndex(const uint64 hash) : hash_(hash){}
    uint64 hash_;
};
```

## types
关于内存类型MemoryType和设备类型DeviceType的定义，还有一些琐碎的信息，详见源码。

## type_traits
定义了一些常用类型判断的模板，主要包含以下四种：
- is_quantized，是否是quantized type；
- is_complex，是否是复数类型；
- is_simple_type，是否是简单类型；
- is_signed，是否是带符号的类型；

## unique_tensor_references
一组唯一的tensor references的集合。在向这个集合中加入tensor的时候会判断，指向这个tensor内部的buffer的引用是否已经存在的，如果已存在，不做任何操作，如果不存在，则插入。

这里做了一个小小的优化，因为这个类中存储的不同的张量引用不会太多，所以一开始可以用一个最大长度为4的内联向量存储这些引用。当不同的张量引用数超过4时，使用一个set来存储。这样在大部分情况下，都能保证较好的插入和查询性能。这种优化方式比较普遍，在TF中也经常被使用。

## variant
一个类型擦除的容器，可用于存储各种数据类型。实现方式与std::any很像，但对于存储的数据类型有限制，只能存储有限几个类型的数据。它能存储的数据必须满足以下的几个条件：
- 这个类是可以复制构造的；
- 类有默认构造函数；
- 它要么是一个protobuf，要么是一个TF的tensor，要么定义了以下的三种函数
```
    string TypeName() const;
    void Encode(VariantTensorData* data) const;
    void Decode(const VariantTensorData& data);
```
使用get<T>函数是获取Variant中存储的内部数据的主要方式。这种方式是类型安全的，如果内部存储的数据类型不是T，将返回null。

Variant对象将序列化和反序列化的工作交给内部的数据类型去做，对于一些未实现序列化和反序列化功能的POD类型（plain old data），TF的tensor类型，以及protobuf类型，TF在文件中提供了一些辅助函数。如果需要在Variant中存放其它数据类型，需要单独提供Encode和Decode实现。

在Variant中存储的数据类型，通常会包含指向其它TF的tensor的引用。为了有效的支持这种情况，TF给序列化的结构提供了明确的结构，也就是说，必须将它们的结构序列化为一个VariantTensorData对象，类结构如下：
```
struct VariantTensorData {
    string type_name;
    string metadata;
    std::vector<Tensor> tensors;
};
```
如果对象内包含指向其它张量的引用，可以把它们包含在tensors中，把其它的元数据内容放入metadata中。

## versions
提供了版本信息检查的功能。