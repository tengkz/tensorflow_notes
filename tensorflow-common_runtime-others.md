把common_runtime中剩余的内容，按照文件名排序进行了简单的解析，时间原因写的很仓促，算是占个坑，后续有了新的理解再来补充。

## allocator_retry
有时候内存分配不可能一次完成，为了在内存分配失败时能够不断尝试，TF给出了一个在内存分配多次尝试的接口：
```
class AllocatorRetry {
  public:
    AllocatorRetry();
    void* AllocateRaw(std::function<void*(size_t alignment, size_t num_bytes, bool verbose_failure)> alloc_func, int max_millis_to_wait, size_t alignment, size_t bytes);
    void NotifyDealloc();
  private:
    Env* env_;
    mutex mu_;
    condition_variable memory_returned_;
    
}
```
其中，通过调用alloc_func来获取内存。首次调用时，verbose_failure会被设置为false，如果返回指针为空，那么等待max_millis_to_wait毫秒，之后每当检测到DeallocateRaw()函数被调用时，就尝试重新申请内存，直到内存申请成功，或者到达deadline，如果deadline到了，将verbose_failure设置为true，再尝试一次。

## bfc_allocator
这里就是TF中使用的BFC（best-fit with coalescing）内存分配算法，它是Doug Lea的内存分配算法（dlmalloc）的简化版本。

在图计算的过程中，需要申请和释放大量的内存，从而产生很多内存碎片，如果使用默认的内存分配算法，大量的内存碎片会降低内存使用的效率。因此BFC算法的一个目标就是，对内存分配和回收过程中产生的碎片进行回收。

它的核心数据结构有两个，分别是chunk和bin。chunk代表每一块具体的内存，为了能追踪到任何一块内存的使用状态，用一个双向链表将chunk链接起来。这些chunk有些是已经被使用的，有些是未被使用的。在内存分配时，为了在未被使用的内存中挑选出合适大小的内存块，使用bin结构对指定大小内存进行快速检索。

具体的，每个bin包含了大小处在某个区间内的所有未使用的内存块，它们按照内存由小到大的顺序，形成了一个单向的链表。每个bin维护的内存块的大小都在2^n到2^n+1的范围内，这种结构方便了对未使用内存的搜索、删除和插入。

所有的chunk链接成一个双向的链表，当我们通过bin结构找到了一个合适大小的chunk时，对它的大小进行判断，如果它是我们需要的内存大小的两倍，那么就对该chunk进行split，其中一块返回使用，另外一块插入bin结构，同时对chunk结构进行迭代，恢复双链表结构。如果有内存被回收，那么判断回收的chunk的前驱和后驱是否也有空内存，如果是，那么对两个chunk合并，维护双链表结构，同时将新chunk插入bin结构内。

纯文字表现力有限，后续补张图。

## build_graph_options
在executor中提到，有时候我们并不需要运行完整的图，而只需要运行其中的一部分。因此需要根据给定的输入输出，从完整的图构建需要运行的子图。这个从完整图到子图的过程，就需要参考这个BuildGraphOptions信息，它主要包含了输入和输出的节点。
```
struct BuildGraphOptions {
    std::vector<string> feed_endpoints;
    std::vector<string> fetch_endpoints;
    
    //如果为true，使用FunctionCallFrame结构来应对输入和输出，否则使用Redezvous结构来应对
    bool use_function_convention = false;
}
```

## constant_folding
常量折叠，是TF图优化中的一种手段。简单来说，如果图中的某个节点仅依赖于常数，那么这个节点的值在图计算之前就能确定，因此我们可以在图计算之前，就在CPU设备上将这些节点的值计算出来。

## copy_tensor
通过它，可以注册在设备间进行数据拷贝的函数，比如通过DMA的方式。

## costmodel_manager
用于为对话管理cost model。内部包含了一个从Graph到其对应的CostModel的映射。
```
class CostModelManager {
  public:
    typedef std::unordered_map<const Graph*, CostModel*> CostModelMap;
  private:
    CostModelMap cost_models_ GUARDED_BY(mu_);
}
```

## debugger_state_interface
当我们需要对计算图进行debug的时候，需要在计算图中插入一些以debug为目的的额外节点，然后在需要时输出图的即时信息。这里TF推出了两个结构，一个是DebugGraphDecorator，这是为了对原图进行修改，插入一些debug节点，另一个是DebuggerState，这是为了存储debug信息，并提供辅助结构方便对debug信息进行检索。这两套结构实现了一种并行的结构体系，包含了Interface，Registry，Factory三种内容。其中Factory是一个函数，用于产生Interface，而Registry是用来管理Factory的注册器。对于DebuggerState和DebugGraphDecorator来说，它们的注册器都只有一个静态的注册对象。
```
class DebuggerStateRegistry {
  public:
    //...
  private:
    static DebuggerStateFactory* factory_;
};
class DebugGraphDecoratorRegistry {
  public:
    //...
  private:
    static DebugGraphDecoratorFactory* factory_;
}
```
它们的结构图如下：
```
graph TB
    DebuggerStateRegistry-->|包含|DebuggerStateFactory
    DebuggerStateFactory-->|产生|DebuggerStateInterface
    DebugGraphDecoratorRegistry-->|包含|DebugGraphDecoratorFactory
    DebugGraphDecoratorFactory-->|产生|DebugGraphDecoratorInterface
```

## dma_helper
仅TF内部使用的一些DMA辅助函数。

## eigen_thread_pool
线程池，它的作用是调度运行一个函数，它本质上是对CPU的一个抽象。Eigen库中包含了一个线程池接口，这里TF设计了一个对Eigen中线程池结构的封装类，EigenThreadPoolWrapper，结构如下：
```
class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface {
  public:
    void Schedule(std::function<void()> fn) override {
        pool_->Schedule(std::move(fn));
    }
    int NumThreads() const override { return pool_->NumThreads(); }
    int CurrentThreadId() const override { return pool_->CurrentThreadId(); }
  private:
    thread::ThreadPool* pool_ = nullptr;
};
```

## gpu/gpu_device
这里主要引入了两个类：
```
graph TB
    LocalDevice-->|派生|BaseGPUDevice
    DeviceFactory-->|派生|BaseGPUDeviceFactory
```

## gpu_device_context
GPU的上下文，基于DeviceContext类，包含了各种stream作为私有数据成员：
```
class GPUDeviceContext : public DeviceContext {
  public:
    //...
  private:
    int stream_id_;
    //当前上下文使用的默认主stream，所有的内存都属于这个stream
    gpu::Stream* stream_;
    //从host复制数据到GPU的stream
    gpu::Stream* host_to_device_stream_;
    //从GPU到host复制数据的stream
    gpu::Stream* device_to_host_stream_;
    //在GPU之间拷贝数据的stream
    gpu::Stream* device_to_device_stream_;
};
```
目前我们把stream理解为在设备之间拷贝数据的功能体，后续有了新的理解再过来补充。

## graph_runner
输入一张图，指定输入和输出，GraphRunner可以执行这张图，计算得到输出。这个类仅供内部使用，它被用来局部的计算图中的非复杂节点，比如形状推断或者常量折叠。由于它的局限性，它所有的计算都将在CPU上进行，并且不具备轻量级、快速和高效等特点。
```
class GraphRunner {
  public:
    GraphRunner(Env* env);
    typedef std::vector<std::pair<string, Tensor>> NamedTensorList;
    Status Run(Graph* graph, FunctionLibraryRuntime* function_library, const NamedTensorList& inputs, const std::vector<string>& output_names, std::vector<Tensor>* outputs);
  private:
    std::unique_ptr<Device> cpu_device_;
};
```

## local_device
LocalDevice类被ThreadPoolDevice和GPUDevice共享，它初始化了一个共享的Eigen计算设备，被两者共用。这个类最终将会被删除，我们会将ThreadPoolDevice和GPUDevice重构为进程级别的抽象。
```
class LocalDevice : public Device {
  public:
    //...
  private:
    static bool use_global_threadpool_;
    static void set_use_global_threadpool(bool use_global_threadpool);
    
    struct EigenThreadPoolInfo;
    std::unique_ptr<EigenThreadPoolInfo> owned_tp_info_;
};
```
（这里转到device篇章下，并添加一个结构图）

## memory_types
内存类型相关的辅助函数。
```
//对于一张仅运行在device_type设备类型上的图g来说，如果它的某个边的源节点或者目的节点包含了非该设备上的内存，则返回错误
Status ValidateMemoryTypes(const DeviceType& device_type, const Graph* g);

//通过插入合适的HostSend/Recv和Send/HostRecv的方式，对图g进行迭代，使得它的每条边的源和目的都与device_type相容
Status EnsureMemoryTypes(const DeviceType& device_type, const string& device_name, Graph* g);

//获取节点n的第index个输出的内存类型
Status MemoryTypeForOutput(const DeviceType& device_type, const Graph* g, const Node* n, int index, MemoryType* memory_type);
```

## mkl_cpu_allocator
一个简单的CPU内存分配器，它简单的将来自MKL库的内存申请/释放请求重定向给TF的BFC内存分配器。

## renamed_device
在特定的情况下，设备的名称会发生变化，TF提出了一种重命名的设备类，来应对这种变化。它用了一个新的名称来包装一个设备，并且将所有的工作代理给包裹的设备。
```
class RenamedDevice : public Device {
  public:
    //...
  private:
    RenamedDevice(Device* underlying, const DeviceAttributes& attributes, bool owns_underlying);
    Device* const underlying_;
    const bool owns_underlying_;
};
```

## rendezvous_mgr
包含了一个新类IntraProcessRendezvous，它表示一种所有的生产者和消费者都在同一个进程内部的Rendezvous，也就是说，在生产者和消费者之间进行通信，不需要RPC。张量值的存储交给了一个局部的Rendezvous代理，这个类只是增加了一些协助进行进程内部设备间数据传输的辅助接口。
```
class IntraProcessRendezvous : public Rendezvous {
  public:
    explicit IntraProcessRendezvous(const DeviceMgr* device_mgr);
    Status Send(...);
    void RecvAsync(...);
  private:
    const DeviceMgr* device_mgr_;
    Rendezvous* local_;
    //...
};
```

## shape_refiner
ShapeRefiner为TF图进行形状推断。它负责为图中的每一个节点实例化一个InferenceContext对象，然后提供/存储input_tensor给形状推断函数使用，这个过程发生在图构建时刻

## simple_graph_execution_state
一个可执行的图，与GraphDef的区别在于，前者的节点是被放置了的，即可执行的图的节点必须有明确的放置设备。SimpleGraphExecutionState类的作用就是，首先按照原图中对节点放置位置的软性要求，对图中的节点进行完全的放置，然后，根据BuildGraphOption结构的内容，从原图中裁切出一个需要运行的子图，这个子图就用SimpleClientGraph表示。用以下的流程图表示：
```
graph TB
    CompleteGraph-->|节点放置算法|PlacedGraph
    PlacedGraph-->|根据BuildGraphOption信息|SimpleClientGraph
```
下面看下主要结构：
```
struct SimpleGraphExecutionStateOptions {
    const DeviceSet* device_set = nullptr;
    const SessionOptions* session_options = nullptr;
    //一个从节点名称到设备名称的映射，代表了不能被改变的节点放置选择
    std::unordered_map<string, string> stateful_placements;
};

//SimpleClientGraph是全图的一个子图，子图的属性通过BuildGraphOptions推断
struct SimpleClientGraph {
    explicit SimpleClientGraph(...);
    //每个客户端图都有一个自己的函数库，因为优化遍历有可能会对图进行重写，从而加入一些新的函数
    std::unique_ptr<FunctionLibraryDefinition> flib_def;
    Graph graph;
    DataTypeVector feed_types;
    DataTypeVector fetch_types;
};

class SimpleGraphExecutionState {
  public:
    //...
  private:
    std::unordered_map<string, string> stateful_placements_;
    Status OptimizeGraph(const BuildGraphOptions& options, std::unique_ptr<Graph>* optimized_graph);
    GraphDef original_graph_def_;
    const DeviceSet* device_set_;
    const SessionOptions* session_options_;
    mutable mutex mu_;
    CostModel costs_ GUARDED_BY(mu_);
    
    //节点名称到全图中的放置位置的映射
    NodeNameToCostIdMap node_name_to_cost_id_map_;
    
    //flib_def_使用原始图中的def初始化，并且随着图优化遍历的进行，可能加入新的函数
    std::unique_ptr<FunctionLibraryDefinition> flib_def_;
    
    //被当前对象拥有的数据流图
    Graph* graph_;
};
```

## simple_placer
一个简单的节点放置算法，在给定图以及可放置的设备之后，确定每个节点放置的设备，考虑的因素如下：
- 已经给定的节点放置约束不能改变；
- 节点需求的（部分需求或者完全需求）的设备放置限制，必须满足；
- 被引用类型的边连接在一起的节点，必须被放置在同一个设备上；
- 给定节点A和B，如果节点B有一个共置的组@loc:A，那么节点A和B必须被放置在一台设备上；
```
class SimplePlacer {
  public:
    typedef std::unordered_map<string, int> NodeNameToIdMap;
    //...
  private:
    Graph* const graph_;
    const DeviceSet* const devices_;
    const SessionOptions* options_;
    const bool log_device_placement_;
};
```

## stats_publisher_interface
StatsPublisherInterface描述了一个发布会话导出信息的对象。当前还处在试验状态。
```
class StatsPublisherInterface {
  public:
    //发布step_stats
    virtual void PublishStatsProto(const StepStats& step_stats) = 0;
    
    //发布每个分割子图的graph_defs
    virtual void PublishGraphProto(const std::vector<const GraphDef*>& graph_defs) = 0;
    
    //基于execution_count和RunOptions为一个给定的step返回一个profile handler
    virtual std::unique_ptr<ProfileHandler> GetProfileHandler(uint64 step, int64 execution_count, const RunOptions& ropts) = 0;
};
```

## step_stats_collector
StepStatsCollector类管理了一个StepStats对象的集合。还记得之前对StepStats的分析吗？StepStats包含了多个DeviceStats，而每个DeviceStats对象又包含了多个NodeExecStats。
```
class StepStatsCollector {
  public:
    void BuildCostModel(CostModelManager* cost_model_manager, const std::unordered_map<string, const Graph*>& device_map);
    void Save(const string& device, NodeExecStats* nt);
    void Swap(StepStats* ss);
  private:
    StepStats* step_stats_ GUARDED_BY(mu_);
    uint64 collectedNodes GUARDED_BY(mu_) = 0;
};
```

## threadpool_device
ThreadPoolDevice类就是CPU设备的实现。
```
class ThreadPoolDevice : public LocalDevice {
  public:
    ThreadPoolDevice(...);
    void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
    Allocator* GetAllocator(AllocatorAttributes attr) override;
    Status MakeTensorFromProto(const TensorProt& tensor_proto, const AllocatorAttributes alloc_attrs, Tensor* tensor) override;
    Status Sync() override { return Status::OK(); }
  private:
    Allocator* allocator_;
};
```

## visitable_allocator
如果一个内存分配器需要对分配的内存进行一些注册/解注册的操作，那么可以让它继承VisitableAllocator，而不是直接继承Allocator，因为前者为注册/解注册提供了接口。
```
class VisitableAllocator : public Allocator {
  public:
    typedef std::function<void(void*, size_t)> Visitor;
    virtual void AddAllocVisitor(Visitor visitor) = 0;
    virtual void AddFreeVisitor(Visitor visitor) = 0;
};
```
还记得之前提过的TrackingAllocator吗？它可以保存每一次分配的内存的信息，如果我们既需要注册/解注册内存，又需要保存分配的内训信息呢？这就需要一个复合的类TrackingVisitableAllocator，这里用到了多重继承这种C++独有的特性，多重继承在这里是可以使用的，因为VisitableAllocator仅是一个虚的接口，只有TrackingAllocator拥有默认的实现。
```
class TrackingVisitableAllocator : public TrackingAllocator, public VisitableAllocator {
  public:
    //...
  protected:
    VisitableAllocator* allocator_;
};
```
（开一个内存分配器篇章，添加一个架构图）