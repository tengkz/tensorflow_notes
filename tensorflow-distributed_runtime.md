本篇主要介绍TF的分布式运行时的基本概念。为了对TF的分布式运行机制有一个大致的了解，我们先结合/tensorflow/core/protobuf中的文件给出对TF分布式集群的初步理解，然后介绍/tensorflow/core/distributed_runtime路径下的核心概念。

***

## TF分布式集群
### 集群定义和理解
在研读TF的分布式运行时代码之前，我们需要先看下TF分布式运行的基本架构。TF的集群（cluster）由作业（job）构成，作业由任务（task）构成。举个例子，一个由两个作业构成的集群，作业1名为“worker”，包含了3个任务，作业2名为“ps”，包含了2个任务，如下：
```
Cluster:
    job { name:'worker'
            tasks {key:0 value:'worker1:2222'}
            tasks {key:1 value:'worker2:2222'}
            tasks {key:2 value:'worker3:2222'}
    }
    job { name:'ps'
            tasks {key:0 value:'ps0:2222'}
            tasks {key:1 value:'ps1:2222'}
    }
```
下面再看TF对于集群的定义，就一目了然了：
```
message JobDef {
    string name = 1;//作业的名称
    
    //作业包含的任务id到hostname:port字符串的映射，也就是任务的编号到任务的数据传输接口
    map<int32, string> tasks = 2;
}

message ClusterDef {
    repeated JobDef job = 1;
}
```
以下我们会分别介绍Master服务和Worker服务，注意，Master服务是由Master提供，供客户端使用的，而Worker服务是由Worker提供，供Master使用的。

### master
先来讲Master服务。Master服务是一种被客户端用来与分布式的TF计算交互的服务。

一个Master服务通常会包含了多个master会话，每一个会话包含了一张计算图以及与之相关的状态，这些master会话通常会对应同一个client会话。

一个Master会话的职责包括：
- 节点放置；
- 插入恰当的节点以实现跨设备和跨进程的数据流和资源管理；
- 发布命令给worker，使之运行分配给它的计算子图；

通常，客户端可以通过RPC的形式与一个Master之间保持一个交互式的计算。客户端首先建立一个客户端的会话，连接到一个特定的Master，这个Master接着创建一个对应的Master会话，并且在客户端的调用之间维持状态。

Master会话创建之后，Master会返回一个句柄给客户端，这个句柄可以被用来进行客户端和Master会话之间的交互。

客户端可以在CreateSession调用中传递一个初始的图给Master，并且使用ExtendSession向图中添加节点。

对于一个Master来说，最常用的操作是RunStep，它实现了一个Session::Run()的API。它支持提供输入，执行图计算，返回输出。

最后，当客户端不再需要Master会话的时候，它需要通过CloseSession关闭这个会话，Master可以回收跟会话相关的资源。Master在关闭会话期间可以会因为垃圾回收而休眠一段时间。

我们来总结下MasterService包含的内容：
```
service MasterService {
    rpc CreateSession(CreateSessionRequest) returns (CreateSessionResponse);
    rpc ExtendSession(ExtendSessionRequest) returns (ExtendSessionResponse);
    rpc PartialRunStep(PartialRunStepRequest) returns (PartialRunSetupResponse);
    rpc RunStep(RunStepRequest) returns (RunStepResponse);
    rpc CloseSession(CloseSessionRequest) returns (CloseSessionResponse);
    rpc ListDevices(ListDevicesRequest) returns (ListDeviceResponse);
    rpc Reset(ResetRequest) returns ( ResetResponse);
}
```
代码中提到的xxxRequest和xxxResponse，都有对应的结构，详见/tensorflow/core/protobuf/master.proto。

### woker
Worker服务定义了一种TF的服务，它可以代表MasterService，在一些局部的设备上执行数据流图。

一个Worker服务保留了多个注册图，每一个注册图都是客户端完整图的一个子图，包含了仅需要在当前worker上计算的节点。
```
service WorkerService {
    rpc GetStatus(GetStatusRequest) returns (GetStatusResponse);
    rpc CreateWorkerSession(CreateWorkerSessionRequest) returns (CreateWorkerSessionResponse);
    rpc RegisterGraph(RegisterGraphRequest) returns (RegisterGraphResponse);
    rpc DeregisterGraph(DeregisterGraphRequest) returns (DeregisterGraphResponse);
    rpc RunGraph(RunGraphRequest) returns (RunGraphResponse);
    rpc CleanupGraph(CleanupGraphRequest) returns (CleanupGraphResponse);
    rpc CleanupAll(CleanupAllRequest) returns (CleanupAllResponse);
    rpc RecvTensor(RecvTensorRequest) returns (RecvTensorResponse) {}
    rpc Logging(LoggingRequest) returns (LoggingResponse);
    rpc Tracing(TracingRequest) returns (TracingResponse);
}
```

***
以上内容来自/tensorflow/core/protobuf，主要为了讲解TF中集群的基本概念和运行过程，以下内容来自/tensorflow/core/distributed_runtime，介绍TF中分布式运行时环境中的核心概念。
## worker
Worker代表了执行计算的实体，与Client和Master相对应。以下是相关类的关系图：
```
graph TB
    WorkerCacheInterface-->|用于产生|WorkerInterface
    WorkerCache-->|用于产生|Worker
    WorkerCacheInterface-->|派生|WorkerCache
    WorkerInterface-->|派生|Worker
    WorkerCacheLogger-->|提供日志记录服务|WorkerCache
    Worker-->WorkerEnv
    Worker-->WorkerSession
```

## tensor_coding
包含了TensorResponse类，这个类的作用是，当一个RPC返回了数据时，通过这个类可以把返回结果中的数据解析为张量，以及其它的元数据信息。

## session_mgr
包含了SessionMgr类，它存在于Worker上，为Worker管理会话，包括了会话的产生和销毁，同时还维护了一个当前Worker上的会话句柄到会话的映射。
```
class SessionMgr {
  public:
    Status CreateSession(...);
    Status DeleteSession(...);
  private:
    const WorkerEnv* const worker_env_;
    const WorkerCacheFactory worker_cache_factory_;
    std::map<string, std::unique_ptr<WorkerSession>> sessions_ GUARDED_BY(mu_);
};
```

## server_lib
TF中的server，可以表现为两种形式，一种是Worker，一种是Master，可以认为，两者都是对外提供了“服务”，只不过是两种不同的形式。ServerInterface为它们提供了统一的接口：
```
class ServerInterface {
  public:
    virtual Status Start() = 0;
    virtual Status Stop() = 0;
    virtual Status Join() = 0;
};
```
而所有的Server必须由其对应的工厂类产生，工厂类还提供了对其子类的注册接口：
```
class ServerFactory {
  public:
    virtual Status NewServer(...);
    
    //任何一个工厂类的子类，都必须用这个方法将其一个对象注册到这里
    static void Register(const string& server_type, ServerFactory* factory);
    
    //根据server_def，寻找一个能产生指定server的工厂
    static Status GetFactory(const ServerDef& server_def, ServerFactory** out_factory);
};
```

## scheduler
根据Graph和CostModel的信息，计算不同调度策略下，每个节点的最早开始时间和最晚开始时间，三个类SlackAnalysis，GreedyScheduler，PriorityScheduler分别代表了松弛策略、贪心调度策略和优先级调度策略。

## rendezvous_mgr_interface
类RendezvousMgr管理着一个局部rendezvous对象的集合。所有被当前的Worker发送的张量，在接收之前都在这个RendezvousMgr中保存着。每一个全局的step_id都对应着一个被RendezvousMgr管理的一个局部的rendezvous实例。

## remote_device
包含了一个函数，NewRemoteDevices，它可以发现remote_worker上的可用设备。

## partial_run_mgr
PartialRunMgr保存了未完成的局部运行的需求，它保证只有当对应的执行器完成运行时，它才会被标记为完成。

在TF的worker中，执行器会异步的执行，直到需求的输出（能够返回张量的操作）或者目标（不会返回张量的操作）完成。也就是说，计算图中有两类节点都可以作为worker执行的目标，一类是返回张量的操作对应的节点，一类是不返张量的操作对应的节点。一个局部运行包含两步，第一，设置所有需要的输出和目标，第二，获得输出。在第二步时，可能存在一种情况，即计算图中需求的输出已经计算完成，但需求的目标仍在计算。这时候，PartialRunMgr就发挥作用了，虽然这时理论上可以返回了，因为所有需求的输出都计算完成了，剩余的需求目标并不影响返回的结果。但TF仍然要求必须等到所有的目标都完成计算才行，因为在目标完成计算之前，我们并不知道中间的输出是否会发生变化。

## message_wrappers
在Master和Worker之间相互通信的Request/Response的包装类。
```
// Wrapper classes for the `MasterService.RunStep` request message.
class RunStepRequestWrapper {}
class MutableRunStepRequestWrapper : public RunStepRequestWrapper {}
class InMemoryRunStepRequest : public MutableRunStepRequestWrapper {}
class MutableProtoRunStepRequest : public MutableRunStepRequestWrapper {}
class ProtoRunStepRequest : public RunStepRequestWrapper {}

// Wrapper classes for the `WorkerService.RunGraph` request message.
class RunGraphRequestWrapper {}
class MutableRunGraphRequestWrapper : public RunGraphRequestWrapper {}
class InMemoryRunGraphRequest : public MutableRunGraphRequestWrapper {}
class MutableProtoRunGraphRequest : public MutableRunGraphRequestWrapper {}
class ProtoRunGraphRequest : public RunGraphRequestWrapper {}

// Wrapper classes for the `WorkerService.RunGraph` response message.
class MutableRunGraphResponseWrapper {}
class InMemoryRunGraphResponse : public MutableRunGraphResponseWrapper {}
class OwnedProtoRunGraphResponse : public MutableRunGraphResponseWrapper {}
class NonOwnedProtoRunGraphResponse : public MutableRunGraphResponseWrapper {}

// Wrapper classes for the `MasterService.RunStep` response message.
class MutableRunStepResponseWrapper {}
class InMemoryRunStepResponse : public MutableRunStepResponseWrapper {}
class OwnedProtoRunStepResponse : public MutableRunStepResponseWrapper {}
class NonOwnedProtoRunStepResponse : public MutableRunStepResponseWrapper {}
```

## master_session
与单机情况下的DirectSession对应的，分布式情况下的Master会话，它包含了图计算的基本步骤，比如资源分配、节点放置、图执行等。

## master_interface
用于与TF的Master服务通信的虚拟接口。这个接口既支持基于RPC的master实现，也支持进程内部的master实现。

## master
TF中Master服务的实现。与Worker服务对应。

## master_env
Master的环境类，包含了一个Master所必须的环境资源指针。注意Master并不拥有这些指针。

## local_master
局部Master的实现。局部Master的含义是，与Client的通信不是跨设备的，而是直接在进程内部进行的。这个Master的实现，是为了给同进程内部的Client提供更高效的Master服务。

## graph_mgr
GraphMgr包含了注册到某个worker的图的集合。每一个注册的图都会被一个句柄标识，这个句柄由GraphMgr产生，并且返回给调用者。在注册成功之后，调用者通过一个图句柄来执行一张图。每一次的执行都被一个全局的"step_id"唯一标识。在同一张图上，可以重复和独立的执行多次，只要每一次执行的"step_id"都是不同的。

## call_options
为不同的RPC系统提供了可插拔的调用接口。

## base_rendezvous_mgr
为RendezvousMgrInterface提供了不同的实现，具体框架图如下：
```
graph TB
    RendezvousMgrInterface-->|派生|BaseRendezvousMgr
    RemoteRendezvous-->|派生|BaseRemoteRendezvous
    Rendezvous-->|派生|RemoteRendezvous
    
```

[github地址](https://github.com/tengkz/tensorflow_notes)