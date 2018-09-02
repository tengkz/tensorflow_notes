## 目录
1. 核心概念
2. session
3. session_factory

## 1. 核心概念
session可以认为是一个执行代理。我们在客户端构建计算图，提供输入，然后把计算图丢给session去执行。因此，session应该具备一定的执行功能。另外TF还提供了session的工厂类，session_factory，用于产生session。

## 2. session
session没有提供头文件声明，直接在session.cc文件中提供了实现，我们略去空实现，如下：
```
Status Session::Run(const RunOptions& run_options, const std::vector<std::pair<string, Tensor>>& inputs, const std::vector<string>& output_tensor_names, const std::vector<string>& target_node_names, std::vector<Tensor>* outputs, RunMetadata* run_metadata);
Status Session::PRunSetup(const std::vector<string>& input_names, const std::vector<string>& output_names, const std::vector<string>& target_nodes, string* handle);
Status Session::PRun(const string& handle, const std::vector<std::pair<string, Tensor>>& inputs, const std::vector<string>& output_names, std::vector<Tensor>* outputs);
Session* NewSession(const SessionOptions& options);
Status NewSession(const SessionOptions& options, Session** out_session);
Status Reset(const SessionOptions& options, const std::vector<string>& containers);
```
可以看到，Session除了提供Run接口之外，还提供了部分执行的接口（PRunSetup和PRun）。部分运行是指，我们并不是运行整张图，而是给定了图中的某几个节点作为输入，某几个节点作为输出，运行部分的图。另外，Session还可以根据提供的SessionOptions产生新的会话。

## 3. session_factory
SessionFactory提供了工厂类的功能，其API如下：
```
class SessionFactory {
  public:
    virtual Session* NewSession(const SessionOptions& options) = 0;
    virtual bool AcceptsOptions(const SessionOptions& options) = 0;
    virtual Status Reset(const SessionOptions& options, const std::vector<string>& containers);
    static void Register(const string& runtime_type, SessionFactory* factory);
    static Status GetFactory(const SessionOptions& options, SessionFactory** out_factory);
}
```
- 对于其中的Reset函数，它的作用是终止和关闭所有现有的session，断开所有的资源与它们的连接。Reset函数能够使那些运行过慢或者运行出错的会话终止和关闭，并且释放与之相关的资源。Reset不会等待旧会话里的计算结束，它会启动一个将计算终止的进程。然而，如果在Reset之后启动了一个新的会话，那么这个新的会话将会与针对旧会话的操作隔离开。
- 如果旧会话的某些资源没有被列入containers，那么旧会话仍然可能在某些地方影响后续会话的计算，而且这种影响很难预测，因此，为了安全尽量在Reset的containers参数中包含尽量全的容器。
- 如果containers向量是空的，那么默认的container将会被使用，如果containers是非空的，那么默认容器需要被显式的放入。
- 支持资源容器的会话，需要重写这个函数。