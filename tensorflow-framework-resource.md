## 目录
1. core/framwork
    1. resource

## 核心概念
资源大多数情况下指的是内存，首先我们来看下相关的序列化结构，ResourceHandleProto
```
message ResourceHandleProto {
    //包含该资源的设备唯一名称
    string device = 1;
    //包含该资源的容器名称
    string container = 2;
    //该资源的唯一名称
    string name = 3;
    //该资源所属类型的唯一哈希值，仅在当前设备和当前执行中有效
    uint64 hash_code = 4;
    //如果可以获取的话，表示当前句柄指向资源的类型，仅调试使用
    string maybe_type_name = 5;
}
```
简单来说，这是一个资源的句柄，它包含了描述这个资源的位置、属性的关键信息。

## resource_handle
细看一下resource_handle.h，就会发现其中的ResrouceHandle类根本就是前面proto的C++实现。之所以要单独再实现一个C++版本的资源句柄，是为了避免让kernels依赖于protos。

## resource_mgr
系统中的大量资源需要被高效的管理，这些资源类型繁多，用处又各有不同，因此TF提出了包含容器的资源管理类——ResourceMgr。三者的关系如下：
```mermaid
graph LR
ResourceMgr-->Container
Container-->Resource
```
看看ResourceMgr类中都包含了哪些私有数据成员：
```
class ResourceMgr {
    //...
private:
    //...
    typedef std::pair<uint64,string> Key;
    typedef std::unordered_map<Key,ResourceBase*,KeyHash,KeyEqual> Container;
    const string default_container_;
    mutable mutext mu_;
    std::unordered_map<string,Container*> containers_ GUARDED_BY(mu_);
    std::unordered_map<uint64,string> debug_type_names GUARDED_BY(mu_);
}
```
其中，容器Container本质上是一个映射，从Key到ResourceBase*的映射，前者包含资源类型的哈希值（想想ResourceHandleProto中的hash_code字段）和资源的名称，而后者就是所有资源的基类对象的指针，这个基类对象我们后面会讲到。资源管理器中最核心的私有数据是containers_，它也是一个映射，把容器的名称映射为容器指针。通过这样一个两层的映射，TF实现了资源管理的功能。
刚才提到了ResourceBase类，我们看一下它的实现：
```
class ResourceBase : public core::RefCounted {
public:
    virtual string DebugString() = 0;
    virtual int64 MemoryUsed() const {return 0;};
};
```
因此，ResourceBase徒有其名，本质上只是一个提供了引用计数功能的对象。资源的使用一定要慎之又慎，提供引用计数功能也是为了方便对资源做回收。
再回到ResourceMgr类，为了透彻理解它的功能，我们再看下它包含的主要接口：
```
class ResourceMgr {
public:
    //在container容器中创建一个名为name的资源
    Status Create(const string& container, const string& name, T* resource);
    //在container中查找一个名为name的资源
    Status Lookup(const string& container, const string& name, T** resource) const;
    //如果container中包含名为name的资源，填充到*resource中，否则，使用creater()创建一个资源
    Status LookupOrCreate(const string& container, const string& name, T** resource, std::function<Status(T**)> creater);
    //删除container中的名为name的资源
    Status Delete(const string& container, const string& name);
    //删除句柄handle指向的资源
    Status Delete(const ResourceHandle& handle);
    //删除container中的所有资源，并删除该container
    Status Cleanup(const string& container);
    //删除所有容器中的所有资源
    void Clear();
}
```
有意思的是，ResourceMgr包含了两个Delete函数，其中一个以容器名称和资源名称为参数，另一个以ResourceHandle为参数，这里资源句柄的作用就很明显了，它可以方便的指代一个资源的位置。
为了更方便的使用ResourceHandle，TF还提供了很多辅助函数，为了节省篇幅，仅列出函数名：
```
//产生ResourceHandle
MakeResourceHandle
MakeResourceHandleToOutput
MakePerStepResourceHandle
HandleFromInput
//根据ResourceHandle查找或构造资源
CreateResource
LookupResource
LookupOrCreateResource
DeleteResource
```
说了这么多，resource到底是什么呢？在TF中，有些kernel是带状态的，在某次执行完成后，它需要保存一些状态信息，方便下次执行的时候时候。这些状态信息就是resource的一种，为了方便管理这些状态信息，才构造了资源相关的类。对于一个kernel来说，一个很自然的问题是，如果它需要将中间状态作为resource保存起来，它需要选择哪个container呢？如果不同kernel的临时状态被随机的存放在资源管理器中，很不方便，因此TF设计了一个类，专门用来辅助kernel，帮它找到一个合适的容器，这个类就是ContainerInfo：
```
class ContainerInfo {
public:
    Status Init(ResourceMgr* rmgr, const NodeDef& ndef, bool use_node_name_as_default);
    //...
private:
    ResourceMgr* rmgr_ = nullptr;
    string container_;
    string name_;
    bool resource_is_private_to_kernel_ = false;
}
```
这个决定的具体过程是这样的：
- 如果container参数非空，就使用它，否则使用资源管理器的默认container
- 如果shared_name参数是非空的，就使用它，否则，如果use_node_name_as_default为真，这个kernel的节点名称被用作资源名称，否则，创建一个专属于当前进程的名称
注意，这里的container和shared_name都是NodeDef的属性值。
于是就有了如下这个，帮助kernel从ctx->resource_manager()中获取资源的函数：
```
Status GetResourceFromContext(OpKernelContext* ctx, const string& input_name, T** resource);
```
另外，resource_mgr.h还包含了如下的内容：
- 一个判断资源是否被初始化的OpKernel
- 一个用来注册op的宏，这个op可以生成某种类型的资源句柄
- 一个用于生成指定资源的句柄的类
- 一个用于注册kernel的宏，这个kernel针对的是可以生成某种类型资源句柄的op