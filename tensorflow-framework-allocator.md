## 目录
1. 核心概念
2. allocator
    1. Allocator
    2. AllocatorAttributes
    3. AllocationAttributes
    4. AllocatorWrapper
    5. AllocatorStats
3. allocator_registry
    1. AllocatorRegistry

## 1. 核心概念
allocator给出的只是内存分配器的接口，没有给出具体实现。allocator_registry用单例模式实现了一个全局的内存分配器注册类，负责管理和注册所有的内存分配器。

## 2. allocator
### 2.1 Allocator
Allocator是一个内存分配器的接口类，它规定了一个内存分配器需要具有哪些API。具体看代码：
```
class Allocator {
public:
    virtual void* AllocateRaw(size_t alignment, size_t num_bytes) = 0;
    virtual void DeallocateRaw(void* ptr) = 0;
    T* Allocate(size_t num_elements);
    T* Allocate(size_t num_elements, const AllocationAttributes& allocation_attr);
    void Deallocate(T* ptr, size_t num_elements);
    virtual bool TracksAllocationSizes();
    virtual bool ShouldAllocateEmptyTensors();
    virtual size_t RequestedSize(void* ptr);
    virtual size_t AllocatedSize(void* ptr);
    virtual int64 AllocationId(void* ptr);//本次内存分配的编号
    virtual size_t AllocatedSizeSlow(void* ptr);
    virtual void GetStats(AllocatorStats* stats);
}
```
另外，Allocator除了提供申请内存的接口之外，还提供了为申请好的内存调用默认构造和析构函数的接口。如果在申请的时候指定了对象的类型，就可以选择调用对象所属类的构造和析构方法。Allocator提供了针对三种常用类的构造方法，分别是String，ResourceHandle,Variant。
```
class Allocator {
public:
    //...
private:
    void RunCtor(T* p, size_t n);
    virtual void RunStringCtor(string* p, size_t n);
    virtual void RunStringDtor(string* p, size_t n);
    virtual void RunResourceCtor(ResourceHandle* p, size_t n);
    virtual void RunResourceDtor(ResourceHandle* p, size_t n);
    virtual void RunVariantCtor(Variant* p, size_t n);
    virtual void RunVariantDtor(Variant* p, size_t n);
}
```

### 2.2 AllocatorAttributes
不同的设备分配内存的方法并不相同，那是不是各设备只需要实现自身的内存分配器就可以了呢？如果在计算中每个设备只需要用到自己的内存，当然是没有问题的，但在TF中，有些情况下为了效率，GPU也需要用到CPU内存，比如，为了使用DMA给某些设备传送数据，我们仍然需要申请CPU内存。因此，当我们向一个设备索要内存分配器时，需要给它提供一些信息，告诉设备我们想要申请哪种类型的内存，这些信息就存储在AllocatorAttributes类中。
```
struct AllocatorAttributes {
    void set_on_host(bool v);
    bool on_host() const;
    void set_nic_compatible(bool v);
    bool nic_compatible() const;
    void set_gpu_compatible(bool v);
    bool gpu_compatible() const;
    void set_track_sizes(bool v);
    bool track_sizes() const;
    void Merge(AllocatorAttributes other);
    bool IsEqualOrLessRestrictiveThan(const AllocatorAttributes& other);
    uint32 value = 0;//这个数值的高8位被保留为设备相关的设置。各设备的实现可以根据需要自行解析，为这些设备实现的操作也需要正确的解析它
}
```

### 2.3 AllocationAttributes
AllocatorAttributes很容易与另外一个类混淆，那就是AllocationAttributes。后者是为内存分配器的某一次具体的内存分配准备信息的，而前者是为向设备索要合适的内存分配器提供给设备的，使用时机完全不一样。
```
class AllocationAttributes {
    bool no_retry_on_failure = false; //如果首次内存分配失败了，不再尝试。
    bool allocation_will_be_logged = false;//本次内存分配是否会被记录
}
```

### 2.4 AllocatorWrapper
有时候我们想对某个内存分配器进行封装，以便在某个API上实现定制化。这时就需要用到AllocatorWrapper类，它本质上就是对Allocator类的直接封装。

### 2.5 AllocatorStats
为了对某个内存分配器已分配的内存进行统计，TF还设计了一个结构，AllocatorStats。
```
struct AllocatorStats {
    int64 num_allocs;//内存分配次数
    int64 bytes_in_use;//分配的内存中，当前正在使用的大小
    int64 max_bytes_in_use;//使用中的内存大小的峰值
    int64 max_alloc_size;//最大的单次内存分配大小
    int64 bytes_limit;//当前内存分配器能分配的最大内存量，如果申请内存大小超过这个阈值，返回0
    //...
}
```

## 3. allocator_registry
### 3.1 AllocatorRegistry
先看一下，注册器内部是怎样存储内存分配器的：
```
class AllocatorRegistry {
    //...
private:
    std::vector<AllocatorRegistryEntry> allocators_;
    //...
}
```
其实就是把内存分配器存储在一个向量里，但并不是直接存储内存分配器本身，而是对它的一个封装，我们看下这个封装的结构：
```
typedef struct {
    string name;
    int priority;
    Allocator* allocator;
} AllocatorRegistryEntry;
```
除了内存分配器之外，这个entry里还存放了内存分配器的名称和优先级。当我们向AllocatorRegistry请求一个内存分配器时，它返回的是具有最高优先级的分配器，如果多个分配器有相同的优先级，就返回其中的一个。