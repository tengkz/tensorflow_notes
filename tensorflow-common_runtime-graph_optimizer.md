## 目录
1. 核心概念
2. graph_optimizer
3. function
4. optimization_registry

## 1. 核心概念
本篇主要讲图的优化迭代器。我们在构建原始图的时候，专注于达到目的，但不会去考虑图的执行效率。如果把图的设计过程比喻为高级语言的编写，那么图的优化过程就相当于，将高级语言编译为机器语言的过程中，为了能够加速进行的编译优化。比如，将相同的常数折叠，将Identity节点去除等等。本节主要用来讨论，跟图优化相关的类和函数。

## 2. graph_optimizer
进行图优化，需要有一个统一的入口，它的输入是图本身，以及图执行的环境，以及优化的配置，输出是优化后的图。这个入口就是GraphOptimizer，我们先来看看它的结构和接口：
```
class GraphOptimizer {
  public:
    GraphOptimizer(const OptimizerOptions& opts);
    void Optimize(FunctionLibraryRuntime* runtime, Env* env, Device* device, std::unique_ptr<Graph>* graph, const std::unordered_map<const Node*, std::vector<PartialTensorShape>>* shape_map);
  private:
    OptimizerOptions opts_;
};
```
显然，其中的Optimize就是这个类最重要的API，它将图优化配置opts中的优化过程应用的graph上。可能会将graph替换为另外一个图对象。device是这张图将要运行的设备，它使得优化算法可以考虑针对设备应当考虑的优化选项。shape_map如果是非空的话，它将图中节点的名称映射为部分可知的节点输出形状，可能在某些图优化中会被应用，比如常量折叠优化。
关于图优化，我们需要了解的更为细致一些，所以，先看一下这个类的构造函数具体的实现方式。
```
GraphOptimizer::GraphOptimizer(const OptimizerOptions& opts) : opts_(opts) {
    if(opts_.opt_level()>=OptimizerOptions::L1){
        opts_.set_do_common_subexpression_elimination(true);
        opts_.set_do_constant_folding(true);
    }
}
```
通过这个函数我们了解到，优化配置是有级别概念的，当级别大于等于1时，某些默认的优化配置需要被开启，比如“公共子项消除”和“常量折叠”。这些内容我们在具体的优化步骤中也会看到。下面就来看一下核心API，Optimize的内容：
```
void GraphOptimizer::Optimize(FunctionLibraryRuntime* runtime, Env* env, Device* device, std::unique_ptr<Graph>* graph, const std::unordered_map<const Node*, std::vector<PartialTensorShape>>* shape_map){
    Graph* g = graph->get();
    DumpGraph("Initial",g);//导出当前图的结构
    
    bool changed = true;
    const int kMaxRounds = 10;
    for(int rounds = 0; rounds < kMaxRounds; ++rounds){
        changed = false;
        if(RemoveListArrayConverter(g)){
            DumpGraph("RemoveListArrayConverter", g);
            changed = true;
        }
        if(opts_.do_function_inlining() && RemoveDeadNodes(g)){
            DumpGraph("RemoveDeadNodes", g);
            changed = true;
        }
        if(opts_.do_function_inlining() && RemoveIdentityNodes(g)){
            DumpGraph("RemoveIdentityNodes", g);
            changed = true;
        }
        if(opts_.do_constant_folding()){
            ConstantFoldingOptions cf_opts;
            cf_opts.shape_map = shape_map;
            bool was_mutated;
            ConstantFold(cf_opts, runtime, env, device, g, &was_mutated).IgnoreError();
            if(was_mutated){
                RemoveDeadNodes(g);
                DumpGraph("ConstFolding",g);
                changed = true;
            }
        }
        if(opts_.do_function_inlining() && FixupSourceAndSinkEdges(g)){
            DumpGraph("FixupSourceAndSinkEdges",g);
            changed = true;
        }
        if(opts_.do_common_subexpression_elimination() && OptimizeCSE(g,nullptr)){
            DumpGraph("ExpandInlineFunctions",g);
            changed = true;
        }
        if(!changed) break;
    }
    
    //由于flib_def永远不会消失，因此我们可以放心的使用它来构建新图
    std::unique_ptr<Graph> copy(new Graph(g->flib_def()));
    CopyGraph(*g, copy.get());
    graph->swap(copy);
    
    DumpGraph("ReCopy", graph->get());
}
```
在对图进行优化时，我们不可能一蹴而就的，因为优化之间会相互影响，比如我们对图进行了A优化，对于A优化来说，此时图已经是最优的了，但之后我们又对图进行了B优化，此时对于B优化来说，图已经是最优的了，但对于A优化来说则未必。因此图优化是一个循环上升的过程，TF设置了最高的优化是10遍，对于大多数图来说，也就足够了。
在图优化的过程中，我们发现了很多之前没见过的函数，这些函数的定义都在function.h文件中，为了加深对于图优化过程的理解，下面我们了解下这个文件中的函数。

## 3. function
function.h文件中，没有类定义，全部都是硬生生的函数定义，干货满满。
```
//kernel生成器，根据FunctionLibraryRuntime和NodeDef来生成kernel
typedef std::function<Status(FunctionLibraryRuntime*, const NodeDef&, std::unique_ptr<OpKernel>*)> CustomKernelCreator;
void RegisterDefaultCustomKernelCreator(CusteomKernelCreator cb);//kernel生成器的注册器

//创建一个FunctionLibraryRuntime，用来实例化lib_def中的函数，并在device上运行，如果custom_kernel_creator是非空的，它会被返回的runtime用来生成kernel
std::unique_ptr<FunctionLibraryRuntime> NewFunctionLibraryRuntime(const DeviceMgr* device_mgr, Env* env, Device* device, int graph_def_version, const FunctionLibraryDefinition* lib_def, const OptimizerOptions& optimizer_options, CusteomKernelCreator custom_kernel_creator);

//与之前的函数类似，只不过返回的runtime直接利用RegisterDefaultCustomKernelCreator注册的全局custom_kernel_creator来生成新的kernel
std::unique_ptr<FunctionLibraryRuntime> NewFunctionLibraryRuntime(const DeviceMgr* device_mgr, Env* env, Device* device, int graph_def_version, const FunctionLibraryDefinition* lib_def, const OptimizerOptions& optimizer_options);

//函数体的内容
struct FunctionBody {
    FunctionDef fdef;
    Graph* graph = nullptr;
    DataTypeVector arg_types;
    DataTypeVector ret_types;
    gtl::InlinedVector<Node*, 4> arg_nodes;
    gtl::InlinedVector<Node*, 4> ret_nodes;
    
    FuntionBody(){}
    FunctionBody(const FunctionDef& f, DataTypeSlice arg_types, DataTypeSlice ret_types, Graph* g);
    ~FunctionBody();
};

//删除以下节点，第一，无状态的，第二，无参数的，第三，对输出无贡献的
bool RemoveDeadNodes(Graph* g);

//寻找如下的模式，src-(in)->node-(out)->dst，如果node是identity节点，in是唯一的输入数据边，out是唯一的输出数据边，则使用src->dst重写以上模式
bool RemoveIdentityNodes(Graph* g);

//将图中的_ListToArray和_ArrayToList转化为Identity节点
bool RemoveListArrayConverter(Graph* g);

//对于图中的每个节点，如果lib指明这个节点是一个函数调用，那么内联这个函数体。如果至少一个节点被内联了，返回true。
bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph);

//将graph中的内容导出到日志文件，如果日志级别足够高的话
void DumpGraph(StringPiece label, const Graph* g);

//应用图重写的优化，例如内联、死节点移除等
void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g);

//将一个函数的图转化为GraphDef
void ToGraphDef(const Graph* g, GraphDef* gdef, bool pretty = false);

//给定一个数值函数，返回它的导数函数
FunctionBody* SymbolicGradient(const FunctionBody& f);

//将一个FunctionDef示例化为一个graph，设置fbody指向拥有FunctionDef的FunctionBody
Status FunctionDefToBodyHelper(const FunctionDef& fdef, const AttrSlice& attrs, const FunctionLibraryDefinition* const lib_def, const std::function<Status(const string&, const OpDef**)>& get_func_sig, FunctionBody** fbody);
```
现在回过头来看GraphOptimizer类中的Optimize函数，首先它把Array和List相互转换节点变为Identity节点，然后删除了死节点，删除Identity节点，进行常量折叠，修复输入输出边，进行公共子项消除，最终完成了对图的优化。

## 4. optimization_registry
optimization_registry.h文件中，包含了一些维护一个全局的图优化遍历注册器所需要的类，在会话初始化一张图时，会使用这个全局优化遍历注册器来对图进行优化。
首先我们来看第一个类，GraphOptimizationPassOptions，顾名思义，它包含了图优化遍历所需要的参数。这些足够作为一个字典的键值，我们通常会使用一个字典来保持各个图优化遍历器的状态。
```
struct GraphOptimizationPassOptions {
    string session_handle;
    const SessionOptions* session_options = nullptr;
    const CostModel* cost_model = nullptr;
    FunctionLibraryDefinition* flib_def = nullptr;
    const DeviceSet* device_set = nullptr;
    //如果优化遍历在图分割之前被使用，那么它优化的对象就是这个graph，如果是图分割之后被使用，那么这个graph是null
    std::unique_ptr<Graph>* graph = nullptr;
    //进行图分割后的优化遍历时使用
    std::unordered_map<string, std::unique_ptr<Graph>* partition_graphs = nullptr;
};
```
图优化遍历，按照在图分割之前还是之后进行，可以分为两类，但我们使用了GraphOptimizationPassOptions这样一个接口。
接下来是GraphOptimizationPass类，所有的图优化遍历类，都是这个类的子类，它的结构也非常简单。
```
class GraphOptimizationPass {
  public:
    virtual ~GraphOptimizationPass() {}
    virtual Status Run(const GraphOptimizationPassOption& options) = 0;
};
```
当我们拥有了多种图优化遍历的算法之后，需要对这些进行统一管理，因此TF提出了一种对图优化遍历算法进行统一注册和管理的类：
```
//这里的键值为phase，图优化遍历算法是按照phase的升序顺序执行的，在一个phase内部，执行顺序是未定义的
typedef std::map<int, std::vector<std::unique_ptr<GraphOptimizationPass>>> GraphOptimizationPasses;

class OptimizationPassRegistry {
  public:
    enum Grouping {
        PRE_PLACEMENT,//在cost model赋值之后，在节点放置算法之前
        POST_PLACEMENT,//在节点放置算法之后
        POST_REWRITE_FOR_EXEC,//在利用feed/fetch节点进行重写之后
        POST_PARTITIONING,//在图分割之后
    };
    void Register(Grouping grouping, int phase, std::unique_ptr<GraphOptimizationPass> pass);//注册图优化遍历算法
    Status RunGrouping(Grouping grouping, const GraphOptimizationPassOptions& options);//运行一个groupping中所有的图优化遍历算法，按照phase的升序运行
    static OptimizationPassRegistry* Global();//返回一个全局的图优化遍历注册器
  private:
    std::map<Grouping, GraphOptimizationPasses> groups_;
};
```
总结一下，groups是一个双层的映射，先从Grouping映射到图优化遍历算法组，这个算法组本身也是个映射，从phase映射到真正的图优化遍历算法，如下：
```
graph LR
Grouping-->GraphOptimizationPasses
phase-->GraphOptimizationPass
```
最后，TF为刚才的注册器提供了一个全局的入口：
```
class OptimizationPassRegistration {
  public:
    OptimizationPassRegistration(OptimizationPassRegistry::Grouping grouping, int phase, std::unique_ptr<GraphOptimizationPass> pass){
        OptimizationPassRegistry::Global->Register(grouping,phase,std::move(pass));
    }
};
```