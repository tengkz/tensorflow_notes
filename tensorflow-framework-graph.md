## 目录
1. 核心概念
2. GraphDef
3. graph_def_util
4. graph_transfer_info

## 1. 核心概念
在讲完了node之后，graph其实就没什么好讲的了。还记得之前我们说过，NodeDef本身就是自带结构的，给定一个图中所有的node，就能把这张图的结构恢复出来，所以graph本身没有添加很多的数据成员，下面我们看下GraphDef的定义：
```
message GraphDef {
    repeated NodeDef node = 1;
    VersionDef versions = 4;
    int32 version = 3 [deprecated = true];
    FunctionDefLibrary library = 2;
};
```
可以看到，这个结构里除了添加了一个函数定义库之外，没有实质性的数据了。
有一点想说一下，我们之前讲完op, node, kernel，发现它们都有一个build构建类，为啥graph没有呢？我的理解是，前三者都是基础的组成元素，像活字印刷里的活字，基础元素的批量产生，因为几乎是重复的动作，所以当然要流程化便捷化，但graph是我们最终的目标，是活字印刷的排版结果，它的生成需要高度的定制化，因此没有相应的build类。

## 2. graph_def_util
前面提到了，graph由于需要高度的定制化，不能批量生产，因此没有相应的build构建类。但TF为了方便构建GraphDef，还是提供了很多辅助函数，下面我们来看下：
```
//生成一个可读性好的关于GraphDef的描述，而不是返回一个可读性差的proto文本
string SummarizeGraphDef(const GraphDef& graph_def);

//校验一个GraphDef
Status ValidateExternalGraphDefSyntax(const GraphDef& graph_def);

//从节点索引node_offset开始，为GraphDef中的节点加入默认参数值，节点对应的op的默认参数值在op_registry中
Status AddDefaultAttrsToGraphDef(GraphDef* graph_def, const OpRegistryInterface& op_registry, int node_offset);

//从GraphDef中除去那些，在producer_op_registry中出现过，但在consumer_op_registry中未出现过的默认参数值
Status RemoveNewDefaultAttrsFromGraphDef(GraphDef* graph_def, const OpRegistryInterface& consumer_op_registry, const OpRegistryInterface& producer_op_registry, std::set<std::pair<string,string>>* op_attr_removed);

//收集图使用的op，以字符串集合的形式返回
void OpsUsedByGraph(const GraphDef& graph_def, std::set<string>* ops_used_in_graph);

//将graph_def中出现过，同时也在op_registry中出现过的op放入stripped_op_list
Status StrippedOpListForGraph(const GraphDef& graph_def, const OpRegistryInterface& op_registry, OpList* stripped_op_list);
```
我们发现，其中有很多是跟图中op的默认参数值相关的函数。这些函数出现的背景是这样的，假设在一个分布式的环境下，master需要workder执行一个子图，但这个图是master产生的，图中操作的默认值使用的是master所在机器的运行时环境中，OpRegistry中注册的操作所包含的默认值，但关键是，workder所在机器使用的运行时环境，跟master可能不一样！比如，master机器及时对TF进行了升级，但workder却没有。而升级之后，master所在运行时的op参数，可能之前没有默认值，现在加上了默认值，或者之前的默认值改成了现在的默认值，这时候，为了让这张子图具有向前兼容特性，即为了让它能够在workder机器上运行，需要对这张子图进行处理，删除仅在master运行时的OpRegistry中出现的op参数默认值。于是就出现了最后的三个函数。

## 3. graph_transfer_info
最后我们来看一下，在图转化中需要用到的信息。关于它的具体用途我还没找到，等找到了再回头补充。
```
message GraphTransferInfo {
    enum Destination {
        NOP = 0;
        HEXAGON = 1;
    }
    message NodeInput {
        int32 node_id = 1;
        int32 output_port = 2;
    }
    message NodeInfo {
        string name = 1;
        int32 node_id = 2;
        string type_name = 3;
        int32 soc_op_id = 4;
        int32 padding_id = 5;
        int32 input_count = 6;
        int32 output_count = 7;
    };
    message ConstNodeInfo {
        string name = 1;
        int32 node_id = 2;
        repeated int64 shape = 3;
        bytes data = 4;
        DataType dtype = 5;
    };
    message NodeInputInfo {
        int32 node_id = 1;
        repeated NodeInput node_input = 2;
    };
    message NodeOutputInfo {
        int32 node_id = 1;
        repeated int32 max_byte_size = 2;
    };
    message GraphInputNodeInfo {
        string name = 1;
        repeated int64 shape = 2;
        DataType dtype = 3;
    };
    message GraphOutputNodeInfo {
        string name = 1;
        repeated int64 shape = 2;
        DataType dtype = 3;
    };
    repeated NodeInfo node_info = 1;
    repeated ConstNodeInfo const_node_info = 2;
    repeated NodeInputInfo node_input_info = 3;
    repeated NodeOutputInfo node_output_info = 4;
    repeated GraphInputNodeInfo graph_input_node_info = 5;
    repeated GraphOutputNodeInfo graph_output_node_info = 6;
    Destination destination = 7;
};
```