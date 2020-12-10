
Neo4j 图形数据库
========================


1.Neo4j 介绍
------------------------

1.1 Neo4j 简介
~~~~~~~~~~~~~~~~~~~~~~~~

- Neo4j是什么

    - Neo4j是一个高性能的 NOSQL 图形数据库，它将结构化数据存储在网络上而不是表中。
      它是一个嵌入式的、基于磁盘的、具备完全的事务特性的 Java 持久化引擎，但是它将
      结构化数据存储在网络(从数学角度叫做图)上而不是表中。

    - Neo4j 也可以被看作是一个高性能的图引擎，该引擎具有成熟数据库的所有特性。程序员
      工作在一个面向对象的、灵活的网络结构下而不是严格、静态的表中——但是他们可以享受到
      具备完全的事务特性、企业级的数据库的所有好处。

- Neo4j的特点

    - SQL 就像简单的查询语言 Neo4j CQL
    - 遵循属性图数据模型
    - 通过使用 Apache Lucence 支持索引
    - 支持 UNIQUE 约束
    - 包含一个用于执行 CQL 命令的 UI:Neo4j 数据浏览器
    - 支持完整的 ACID(原子性，一致性，隔离性和持久性)规则
    - 采用原生图形库与本地 GPE(图形处理引擎)
    - 支持查询的数据导出到 JSON 和 XLS 格式
    - 提供了 REST API，可以被任何编程语言 (如Java，Spring，Scala等) 访问
    - 提供了可以通过任何 UI MVC 框架 (如Node JS) 访问的 Java 脚本 
    - 支持两种 Java API: Cypher API 和 Native Java API 来开发 Java 应用程序

- Neo4j的优点

    - 很容易表示连接的数据
    - 检索/遍历/导航更多的连接数据是非常容易和快速的
    - 非常容易地表示半结构化数据
    - Neo4j CQL 查询语言命令是人性化的可读格式，非常容易学习
    - 使用简单而强大的数据模型
    - 不需要复杂的连接来检索连接的/相关的数据，因为它很容易检索它的相邻节点或关系细节没有连接或索引


1.2 Neo4j 概念详解
~~~~~~~~~~~~~~~~~~~~~~~~

Neo4j 图数据库遵循属性图模型来存储和管理其数据。根据属性图模型，关系应该是定向的，
否则，Neo4j 将抛出一个错误消息。基于方向性，Neo4j 关系被分为两种类型：

    - 单向关系

    - 双向关系

1.使用新节点创建关系

    - 示例

        CREATE (e:Employee)-[r:DemoRelation]->(c:Employee)

        CREATE (e:Employee)<-[r:DemoRelation]->(c:Employee)

2.使用已知节点创建带属性的关系

    - 语法

        .. code-block:: 

            MATCH (<node1-label-name>:<node1-name>),(<node2-label-name>:<node2-name>)
            CREATE
                (<node1-label-name>)-[<relationship-label-name>:<relationship-name>{<define-properties-list>}]->(<node2-label-name>)
            RETURN <relationship-label-name>

    - 示例

        MATCH (cust:Customer), (cc.CreditCard)
        CREATE (cust)-[r:DO_SHOPPING_WITH{shopdate:"12/12/2014", price:55000}]->(cc)
        RETURN r

3.检索关系节点的详细信息

    - 语法

        .. code-block:: 

            MATCH (<node1-label-name>)-[<relationship-label-name>:<relationship-name>]->(<node2-label-name>)
            RETURN <relationship-label-name>

    - 示例

        .. code-block:: 

            MATCH (cust)-[r:DO_SHOPPING_WITH]->(cc)
            RETURN cust, cc

1.2 Neo4j CQL
~~~~~~~~~~~~~~~~~~~~~~~~

1.2.1 CQL 简介
^^^^^^^^^^^^^^^^^^^^^^^^

- CQL 代表 Cypher 查询语言.

     - CQL 是 Neo4j 图形数据库的查询语言

     - CQL 是一种声明性模式匹配语言

     - CQL 遵循 SQL 语法

     - CQL 的语法非常简单且人性化、可读性强


1.2.2 CQL 命令关键字
^^^^^^^^^^^^^^^^^^^^^^^^

========= ======================
CQL命令     用法
========= ======================
CREATE     创建节点、关系、属性
MATCH      返回节点、关系、属性数据
RETURN     返回查询结果
WHERE      提供条件过滤检索数据
DELETE     删除节点、关系
REMOVE     删除节点、关系的属性
ORDER BY   排序检索数据
SET        添加或更新标签
========= ======================

1.2.3 CQL 函数
^^^^^^^^^^^^^^^^^^^^^^^^

============== =========================================
定制列表功能      用法
============== =========================================
String          用户使用 String 字面量
Aggregation     用于对 CQL 查询结果执行一些聚合操作
Relationshop    用于获取关系的细节，startnode, endnode等
============== =========================================

1.2.4 CQL 数据类型
^^^^^^^^^^^^^^^^^^^^^^^^

============== =========================================
CQL 数据类型     用法
============== =========================================
boolean         用于表示布尔文字: true, false
byte            用于表示8位整数
short           用于表示16位整数
int             用于表示32位整数
long            用于表示64位整数
float           I用于表示32位浮点数
double          用于表示64位浮点数
char            用于表示16位字符
String          用于表示字符串
============== =========================================

1.2.5 CQL 命令
^^^^^^^^^^^^^^^^^^^^^^^^

1.2.5.1 CREATE
''''''''''''''''''''''''

1.创建没有属性的节点

    - 创建 **节点标签名称**，相当于 MySQL 数据库中的表名，

    - 语法

        .. code-block::

            CREATE (<node-name>:<label-name>)

    - 示例

        .. code-block:: 

            CREATE (emp:Employee)

            CREATE (:Employee)

2.创建具有属性的节点

    - 创建一个具有一些属性(键-值对)的节点来存储数据

    - 语法

        .. code-block::

            CREATE (
                <node-name>:<label-name> {
                    <key>:<value>,
                    <key>:<value>,
                    ...,
                    <key>:<value>,
                }
            )
    
    - 示例

        .. code-block:: 

            CREATE (
                dept:Dept {
                    deptno:10,
                    dname:"Accounting",
                    location:"Hyderabad"
                }
            )

3.创建多个标签的节点

    - 语法

        .. code-block:: 

            CREATE (<node-name>:<label-name1>:<label-name2>...:<label-namen>)

    - 示例

        .. code-block:: 

            CREATE (m:Movie:Cinema:Film:Picture)

1.2.5.2 MATCH
''''''''''''''''''''''''

    - Neo4j CQL ``MATCH`` 命令用于：

        - 从数据库获取有关节点和属性的数据

        - 从数据库获取有关节点，关系和属性的数据

    - 语法

        .. code-block:: 
        
            MATCH (<node-name>:<label-name>)

    - 示例

        .. code-block:: 

            MATCH (dept:Dept) // 会报错

1.2.5.3 RETURN
''''''''''''''''''''''''

    - Neo4j CQL ``RETURN`` 用于:

        - 检索节点的某些属性
        
        - 检索节点的所有属性
        
        - 检索节点和关联关系的某些属性
        
        - 检索节点和关联关系的所有属性

    - 语法

        .. code-block:: 

            RETURN
                <node-name>.<property1-name>,
                <node-name>.<property1-name>,
                ...,
                <node-name>.<property1-name>,

    - 示例

        .. code-block:: 

            MATCH (e:Employee) RETURN e

            MATCH (dept:Dept) RETURN dept.deptno, dept.dname, dept.location

1.2.5.4 WHERE
''''''''''''''''''''''''

    - Neo4j CQL ``WHERE`` 过滤 ``MATCH`` 查询的结果
    
    - 语法

        .. code-block:: 

            WHERE <property-name> <comparison-operator> <value>

        - ``<comparison-operator>``:

            - ``=``
            - ``<>``
            - ``<``
            - ``>``
            - ``<=``
            - ``>=``

        - Neo4j CQL 布尔运算符

            - AND
            - OR
            - NOT
            - XOR

    - 示例

        .. code-block:: 
        
            MATCH (emp:Employee)
            WHERE emp.name = 'Abc' OR emp.name = 'Xyz'
            RETURN emp

            MATCH (cust:Customer), (cc:CreditCard)
            WHERE cust.id = '1001' AND cc.id = '5001'
            CREATE (cust)-[r:DO_SHOPPING_WITH{shopdate:"12/12/2014", price:55000}]->(cc)
            RETURN r

            MATCH p = (m:Bot{id:123})<-[:BotRelation]->(:Bot) RETURN p

1.2.5.5 DELETE
''''''''''''''''''''''''

    - Neo4j 使用 CQL ``DELETE`` 用来:

        - 删除节点

        - 删除节点及相关节点和关系

1.删除节点

    - 语法

        .. code-block:: 

            DELETE <node-name-list>

    - 示例

        .. code-block:: 

            MATCH (e:Employee) DELETE e

2.删除节点和关系

    - 语法

        .. code-block:: 

            DELETE <node-name1>, <node-name2>, <relationship-name>

    - 示例

        .. code-block:: 

            MATCH (cc:CreditCard)-[rel]-(c:Customer)
            DELETE cc, c,rel

1.2.5.6 REMOVE
''''''''''''''''''''''''



1.2.5.7 ORDER BY
''''''''''''''''''''''''

    - Neo4j CQL ``ORDER BY`` 对 MATCH 查询返回的结果进行排序

    - 语法

        ORDER BY <property-name-list> [DESC]

        <node-label-name>.<property1-name>,
        <node-label-name>.<property2-name>, 
        .... 
        <node-label-name>.<propertyn-name> 

    - 示例

        .. code-block:: 

           MATCH (emp:Employee) 
           RETURN emp.empid, emp.name, emp.salary, emp.deptno
           ORDER BY emp.name



1.2.5.8 SET
''''''''''''''''''''''''










2.py2neo
------------------------

- 安装

    .. code-block:: shell

        $ pip install --upgrade py2neo

- 使用

    .. code-block:: python

        from py2neo import Graph

- 核心 API

    - ``Graph`` class

        - ``Subgraph`` class

            - ``Node`` object

            - ``Relationship`` object


2.1 py2neo.database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from py2neo import Graph

    graph = Graph(password = "password")
    graph.run("UNWIND range(1, 3) AS n RETURN n, n * n as n_sq").to_table()

2.1.1 连接
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- GraphService objects

- Graph

    - auto
    - begin
    - call
    - create
    - delete
    - delete_all()
    - evaluate
    - exists
    - match
    - match_one
    - merge
    - name
    - nodes
    - play
    - pull
    - push(subgraph)
    - read(cypher, parameters = None, **kwargs)
    - relationships
    - run(cypher, parameters = None, **kwargs)
    - schema
    - separate
    - service

- SystemGraph objects

- Schema objects

- GraphService objects

- ProcedureLibrary objects

- Procedure objects

    - class py2neo.database.Procedure(graph, name)









- `py2neo <https://py2neo.readthedocs.io/en/latest/>`_ 