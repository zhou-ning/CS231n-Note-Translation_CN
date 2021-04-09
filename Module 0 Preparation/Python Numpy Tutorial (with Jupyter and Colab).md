[toc]

# Python Numpy Tutorial (with Jupyter and Colab)

[![Colab Notebook](Python Numpy Tutorial (with Jupyter and Colab).assets/colab-open.svg)](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb)

本教程最初是由[Justin Johnson](http://cs.stanford.edu/people/jcjohns/)

在本课程中，我们将使用Python编程语言进行所有作业。 Python本身就是一种出色的通用编程语言，但是在一些流行的库（numpy，scipy，matplotlib）的帮助下，Python成为了科学计算的强大环境。

我们希望你们中的许多人都会对Python和numpy有一定的经验；对于其余的人，本节将作为速成课程，介绍Python编程语言及其在科学计算中的用法。我们还将介绍笔记本，这是修补Python代码的一种非常方便的方法。你们中的某些人可能已经掌握了其他语言的知识，在这种情况下，我们还建议您参考：[NumPy for Matlab users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html), [Python for R users](http://www.data-analysis-in-python.org/python_for_r.html), and/or [Python for SAS users](https://nbviewer.jupyter.org/github/RandyBetancourt/PythonForSASUsers/tree/master/).

## Jupyter和Colab笔记本

在深入研究Python之前，我们想简要地介绍一下笔记本。 Jupyter笔记本可让您在Web浏览器中本地编写和执行Python代码。 Jupyter笔记本非常容易修改代码，并一点一点地执行它。因此，它们被广泛用于科学计算。另一方面，Colab是Google的Jupyter笔记本电脑风格，特别适合于机器学习和数据分析，并且完全在云中运行。 Colab基本上是使用类固醇的Jupyter笔记本：它是免费的，无需任何设置，预装了许多软件包，易于与世界共享，并且可以免费使用GPU和TPU等硬件加速器（有一些警告）。

**在Colab中运行教程（推荐）**。如果您希望完全在Colab中运行本教程，请单击此页面顶部的 `Open in Colab`标志。

**在Jupyter Notebook中运行教程**。如果要使用Jupyter在本地运行笔记本，请确保正确安装了虚拟环境（按照设置说明），将其激活，然后运行`pip install Notebook`安装Jupyter笔记本。接下来，打开笔记本并将其下载到您选择的目录，方法是右键单击页面，然后选择“另存为”。然后cd到该目录并运行`jupyter notebook`。

![img](Python Numpy Tutorial (with Jupyter and Colab).assets/file-browser.png)

这将自动启动位于`http://localhost:8888`的笔记本服务器。如果一切正常，您应该会看到一个类似的屏幕，显示当前目录中的所有可用笔记本。单击`jupyter-notebook-tutorial.ipynb`，然后按照笔记本中的说明进行操作。否则，您可以继续阅读下面的代码片段教程。

## python

Python是一种高级的，动态类型化的多范例编程语言。 Python代码通常被称为几乎类似于伪代码，因为它使您可以在很少的代码行中表达非常强大的思想，同时又易于阅读。例如，这是经典的快速排序算法在Python中的实现：

```
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"
```

### Python 版本

从2020年1月1日起，Python正式放弃了对python2的支持。对于此类，所有代码都将使用Python 3.7。在继续本教程之前，请确保您已完成设置说明并正确安装了python3虚拟环境。通过运行python --version激活环境后，可以在命令行中再次检查Python版本。

### 基本数据类型

像大多数语言一样，Python具有许多基本类型，包括整数，浮点数，布尔值和字符串。这些数据类型以其他编程语言所熟悉的方式运行。 

**数字：**整数和浮点数的工作方式与您期望的其他语言一样：

```
x = 3
print(type(x)) # Prints "<class 'int'>"
print(x)       # Prints "3"
print(x + 1)   # Addition; prints "4"
print(x - 1)   # Subtraction; prints "2"
print(x * 2)   # Multiplication; prints "6"
print(x ** 2)  # Exponentiation; prints "9"
x += 1
print(x)  # Prints "4"
x *= 2
print(x)  # Prints "8"
y = 2.5
print(type(y)) # Prints "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"
```

请注意，与许多语言不同，Python没有一元增量（x ++）或减量（x--）运算符。 Python还具有用于复数的内置类型。您可以在[文档](https://docs.python.org/3.5/library/stdtypes.html#numeric-types-int-float-complex)中找到所有详细信息。

**Booleans**：Python实现了布尔逻辑的所有常用运算符，但使用英语单词而不是符号（&&，||等）：

```
t = True
f = False
print(type(t)) # Prints "<class 'bool'>"
print(t and f) # Logical AND; prints "False"
print(t or f)  # Logical OR; prints "True"
print(not t)   # Logical NOT; prints "False"
print(t != f)  # Logical XOR; prints "True"
```

**Strings:** Python对字符串有很好的支持：

```
hello = 'hello'    # String literals can use single quotes
world = "world"    # or double quotes; it does not matter.
print(hello)       # Prints "hello"
print(len(hello))  # String length; prints "5"
hw = hello + ' ' + world  # String concatenation
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"
```

字符串对象有很多有用的方法。例如：

```
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"
```

您可以找到所有字符串方法的列表 [in the documentation](https://docs.python.org/3.5/library/stdtypes.html#string-methods).

### 容器

Python包含几种内置的容器类型：列表，字典，集合和元组。

#### Lists

列表与数组的Python等效，但是列表是可调整大小的，并且可以包含不同类型的元素：

```
xs = [3, 1, 2]    # Create a list
print(xs, xs[2])  # Prints "[3, 1, 2] 2"
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
xs[2] = 'foo'     # Lists can contain elements of different types
print(xs)         # Prints "[3, 1, 'foo']"
xs.append('bar')  # Add a new element to the end of the list
print(xs)         # Prints "[3, 1, 'foo', 'bar']"
x = xs.pop()      # Remove and return the last element of the list
print(x, xs)      # Prints "bar [3, 1, 'foo']"
```

像往常一样，您可以找到有关列表的所有详细信息 [in the documentation](https://docs.python.org/3.5/tutorial/datastructures.html#more-on-lists).

**Slicing:** 除了一次访问一个列表元素外，Python还提供了简洁的语法来访问子列表。这称为切片：

```
nums = list(range(5))     # range is a built-in function that creates a list of integers
print(nums)               # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(nums[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # Assign a new sublist to a slice
print(nums)               # Prints "[0, 1, 8, 9, 4]"
```

我们将在numpy数组的上下文中再次看到切片。

**Loops**：您可以像这样遍历列表的元素：

```
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# Prints "cat", "dog", "monkey", each on its own line.
```

如果要访问循环体内每个元素的索引，请使用内置的枚举函数：

```
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line
```

**List comprehensions:**在编程时，我们经常希望将一种数据类型转换为另一种数据类型。作为一个简单的示例，请考虑以下计算平方数的代码：

```
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # Prints [0, 1, 4, 9, 16]
```

您可以使用列表理解使此代码更简单：

```
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # Prints [0, 1, 4, 9, 16]
```

列表推导也可以包含条件：

```
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"
```

#### Dictionaries

字典存储（键，值）对，类似于Java中的Map或Javascript中的对象。您可以像这样使用它

```
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'     # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del d['fish']         # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```

您可以找到所有需要了解的关于字典的信息 [in the documentation](https://docs.python.org/3.5/library/stdtypes.html#dict).

**Loops:**遍历字典中的键很容易：

```
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

如果要访问键及其相应的值，请使用items方法：

```
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

**Dictionary comprehensions:**这些与列表推导类似，但可让您轻松构造字典。例如：

```
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"
```

#### Sets

集合是不同元素的无序集合。作为一个简单的示例，请考虑以下内容：

```
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
animals.add('fish')       # Add an element to a set
print('fish' in animals)  # Prints "True"
print(len(animals))       # Number of elements in a set; prints "3"
animals.add('cat')        # Adding an element that is already in the set does nothing
print(len(animals))       # Prints "3"
animals.remove('cat')     # Remove an element from a set
print(len(animals))       # Prints "2"
```

像往常一样，可以找到您想知道的关于集的所有信息[in the documentation](https://docs.python.org/3.5/library/stdtypes.html#set).

**Loops:**遍历集合与遍历列表具有相同的语法。但是，由于集合是无序的，因此无法假设访问集合元素的顺序：

```
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: fish", "#2: dog", "#3: cat"
```

**Set comprehensions:**像列表和字典一样，我们可以使用集合推导轻松构建集合：

```
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # Prints "{0, 1, 2, 3, 4, 5}"
```

#### Tuples

元组是（不可变的）有序值列表。元组在很多方面都类似于列表。最重要的区别之一是，元组可以用作字典中的键和集合的元素，而列表则不能。这是一个简单的示例：

```
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"
```

[The documentation](https://docs.python.org/3.5/tutorial/datastructures.html#tuples-and-sequences)有有关元组的更多信息。

### Functions

Python函数是使用`def`关键字定义的。例如：

```
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
# Prints "negative", "zero", "positive"
```

我们通常会定义函数以采用可选的关键字参数，如下所示：

```
def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

hello('Bob') # Prints "Hello, Bob"
hello('Fred', loud=True)  # Prints "HELLO, FRED!"
```

有关Python函数的更多信息 [in the documentation](https://docs.python.org/3.5/tutorial/controlflow.html#defining-functions).

### Classes

在Python中定义类的语法很简单：

```
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```

您可以阅读有关Python类的更多信息 [in the documentation](https://docs.python.org/3.5/tutorial/classes.html).

## Numpy

[Numpy](http://www.numpy.org/) 是Python中科学计算的核心库。它提供了一个高性能的多维数组对象，以及用于处理这些数组的工具。如果您已经熟悉MATLAB，则可能会发现[本教程](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)对于Numpy入门很有用。

### Arrays

numpy数组是所有相同类型的值的网格，并由非负整数的元组索引。维数是数组的等级；数组的形状是一个整数元组，给出沿每个维度的数组大小。

我们可以从嵌套的Python列表中初始化numpy数组，并使用方括号访问元素：

```
import numpy as np

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"
```

Numpy还提供了许多创建数组的功能：

```
import numpy as np

a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
```

您可以阅读有关数组创建的其他方法的信息[in the documentation](http://docs.scipy.org/doc/numpy/user/basics.creation.html#arrays-creation).

### Array indexing

Numpy提供了几种索引数组的方法。

**Slicing:** 与Python列表类似，可以对numpy数组进行切片。由于数组可能是多维的，因此必须为数组的每个维度指定一个切片：

```
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"
```

您也可以将整数索引与切片索引混合使用。但是，这样做将产生比原始数组低级的数组。请注意，这与MATLAB处理数组切片的方式完全不同：

```
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
```

**Integer array indexing:**当您使用切片索引到numpy数组时，生成的数组视图将始终是原始数组的子数组。相反，整数数组索引使您可以使用另一个数组中的数据构造任意数组。这是一个例子：

```
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"
```

整数数组索引的一个有用技巧是从矩阵的每一行中选择或更改一个元素:

```
import numpy as np

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])
```

**Boolean array indexing:**布尔数组索引使您可以挑选出数组中的任意元素。通常，这种类型的索引用于选择满足某些条件的数组元素。这是一个例子：

```
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"
```

为简便起见，我们省略了有关numpy数组索引的许多详细信息；如果您想了解更多，您应该[read the documentation](http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).

### Datatypes

每个numpy数组都是相同类型的元素的网格。 Numpy提供了大量可用于构造数组的数字数据类型。在创建数组时，Numpy会尝试猜测数据类型，但是构造数组的函数通常还包含一个可选参数，以明确指定数据类型。这是一个例子：

```
import numpy as np

x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # Prints "int64"
```

您可以阅读有关numpy数据类型的所有信息 [in the documentation](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html).

### Array math

基本数学函数在数组上按元素进行运算，并且可用作运算符重载和numpy模块中的函数：

```
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
```

请注意，与MATLAB不同，`*`是逐元素乘法，而不是矩阵乘法。相反，我们使用`dot`函数来计算向量的内积，将向量乘以矩阵以及将矩阵相乘。`dot`既可以用作numpy模块中的函数，也可以用作数组对象的实例方法：

```
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))

```

Numpy提供了许多有用的函数来对数组执行计算。最有用的方法之一是`sum`：

```
import numpy as np

x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
```

您可以找到numpy提供的数学函数的完整列表 [in the documentation](http://docs.scipy.org/doc/numpy/reference/routines.math.html)

除了使用数组计算数学函数外，我们经常需要重整形或以其他方式处理数组中的数据。这种操作的最简单的例子是转置矩阵。要转置矩阵，只需使用数组对象的`T`属性：

Numpy提供了更多的操作数组的功能。您可以看到完整列表 [in the documentation](http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html).

### Broadcasting

Broadcasting是一种强大的机制，允许numpy在执行算术运算时使用不同形状的数组。通常，我们有一个较小的数组和一个较大的数组，并且我们想多次使用较小的数组来对较大的数组执行某些操作。

例如，假设我们要向矩阵的每一行添加一个常数向量。我们可以这样做：

```
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)
```

这有效；但是，当矩阵`x`很大时，在Python中计算显式循环可能会很慢。请注意，将向量`v`添加到矩阵`x`的每一行等效于通过垂直堆叠v的多个副本，然后执行`x`和`vv`的元素求和来形成矩阵`vv`。我们可以像这样实现这种方法：

```
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

Numpy广播使我们无需实际创建`v`的多个副本即可执行此计算。请考虑以下版本，使用广播：

```
mport numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

即使`x`由于广播而具有形状`（4，3）`并且`v`具有形状`（3，）`，线`y = x + v`仍有效；这条线的工作方式就像v实际上具有形状`（4，3）`，其中每一行都是`v`的副本，并且总和是逐元素进行的。

Broadcasting 遵循以下规则一起广播两个阵列:

1. 如果数组的秩不同，则将低秩数组的形状加1s，直到两个形状的长度相同。
2. 如果两个数组的尺寸相同，或者其中一个数组的尺寸为1，则称这两个数组在尺寸上兼容。
3. 如果阵列在所有维度上都兼容，则可以一起广播。
4. 广播之后，每个数组的行为就好像它们的形状等于两个输入数组的形状的元素最大值一样。
5. 在一个数组的大小为1而另一个数组的大小大于1的任何维度中，第一个数组的行为就像是沿着该维度复制的一样

如果该说明没有意义，请尝试阅读说明 [from the documentation](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) 或者[this explanation](http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc)

支持广播的功能称为通用功能。您可以找到所有通用功能的列表 [in the documentation](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs).

以下是广播的一些应用：

```
import numpy as np

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(x + v)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((x.T + w).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)
```

广播通常会使您的代码更简洁，更快捷，因此您应尽可能使用它。

### Numpy Documentation

这个简短的概述触及了您需要了解的有关numpy的许多重要知识，但还远远不够。看看[numpy reference](http://docs.scipy.org/doc/numpy/reference/)了解有关numpy的更多信息。

## SciPy

Numpy提供了高性能的多维数组和基本工具，可以使用这些数组进行计算和操作。 [SciPy](https://docs.scipy.org/doc/scipy/reference/)以此为基础，并提供了可在numpy数组上运行的大量功能，可用于不同类型的科学和工程应用程序。

熟悉SciPy的最好方法是浏览[文档](https://docs.scipy.org/doc/scipy/reference/index.html)。我们将重点介绍SciPy的某些部分，这些部分可能对本课程有用。

### Image operations

SciPy提供了一些基本功能来处理图像。例如，它具有从磁盘读取图像到numpy数组，将numpy数组作为图像写入磁盘以及调整图像大小的功能。这是一个简单的示例，展示了这些功能：

```
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)
```

![img](Python Numpy Tutorial (with Jupyter and Colab).assets/cat.jpg) ![img](Python Numpy Tutorial (with Jupyter and Colab).assets/cat_tinted.jpg)

Left: The original image. Right: The tinted and resized image.

### MATLAB files

函数`scipy.io.loadmat`和`scipy.io.savemat`允许您读取和写入MATLAB文件。您可以阅读有关它们的信息 [in the documentation](http://docs.scipy.org/doc/scipy/reference/io.html).

### Distance between points

SciPy定义了一些有用的函数来计算点集之间的距离。

函数`scipy.spatial.distance.pdist`计算给定集中所有点对之间的距离：

```
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print(d)
```

您可以阅读有关此功能的所有详细信息 [in the documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)

类似的函数（scipy.spatial.distance.cdist）计算两组点之间所有对之间的距离；你可以读到它[in the documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).

## Matplotlib

[Matplotlib](http://matplotlib.org/)是一个绘图库。在本节中，将简要介绍`matplotlib.pyplot`模块，该模块提供了类似于MATLAB的绘图系统。

### Plotting

matplotlib中最重要的功能是绘图，它允许您绘制2D数据。这是一个简单的示例：

运行此代码将生成以下图：

![img](Python Numpy Tutorial (with Jupyter and Colab).assets/sine.png)

仅需一点点额外的工作，我们就可以轻松地一次绘制多条线，并添加标题，图例和轴标签：

```

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
```

![img](Python Numpy Tutorial (with Jupyter and Colab).assets/sine_cosine.png)

您可以阅读有关绘图功能的更多信息[in the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot).

### Subplots

您可以使用`subplot`功能在同一图中绘制不同的事物。这是一个例子：

```
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```

![img](Python Numpy Tutorial (with Jupyter and Colab).assets/sine_cosine_subplot.png)

您可以了解有关子图功能的更多信息[in the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot).

### Images

您可以使用imshow功能显示图像。这是一个例子：

```
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
```

![img](Python Numpy Tutorial (with Jupyter and Colab).assets/cat_tinted_imshow.png)

